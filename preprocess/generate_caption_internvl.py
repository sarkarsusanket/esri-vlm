#!/usr/bin/env python3
"""Fastest dense captioning with InternVL3.5 on a single H100 using vLLM.

Why it's fast:
- Continuous batching + paged KV cache (hundreds of sequences in flight)
- Prefix caching (the long shared prompt is computed once)
- Image decode in a thread pool, streamed in chunks (bounded RAM)
- Resumable: progress appended to JSONL, final Parquet/JSONL written at the end
- Failed rows are automatically retried once with light sampling

Install:  pip install vllm pyarrow pandas pillow tqdm
Run:
  python caption_vllm.py --input_parquet in.parquet --output_path out.parquet
"""
from __future__ import annotations

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import io
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FORBIDDEN = (
    "black background", "cannot determine", "cannot see",
    "image processing", "language model", "provided image", "unable to",
)

PROMPT = (
    "<image>\nProvide a detailed and exhaustive dense caption for this image. "
    "Analyze the visual content across background, main foreground subjects, "
    "spatial layouts, textures, colors, and contextual relationships. Keep "
    "in mind that this is an aerial image. The key elements should be a list of objects "
    "present in the image, and visible like 'pool', 'red building', 'black building'...\n"
    "Also don't mention something by inventing, or assuming, unless you see an object.\n"
    "Return only this exact JSON shape: "
    '{"summary":"<one sentence summary>","dense_caption":"<detailed paragraph>",'
    '"key_elements":["<element 1>","<element 2>"]}'
)


def clean_text(v: Any) -> str:
    return " ".join(str(v or "").split()).strip(" \t\r\n\"'")


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    for cand in (text, m.group(0) if m else None):
        if cand:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
    raise ValueError("Failed to parse valid JSON object.")


def parse_dense_caption(raw: str) -> dict[str, Any]:
    obj = parse_json_object(raw)
    summary = clean_text(obj.get("summary", ""))
    caption = clean_text(obj.get("dense_caption", ""))
    elements = obj.get("key_elements", [])
    if not isinstance(elements, list) or len(elements) < 2:
        raise ValueError("key_elements must be a list with at least 2 items")
    if len(summary) < 10 or len(caption) < 30:
        raise ValueError("Caption or summary is too short")
    low = caption.casefold()
    if any(p in low for p in FORBIDDEN):
        raise ValueError("Dense caption contains prohibited meta text")
    return {"summary": summary, "dense_caption": caption,
            "key_elements": [clean_text(e) for e in elements]}


def decode(item: tuple[str, Any]):
    name, raw = item
    try:
        b = raw["bytes"] if isinstance(raw, dict) else raw
        img = Image.open(io.BytesIO(b))
        img.load()
        return name, img.convert("RGB"), None
    except Exception as e:  # noqa: BLE001
        return name, None, str(e)


def iter_chunks(path: str, img_col: str, name_col: str, chunk: int, skip: set[str]):
    pf = pq.ParquetFile(path)
    cols = [img_col] + ([name_col] if name_col in pf.schema_arrow.names else [])
    counter = 0
    for rb in pf.iter_batches(batch_size=chunk, columns=cols):
        imgs = rb.column(img_col).to_pylist()
        names = (rb.column(name_col).to_pylist() if name_col in cols
                 else [f"image_{counter + i}" for i in range(len(imgs))])
        counter += len(imgs)
        pairs = [(n, v) for n, v in zip(names, imgs) if n not in skip]
        if pairs:
            yield pairs


def build_requests(tok, pil_items):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    return [{"prompt": prompt, "multi_modal_data": {"image": img}} for _, img in pil_items]


def run(llm, tok, sp, pil_items):
    outs = llm.generate(build_requests(tok, pil_items), sp, use_tqdm=False)
    res = []
    for (name, _), o in zip(pil_items, outs):
        text = o.outputs[0].text
        try:
            res.append({"file_name": name, "status": "ok", "error": None, **parse_dense_caption(text)})
        except Exception as e:  # noqa: BLE001
            res.append({"file_name": name, "status": "failed", "error": str(e),
                        "summary": "", "dense_caption": "", "key_elements": []})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--image_column", default="image_bytes")
    ap.add_argument("--file_name_column", default="file_name")
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--chunk_size", type=int, default=1024, help="Images submitted to vLLM at once.")
    ap.add_argument("--max_tiles", type=int, default=6,
                    help="Max dynamic tiles per image (12=orig quality, 6~2x faster, 4 fastest).")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--decode_threads", type=int, default=32)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    out = Path(args.output_path)
    progress = out.with_suffix(".progress.jsonl")
    done: dict[str, dict] = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            r = json.loads(line)
            done[r["file_name"]] = r
        logging.info("Resuming: %d rows already done", len(done))

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=32768,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 1},
        # mm_processor_kwargs={"max_dynamic_patch": args.max_tiles},
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    sp_retry = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=args.max_new_tokens + 200)

    total = pq.ParquetFile(args.input_parquet).metadata.num_rows
    pbar = tqdm(total=total, initial=len(done), desc="Captioning")

    with ThreadPoolExecutor(args.decode_threads) as pool, open(progress, "a") as pf:
        # Prefetch-decode next chunk while GPU works on the current one
        chunks = iter_chunks(args.input_parquet, args.image_column,
                             args.file_name_column, args.chunk_size, set(done))
        pending = None
        for pairs in chunks:
            decoded = list(pool.map(decode, pairs))
            ok = [(n, im) for n, im, e in decoded if im is not None]
            bad = [{"file_name": n, "status": "failed", "error": e, "summary": "",
                    "dense_caption": "", "key_elements": []} for n, im, e in decoded if im is None]

            results = run(llm, tok, sp, ok) if ok else []

            # one retry pass for failures (bad JSON / too short)
            imgs = dict(ok)
            retry = [(r["file_name"], imgs[r["file_name"]]) for r in results if r["status"] == "failed"]
            if retry:
                fixed = {r["file_name"]: r for r in run(llm, tok, sp_retry, retry)}
                results = [fixed.get(r["file_name"], r) if r["status"] == "failed" else r for r in results]

            for r in results + bad:
                done[r["file_name"]] = r
                pf.write(json.dumps(r) + "\n")
            pf.flush()
            pbar.update(len(pairs))

    pbar.close()
    final = pd.DataFrame(list(done.values()))
    if str(out).endswith(".parquet"):
        final.to_parquet(out, index=False)
    else:
        final.to_json(out, orient="records", lines=True)
    n_fail = int((final["status"] == "failed").sum())
    logging.info("Done. %d rows (%d failed) -> %s", len(final), n_fail, out)


if __name__ == "__main__":
    main()