#!/usr/bin/env python3
"""Pre-compute Qwen3-Embedding-0.6B EOS text embeddings (1024-d) for caption-XX.parquet files.

For every  <input_dir>/**/caption-XX.parquet  it creates

    caption-XX/
        summary/            <file_name>.npy   shape (1, 1024)        EOS embedding of the one-line summary
        dense_caption/      <file_name>.npy   shape (1 + S, 1024)    row 0 = whole paragraph, rows 1..S = each sentence
        cf_dense_caption/   <file_name>.npy   shape (1 + S, 1024)    same layout (sentence i lines up with dense_caption sentence i)

`--columns summary,dense_caption` computes only those (cf can be added later by re-running with
`--columns cf_dense_caption`; existing .npy files are skipped, so re-runs are cheap/resumable).

"EOS embedding" = last-layer hidden state at the final <|endoftext|> token, which is exactly how
Qwen3-Embedding does last-token pooling (the EOS is appended explicitly, right padding + gather,
so results do not depend on padding side). Use --normalize for unit-norm vectors.

Why it is fast
  * all texts of a chunk are de-duplicated (e.g. a 1-sentence paragraph == its sentence)
  * sorted by length + token-budget dynamic batching (almost no padding), bf16, SDPA/FlashAttn
  * tokenization of chunk k+1 overlaps with GPU work on chunk k
  * async, bounded, atomic .npy writes (tmp -> rename); fp16 storage by default
  * optional multi-GPU (files are sharded across GPUs)

Install:  pip install torch "transformers>=4.51" pyarrow numpy tqdm
Run:
  python precompute_captions.py --input_dir /data/captions
  python precompute_captions.py --input_dir /data/captions --columns summary,dense_caption
  python precompute_captions.py --input_dir /data/captions --columns cf_dense_caption     # later
  python precompute_captions.py --input_dir /data/captions --gpus 0,1,2,3
"""
from __future__ import annotations

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import argparse
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALL_COLS = ("summary", "dense_caption", "cf_dense_caption")
PARAGRAPH_COLS = {"dense_caption", "cf_dense_caption"}
# same splitter as cf_dense_caption_vllm.py so sentence i of the CF caption aligns with sentence i of the original
_SENT_SPLIT = re.compile(r"((?<=[.!?])\s+)")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip())[0::2] if s.strip()]


def texts_for(col: str, text: str) -> list[str]:
    t = text.strip()
    return [t] + split_sentences(t) if col in PARAGRAPH_COLS else [t]


# ----------------------------------------------------------------------------- embedder
class Embedder:
    def __init__(self, model_name: str, device: str, dtype: str = "bfloat16", attn: str = "sdpa",
                 max_length: int = 1024, max_tokens: int = 65536, max_bs: int = 512, normalize: bool = False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype), attn_implementation=attn).to(self.device).eval()
        self.eos = self.tok.eos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos
        self.dim = self.model.config.hidden_size
        self.max_length, self.max_tokens, self.max_bs, self.normalize = max_length, max_tokens, max_bs, normalize
        logging.info("loaded %s on %s (dim=%d, eos_id=%s)", model_name, device, self.dim, self.eos)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        ids = self.tok(texts, add_special_tokens=False, truncation=True, max_length=self.max_length - 1)["input_ids"]
        for x in ids:
            x.append(self.eos)             # explicit EOS: the pooled token
        return ids

    def plan_batches(self, ids: list[list[int]]) -> list[list[int]]:
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        batches, cur = [], []
        for i in order:
            if cur and ((len(cur) + 1) * len(ids[cur[0]]) > self.max_tokens or len(cur) >= self.max_bs):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def embed(self, ids: list[list[int]], batches: list[list[int]]) -> np.ndarray:
        torch = self.torch
        out = torch.empty((len(ids), self.dim), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            for b in batches:
                B, L = len(b), len(ids[b[0]])
                arr = np.full((B, L), self.pad, dtype=np.int64)
                mask = np.zeros((B, L), dtype=np.int64)
                lens = np.empty(B, dtype=np.int64)
                for r, i in enumerate(b):
                    l = len(ids[i])
                    arr[r, :l], mask[r, :l], lens[r] = ids[i], 1, l
                x = torch.from_numpy(arr).to(self.device, non_blocking=True)
                m = torch.from_numpy(mask).to(self.device, non_blocking=True)
                h = self.model(input_ids=x, attention_mask=m).last_hidden_state
                last = torch.from_numpy(lens - 1).to(self.device)
                e = h[torch.arange(B, device=self.device), last].float()      # hidden state at the EOS token
                if self.normalize:
                    e = torch.nn.functional.normalize(e, dim=-1)
                out[torch.tensor(b, device=self.device)] = e
        return out.cpu().numpy()


# ----------------------------------------------------------------------------- async saver
class Saver:
    def __init__(self, threads: int = 4, max_pending: int = 256):
        self.pool = ThreadPoolExecutor(threads)
        self.sem = threading.Semaphore(max_pending)
        self.err: Optional[BaseException] = None

    @staticmethod
    def _write(path: str, arr: np.ndarray) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)                     # atomic -> a crash never leaves a half-written .npy

    def _done(self, fut) -> None:
        self.sem.release()
        if fut.exception() and self.err is None:
            self.err = fut.exception()

    def save(self, path: str, arr: np.ndarray) -> None:
        if self.err:
            raise self.err
        self.sem.acquire()
        self.pool.submit(self._write, path, arr).add_done_callback(self._done)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        if self.err:
            raise self.err


# ----------------------------------------------------------------------------- per-file pipeline
def prepare(emb: Any, tasks: list[tuple[str, str, list[str]]]):
    """tasks: (col, out_name, texts). De-duplicate texts, tokenize, plan batches."""
    uniq: dict[str, int] = {}
    items = []
    for col, name, texts in tasks:
        items.append((col, name, [uniq.setdefault(t, len(uniq)) for t in texts]))
    if not uniq:
        return items, [], []
    ids = emb.tokenize(list(uniq))
    return items, ids, emb.plan_batches(ids)


def process_file(path: Path, root: Path, emb: Any, saver: Saver, a) -> None:
    schema = pq.ParquetFile(path).schema_arrow.names
    name_col = a.name_col if a.name_col in schema else ("id" if "id" in schema else None)
    if name_col is None:
        logging.warning("%s: no '%s' column, skipping", path.name, a.name_col)
        return
    cols = [c for c in a.columns if c in schema]
    for c in a.columns:
        if c not in schema:
            logging.warning("%s: column '%s' not present, skipping that column", path.name, c)
    if not cols:
        return
    table = pq.read_table(path, columns=[name_col] + cols)
    names = table.column(name_col).to_pylist()
    data = {c: table.column(c).to_pylist() for c in cols}
    n = min(len(names), a.limit_rows) if a.limit_rows else len(names)

    parent = (Path(a.output_dir) / path.relative_to(root).parent) if a.output_dir else path.parent
    base = parent / path.stem
    existing: dict[str, set] = {}
    for c in cols:
        (base / c).mkdir(parents=True, exist_ok=True)
        existing[c] = set() if a.overwrite else set(os.listdir(base / c))

    def build(lo: int, hi: int):
        tasks = []
        for i in range(lo, hi):
            fname = os.path.splitext(os.path.basename(str(names[i])))[0] + ".npy"
            for c in cols:
                v = data[c][i]
                if fname in existing[c] or not isinstance(v, str) or not v.strip():
                    continue
                tasks.append((c, fname, texts_for(c, v)))
        return tasks

    dtype = np.float16 if a.dtype == "float16" else np.float32
    starts = list(range(0, n, a.chunk_rows))
    pbar = tqdm(total=n, desc=path.name)
    written = 0
    with ThreadPoolExecutor(1) as prep:        # tokenizes chunk k+1 while the GPU embeds chunk k
        nxt = prep.submit(lambda s=starts[0]: prepare(emb, build(s, min(n, s + a.chunk_rows)))) if starts else None
        for k, s in enumerate(starts):
            items, ids, batches = nxt.result()
            if k + 1 < len(starts):
                s2 = starts[k + 1]
                nxt = prep.submit(lambda s=s2: prepare(emb, build(s, min(n, s + a.chunk_rows))))
            if items:
                vec = emb.embed(ids, batches)
                for col, fname, idxs in items:
                    saver.save(str(base / col / fname), vec[idxs].astype(dtype))
                    written += 1
            pbar.update(min(n, s + a.chunk_rows) - s)
    pbar.close()
    logging.info("%s: wrote %d arrays -> %s", path.name, written, base)


def run_worker(rank: int, world: int, gpu_ids: list[int], files: list[Path], root: Path, a) -> None:
    emb = Embedder(a.model_name, f"cuda:{gpu_ids[rank % len(gpu_ids)]}", a.model_dtype, a.attn,
                   a.max_length, a.max_tokens, a.max_bs, a.normalize)
    saver = Saver(a.save_threads, a.max_pending_saves)
    try:
        for p in files[rank::world]:
            process_file(p, root, emb, saver, a)
    finally:
        saver.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing caption-XX.parquet files.")
    ap.add_argument("--output_dir", default=None, help="Root for caption-XX/ dirs (default: next to each parquet).")
    ap.add_argument("--pattern", default="caption*.parquet")
    ap.add_argument("--columns", default=','.join(ALL_COLS),
                    help="Comma list from summary,dense_caption,cf_dense_caption (default: all that exist).")
    ap.add_argument("--name_col", default="file_name")
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--model_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | eager")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Storage dtype of the .npy files.")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (default: raw EOS hidden state).")
    ap.add_argument("--max_length", type=int, default=1024, help="Max tokens per text (incl. EOS).")
    ap.add_argument("--max_tokens", type=int, default=65536, help="Padded-token budget per GPU batch.")
    ap.add_argument("--max_bs", type=int, default=512)
    ap.add_argument("--chunk_rows", type=int, default=2048, help="Parquet rows embedded per step.")
    ap.add_argument("--save_threads", type=int, default=4)
    ap.add_argument("--max_pending_saves", type=int, default=256)
    ap.add_argument("--gpus", default="0", help="Comma list of GPU ids; files are sharded across them.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit_rows", type=int, default=0, help="Only first N rows per file (quick test).")
    a = ap.parse_args(argv)
    a.columns = [c.strip() for c in a.columns.split(",") if c.strip()]
    bad = [c for c in a.columns if c not in ALL_COLS]
    if bad:
        raise SystemExit(f"unknown columns {bad}; choose from {list(ALL_COLS)}")
    return a


def main() -> None:
    a = parse_args()
    root = Path(a.input_dir)
    files = sorted(p for p in root.glob("**/" + a.pattern) if not p.name.startswith("."))
    if not files:
        logging.warning("no files matching %s in %s", a.pattern, root)
        return
    gpu_ids = [int(x) for x in a.gpus.split(",")]
    world = min(len(gpu_ids), len(files))
    logging.info("%d file(s), columns=%s, %d GPU worker(s)", len(files), a.columns, world)
    if world == 1:
        run_worker(0, 1, gpu_ids, files, root, a)
    else:
        import torch.multiprocessing as mp
        mp.spawn(run_worker, args=(world, gpu_ids, files, root, a), nprocs=world, join=True)
    logging.info("done")


if __name__ == "__main__":
    main()