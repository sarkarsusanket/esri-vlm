#!/usr/bin/env python3
"""Max-throughput DINOv2 [CLS + patch] embedding extraction from parquet image datasets.

For each image*.parquet -> directory `<stem>/` with one `{name}.npy` per sample, shape (257, 384).

Pipeline (everything overlaps):
  reader thread (Arrow -> bytes)  ->  GPU nvJPEG decode + resize + normalize (FP16)
  ->  ViT forward  ->  D2H copy on a side stream  ->  threaded atomic .npy writes

Requires torchvision >= 0.19 for batched GPU JPEG decode (falls back automatically if unavailable).
"""

from __future__ import annotations

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import io
import logging
import queue
import threading
import warnings
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*non-writable.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- data

def load_parquet(parquet_path: Path, image_col: str, name_col: str, existing: set[str]):
    names_in_schema = pq.ParquetFile(parquet_path).schema_arrow.names
    has_names = name_col in names_in_schema
    table = pq.read_table(parquet_path, columns=[image_col] + ([name_col] if has_names else []))
    n = table.num_rows

    col = table.column(image_col)
    if pa.types.is_struct(col.type):
        col = pc.struct_field(col, "bytes")
    images = col.cast(pa.large_binary()).combine_chunks()

    raw = ([Path(str(x)).stem for x in table.column(name_col).to_pylist()]
           if has_names else [f"image_{i}" for i in range(n)])
    counts = Counter(raw)
    dups = sum(c - 1 for c in counts.values() if c > 1)
    if dups:
        logging.warning("%s: %d duplicate file names -> unique '__rowidx' suffix "
                        "(this is what caused overwritten outputs before)", parquet_path.name, dups)
    keys = [k if counts[k] == 1 else f"{k}__{i}" for i, k in enumerate(raw)]
    todo = [i for i, k in enumerate(keys) if k not in existing]
    return images, keys, todo, n


def reader(images, keys, todo, batch_size, q: queue.Queue):
    """Background thread: slice Arrow -> python bytes, push (names, raws)."""
    try:
        for s in range(0, len(todo), batch_size):
            idx = todo[s:s + batch_size]
            raws = images.take(pa.array(idx)).to_pylist()
            q.put(([keys[i] for i in idx], raws))
    finally:
        q.put(None)


def cpu_decode(b: bytes):
    try:
        img = Image.open(io.BytesIO(b))
        img.draft("RGB", (SIZE * 2, SIZE * 2))
        img = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # (3,H,W) uint8
    except Exception:
        return None


def gpu_resize(im: torch.Tensor) -> torch.Tensor:
    return F.interpolate(im.unsqueeze(0).float(), size=(SIZE, SIZE),
                         mode="bilinear", antialias=True, align_corners=False)[0]


def decode_batch(raws, device, cpu_pool):
    """Returns list of (3,224,224) float tensors on `device` (0..255) or None for failures."""
    out = [None] * len(raws)
    is_cuda = device.type == "cuda"
    jpg = [i for i, b in enumerate(raws) if is_cuda and b[:2] == b"\xff\xd8"]

    if jpg:
        datas = [torch.frombuffer(raws[i], dtype=torch.uint8) for i in jpg]
        try:
            decoded = decode_jpeg(datas, device=device, mode=ImageReadMode.RGB)
            for i, im in zip(jpg, decoded):
                out[i] = gpu_resize(im)
        except Exception:
            # batch failed (corrupt file / old torchvision): try one by one, rest falls to CPU
            for i, d in zip(jpg, datas):
                try:
                    out[i] = gpu_resize(decode_jpeg(d, device=device, mode=ImageReadMode.RGB))
                except Exception:
                    out[i] = None

    rest = [i for i in range(len(raws)) if out[i] is None]
    if rest:
        for i, t in zip(rest, cpu_pool.map(cpu_decode, [raws[i] for i in rest])):
            if t is not None:
                out[i] = t.to(device, non_blocking=True).float()
    return out


# ----------------------------------------------------------------------------- io

def write_npy(out_dir: Path, name: str, emb: np.ndarray) -> None:
    tmp = out_dir / f".{name}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, emb, allow_pickle=False)
    os.replace(tmp, out_dir / f"{name}.npy")


# ----------------------------------------------------------------------------- main loop

@torch.inference_mode()
def process_parquet(parquet_path, out_dir, model, device, dtype, mean, std,
                    args, write_pool, cpu_pool, copy_stream) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in out_dir.glob(".*.tmp"):
        t.unlink(missing_ok=True)
    existing = {p.stem for p in out_dir.glob("*.npy")}

    total_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    if len(existing) >= total_rows:
        logging.info("Skipping fully processed: %s", parquet_path.name)
        return

    images, keys, todo, total = load_parquet(parquet_path, args.image_column,
                                             args.file_name_column, existing)
    if not todo:
        logging.info("Nothing left to do in %s", parquet_path.name)
        return

    q: queue.Queue = queue.Queue(maxsize=args.prefetch)
    threading.Thread(target=reader, args=(images, keys, todo, args.batch_size, q),
                     daemon=True).start()

    is_cuda = device.type == "cuda"
    inflight: deque = deque()   # (names, host_tensor, done_event)
    pending: deque = deque()    # write futures
    max_pending = args.writer_threads * 64
    pbar = tqdm(total=len(todo), desc=f"Extracting {parquet_path.name}", unit="img")

    def drain(block_all: bool):
        while inflight and (block_all or len(inflight) > 1):
            names, host, ev = inflight.popleft()
            if ev is not None:
                ev.synchronize()
            arr = host.numpy()
            for nm, e in zip(names, arr):
                pending.append(write_pool.submit(write_npy, out_dir, nm, e))
            while len(pending) > max_pending:
                pending.popleft().result()          # backpressure + surface write errors

    while True:
        item = q.get()
        if item is None:
            break
        names, raws = item
        decoded = decode_batch(raws, device, cpu_pool)
        keep = [i for i, d in enumerate(decoded) if d is not None]
        if len(keep) < len(raws):
            logging.warning("%d undecodable images skipped in this batch", len(raws) - len(keep))
        pbar.update(len(raws))
        if not keep:
            continue

        x = torch.stack([decoded[i] for i in keep]).div_(255)
        x = ((x - mean) / std).to(dtype)
        kept_names = [names[i] for i in keep]

        out = model.forward_features(x)
        emb = torch.cat([out["x_norm_clstoken"].unsqueeze(1), out["x_norm_patchtokens"]], dim=1)
        if args.save_fp32:
            emb = emb.float()

        if is_cuda:
            ev = torch.cuda.Event()
            ev.record()                                   # compute for this batch is queued
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(ev)
                host = torch.empty(emb.shape, dtype=emb.dtype, pin_memory=True)
                host.copy_(emb, non_blocking=True)
                done = torch.cuda.Event()
                done.record(copy_stream)
            emb.record_stream(copy_stream)
            inflight.append((kept_names, host, done))
        else:
            inflight.append((kept_names, emb, None))

        drain(block_all=False)                            # write older batch while GPU runs this one

    drain(block_all=True)
    while pending:
        pending.popleft().result()
    pbar.close()

    n_files = sum(1 for _ in out_dir.glob("*.npy"))
    if n_files < total_rows:
        logging.error("%s: %d files on disk but %d rows", parquet_path.name, n_files, total_rows)
    else:
        logging.info("%s: OK, %d files", parquet_path.name, n_files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--image_column", default="image_bytes")
    ap.add_argument("--file_name_column", default="file_name")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--prefetch", type=int, default=6, help="Batches of raw bytes buffered")
    ap.add_argument("--cpu_threads", type=int, default=min(16, os.cpu_count() or 8),
                    help="CPU decode threads for non-JPEG / fallback images")
    ap.add_argument("--writer_threads", type=int, default=24)
    ap.add_argument("--fp32", action="store_true", help="Run model in FP32 (default FP16)")
    ap.add_argument("--save_fp32", action="store_true", help="Save float32 .npy (default float16)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (extra ~10-30%%)")
    args = ap.parse_args()

    parquet_files = sorted(Path(args.input_dir).glob("**/image*.parquet"))
    if not parquet_files:
        logging.warning("No image*.parquet found in %s", args.input_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if (args.fp32 or device.type == "cpu") else torch.float16
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Loading DINOv2 on %s (%s)...", device, dtype)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device, dtype)
    if args.compile:
        model = torch.compile(model)

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    copy_stream = torch.cuda.Stream() if device.type == "cuda" else None

    with ThreadPoolExecutor(args.writer_threads) as write_pool, \
         ThreadPoolExecutor(args.cpu_threads) as cpu_pool:
        for p in parquet_files:
            process_parquet(p, p.parent / p.stem, model, device, dtype, mean, std,
                            args, write_pool, cpu_pool, copy_stream)

    logging.info("Extraction complete.")


if __name__ == "__main__":
    main()