#!/usr/bin/env python3
"""Fast SAM3 text-prompt mask generation (multi-process, streaming, 1 GPU sync / image).

Why this is faster than the original
- N worker processes share each GPU (bs=1 inference is launch/Python bound, so one
  process leaves the H100 mostly idle). Each worker owns a disjoint set of parquet
  row groups -> no duplicated decoding, no giant per-worker dataset copies.
- Streaming with pyarrow (no pandas merge / iterrows over 1M rows).
- JPEG/PNG decode in a thread pool with look-ahead, overlapped with GPU work.
- All keyword masks are combined/thresholded ON the GPU; one sync + one transfer
  per image instead of one per keyword.
- Bounded async saving (no unbounded RAM growth if disk is slower than the GPU).
- Resumable: images whose mask file already exists are skipped.

Run (examples)
  python sam3_masks_fast.py --limit 200 --procs 4          # quick benchmark per worker
  python sam3_masks_fast.py --procs 6                        # full run, 1 GPU
  python sam3_masks_fast.py --gpus 0,1,2,3 --procs 16        # 4 GPUs x 4 workers
Tune --procs by watching `nvidia-smi` (stop increasing when GPU util is ~100%).
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import csv
import glob
import io
import os
import re
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SAM3_PATH = r"/home/susanket/esri-vlm/preprocess/sam3"
KEYWORD_REGEX = re.compile(r"'(.*?)'")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=r"/data/susanket/vlm/images3/image-00.parquet")
    ap.add_argument("--captions", default=r"/data/susanket/vlm/images3/caption-00.parquet")
    ap.add_argument("--out-dir", default=r"/data/susanket/vlm/images3/masks/mask-00")
    ap.add_argument("--out-csv", default=r"/data/susanket/vlm/images3/masks/mask-00.csv")
    ap.add_argument("--name-col", default="file_name")
    ap.add_argument("--image-col", default="image_bytes")
    ap.add_argument("--procs", type=int, default=8, help="Total worker processes (spread over GPUs).")
    ap.add_argument("--gpus", default=None, help="Comma list of GPU ids (default: all visible).")
    ap.add_argument("--decode-threads", type=int, default=3, help="Image decode threads per worker.")
    ap.add_argument("--prefetch", type=int, default=16, help="Decoded images kept ahead per worker.")
    ap.add_argument("--save-threads", type=int, default=3)
    ap.add_argument("--max-pending-saves", type=int, default=32)
    ap.add_argument("--max-keywords", type=int, default=0, help="Cap keywords per image (0 = no cap).")
    ap.add_argument("--packbits", action="store_true",
                    help="Save .npz with bit-packed masks (8x smaller/faster IO). Default: .npy bool like before.")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Stop each worker after N images (benchmarking).")
    return ap.parse_args()


# ------------------------------------------------------------------ data streaming
def load_caption_map(path):
    """name -> raw key_elements string (parsed lazily, only for images we touch)."""
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(path).schema_arrow.names
    name_col = "id" if "id" in cols else "file_name"
    t = pq.read_table(path, columns=[name_col, "key_elements"])
    names = t.column(0).to_pylist()
    kes = t.column(1).to_pylist()
    return dict(zip(names, kes))


def parse_keywords(raw, max_kw):
    s = str(raw)
    if "unclear image" in s.lower():
        return []
    kws = list(dict.fromkeys(k for k in KEYWORD_REGEX.findall(s) if k.strip()))  # dedupe, keep order
    return kws[:max_kw] if max_kw else kws


def stream_samples(args, rank, world, caps, base_done):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.images)
    ngroups = pf.num_row_groups
    by_row = ngroups < world
    if by_row and rank == 0:
        print(f"[warn] image parquet has only {ngroups} row group(s) < {world} workers: workers split by row "
              f"(each reads the whole file). Rewrite with smaller row groups for best speed.", flush=True)
    groups = list(range(ngroups)) if by_row else [i for i in range(ngroups) if i % world == rank]
    row = -1
    for rb in pf.iter_batches(batch_size=64, row_groups=groups, columns=[args.name_col, args.image_col]):
        names = rb.column(0).to_pylist()
        imgs = rb.column(1).to_pylist()
        for n, b in zip(names, imgs):
            if by_row:
                row += 1
                if row % world != rank:
                    continue
            raw = caps.get(n)
            if raw is None:
                continue
            if os.path.splitext(os.path.basename(n))[0] in base_done:
                continue
            kws = parse_keywords(raw, args.max_keywords)
            if not kws:
                continue
            if isinstance(b, dict):
                b = b.get("bytes")
            yield n, b, kws


def decode(item):
    from PIL import Image
    n, b, kws = item
    try:
        return n, Image.open(io.BytesIO(b)).convert("RGB"), kws
    except Exception:
        return n, None, kws


def prefetch(it, fn, workers, depth):
    ex = ThreadPoolExecutor(workers)
    q = deque()
    try:
        for x in it:
            q.append(ex.submit(fn, x))
            if len(q) >= depth:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()
    finally:
        ex.shutdown(wait=False)


# ------------------------------------------------------------------ async saving
class Writer:
    def __init__(self, csv_path, out_dir, threads, max_pending, packbits):
        new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self.f = open(csv_path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["file_name", "key_elements"])
        self.pool = ThreadPoolExecutor(threads)
        self.sem = threading.Semaphore(max_pending)
        self.out_dir, self.packbits, self.n = out_dir, packbits, 0

    def _save(self, path, arr):
        if self.packbits:
            np.savez(path, masks=np.packbits(arr, axis=-1), shape=np.array(arr.shape))
        else:
            np.save(path, arr)

    def save(self, file_name, kws, arr):
        base = os.path.splitext(os.path.basename(file_name))[0]
        path = os.path.join(self.out_dir, base + (".npz" if self.packbits else ".npy"))
        self.sem.acquire()
        fut = self.pool.submit(self._save, path, arr)
        fut.add_done_callback(lambda _f: self.sem.release())
        self.w.writerow([file_name, ", ".join(kws)])
        self.n += 1
        if self.n % 500 == 0:
            self.f.flush()

    def close(self):
        self.pool.shutdown(wait=True)
        self.f.close()


# ------------------------------------------------------------------ GPU work
def segment_image(processor, image, keywords, torch):
    """Returns (valid_keywords, bool ndarray [K,H,W]) or (None, None). One GPU sync pair per image."""
    state = processor.set_image(image)
    combined, names = [], []
    for kw in keywords:
        out = processor.set_text_prompt(state=state, prompt=kw)
        m = out.get("masks", None)
        if m is None or len(m) == 0:
            continue
        if not torch.is_tensor(m):
            m = torch.as_tensor(np.asarray(m), device="cuda")
        m = m.squeeze() > 0.5
        if m.ndim < 2:
            continue
        if m.ndim > 2:
            m = m.flatten(0, m.ndim - 3).any(0)
        combined.append(m)
        names.append(kw)
    if not combined:
        return None, None
    stacked = torch.stack(combined, 0)
    keep = stacked.flatten(1).any(1).cpu().numpy()          # sync 1 (tiny)
    if not keep.any():
        return None, None
    arr = stacked[torch.from_numpy(keep).to(stacked.device)].cpu().numpy()  # sync 2
    return [k for k, f in zip(names, keep) if f], arr


def worker(rank, world, gpu_ids, args):
    import torch
    gpu = gpu_ids[rank % len(gpu_ids)]
    torch.cuda.set_device(gpu)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    sys.path.append(SAM3_PATH)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    os.makedirs(args.out_dir, exist_ok=True)
    caps = load_caption_map(args.captions)
    base_done = set()
    if not args.no_resume:
        ext = "npz" if args.packbits else "npy"
        base_done = {os.path.splitext(os.path.basename(p))[0] for p in glob.iglob(os.path.join(args.out_dir, f"*.{ext}"))}

    model = build_sam3_image_model().to(device=f"cuda:{gpu}").eval()
    processor = Sam3Processor(model)
    part_csv = f"{args.out_csv}.part{rank}"
    writer = Writer(part_csv, args.out_dir, args.save_threads, args.max_pending_saves, args.packbits)

    from tqdm import tqdm
    stream = prefetch(stream_samples(args, rank, world, caps, base_done), decode, args.decode_threads, args.prefetch)
    pbar = tqdm(desc=f"w{rank}/gpu{gpu}", position=rank, mininterval=5)
    done = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for file_name, image, kws in stream:
            if image is None:
                continue
            try:
                valid, arr = segment_image(processor, image, kws, torch)
            except Exception as e:  # keep the run alive on a bad image
                print(f"[w{rank}] {file_name}: {e}", flush=True)
                continue
            if arr is not None:
                writer.save(file_name, valid, arr)
            done += 1
            pbar.update(1)
            if args.limit and done >= args.limit:
                break
    writer.close()
    pbar.close()


def main():
    args = parse_args()
    import torch
    import torch.multiprocessing as mp
    gpu_ids = [int(x) for x in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise SystemExit("No CUDA devices found.")
    world = args.procs
    print(f"{world} workers over GPUs {gpu_ids}", flush=True)
    mp.spawn(lambda_worker, args=(world, gpu_ids, args), nprocs=world, join=True)

    # merge per-worker CSVs
    parts = sorted(glob.glob(args.out_csv + ".part*"))
    mode = "a" if os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0 else "w"
    with open(args.out_csv, mode, newline="", encoding="utf-8") as out:
        if mode == "w":
            out.write("file_name,key_elements\r\n")
        for p in parts:
            with open(p, encoding="utf-8") as f:
                next(f, None)  # header
                out.writelines(f)
            os.remove(p)
    print("Done.")


def lambda_worker(rank, world, gpu_ids, args):  # top-level so spawn can pickle it
    worker(rank, world, gpu_ids, args)


if __name__ == "__main__":
    main()