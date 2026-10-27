#!/usr/bin/env python3
"""Generate SAM3 keyword masks for every `image-*.parquet` / `caption-*.parquet` pair.

This replaces the old single-shard script.  It takes the dataset directory, discovers
all shards, and writes the layout consumed by `src/shards.py`:

    <data-dir>/masks/mask-xx.csv        (file_name,key_elements)
    <data-dir>/masks/mask-xx/<name>.npy (K, H, W) bool, K == len(key_elements)

Speedups over the old per-image implementation
- Batched image encoding: all images in a chunk go through one backbone forward.
- Batched grounding: for each keyword, only the images that contain it are grounded in
  one forward; common keywords like "trees"/"road" are processed many-at-a-time.
- Text features are cached per worker (keywords repeat constantly across the dataset).
- Mask output is capped (`--mask-size`, default 256) which keeps IO tiny.  Region
  coverage in `src/shards.py` rescales any mask resolution, so this is lossless for
  the 16x16 region grid.
- Streaming parquet -> threaded decode -> GPU -> bounded async `.npy` writer.
- Resumable: images already present in the CSV *and* on disk are skipped.  Images with
  no predicted masks get an empty keyword row so they are not retried forever.
- Shards are partitioned across workers; if there are fewer shards than workers, the
  row groups of a shard are split instead.

Run (examples)
    python preprocess/generate_sam_masks.py /data/susanket/vlm/images3
    python preprocess/generate_sam_masks.py /data/susanket/vlm/images3 --gpus 1,2
    python preprocess/generate_sam_masks.py /data/susanket/vlm/images3 --shards 00 --limit 100
    python preprocess/generate_sam_masks.py /data/susanket/vlm/images3 --batch-size 16 --procs 8

Tuning
- One worker per GPU with `--batch-size 32` is a good default on an 80GB H100.
- Lower `--batch-size` if you run several workers per GPU or hit OOM.
- `--mask-size 0` keeps original image resolution (much larger files, slower IO).
- `--resolution 504` makes the backbone faster but may reduce mask quality.
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import io
import os
import re
import sys
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# sam3 is vendored in this repo; do not hard-code an absolute path.
SAM3_PATH = str(Path(__file__).resolve().parent / "sam3")
QUOTED = re.compile(r"'(.*?)'|\"(.*?)\"")


# --------------------------------------------------------------------------- args
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default="/data/susanket/vlm/images3",
                    help="Directory containing image-*.parquet and caption-*.parquet.")
    ap.add_argument("--out-dir", default=None,
                    help="Mask root (default: <data_dir>/masks). Shard files are named mask-xx.csv / mask-xx/.")
    ap.add_argument("--shards", default=None,
                    help="Optional comma list of shard suffixes, e.g. 00,03,07 (default: all).")
    ap.add_argument("--name-col", default="file_name", help="Image parquet file-name column.")
    ap.add_argument("--image-col", default="image_bytes", help="Image parquet image-byte column.")
    ap.add_argument("--procs", type=int, default=0,
                    help="Worker processes (default: one per visible GPU).")
    ap.add_argument("--gpus", default=None,
                    help="Comma list of logical GPU ids (default: all visible).")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Images per GPU chunk; grounding batches keyword groups within it.")
    ap.add_argument("--mask-size", type=int, default=256,
                    help="Longest mask side (0 = keep original image resolution).")
    ap.add_argument("--resolution", type=int, default=1008,
                    help="SAM3 input resolution (default 1008). Smaller is faster.")
    ap.add_argument("--confidence-threshold", type=float, default=0.5)
    ap.add_argument("--min-entropy", type=float, default=7.0,
                    help="Only generate masks for images whose image-parquet 'entropy' is > this "
                         "value (default 7.0; set 0 to disable).")
    ap.add_argument("--read-batch-size", type=int, default=128,
                    help="Rows pulled from parquet per streaming batch.")
    ap.add_argument("--decode-threads", type=int, default=4, help="Image decode threads per worker.")
    ap.add_argument("--prefetch", type=int, default=64, help="Decoded images kept ahead per worker.")
    ap.add_argument("--save-threads", type=int, default=4)
    ap.add_argument("--max-pending-saves", type=int, default=64)
    ap.add_argument("--max-keywords", type=int, default=0, help="Cap keywords per image (0 = no cap).")
    ap.add_argument("--text-cache-size", type=int, default=2048,
                    help="Per-worker keyword text-feature cache entries (0 = disabled).")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--compile", action="store_true",
                    help="Enable torch.compile inside SAM3 (slow first batches).")
    ap.add_argument("--packbits", action="store_true",
                    help="Save bit-packed .npz (smaller, but src/shards.py reads .npy only).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Recompute images even if their CSV row and mask already exist.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop each worker after N images (0 = no limit; useful for benchmarks).")
    return ap.parse_args()


# ------------------------------------------------------------------------ keywords
def _clean_keywords(seq):
    out, seen = [], set()
    for v in seq:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_keywords(raw, max_kw):
    """Parse a key_elements cell (Arrow list or string) into a deduped ordered list."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)) or hasattr(raw, "tolist"):
        seq = raw.tolist() if hasattr(raw, "tolist") else raw
        if any("unclear image" in str(x).lower() for x in seq):
            return []
        kws = _clean_keywords(seq)
    else:
        s = str(raw).strip()
        if not s or "unclear image" in s.lower():
            return []
        kws = []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                kws = _clean_keywords(parsed)
        except (ValueError, SyntaxError):
            pass
        if not kws:
            found = [a or b for a, b in QUOTED.findall(s)]
            kws = _clean_keywords(found if found else s.strip("[]").split(","))
    return kws[:max_kw] if max_kw else kws


# ------------------------------------------------------------------------- shards
def discover_shards(data_dir, only):
    paths = sorted(glob.glob(os.path.join(data_dir, "image-*.parquet")))
    shards = []
    for image_path in paths:
        suffix = os.path.basename(image_path)[len("image-"):-len(".parquet")]
        if only is not None and suffix not in only:
            continue
        caption_path = os.path.join(data_dir, f"caption-{suffix}.parquet")
        if not os.path.exists(caption_path):
            print(f"[warn] {caption_path} missing; skipping shard {suffix}", flush=True)
            continue
        shards.append((suffix, image_path, caption_path))
    return shards


def build_assignments(shards, world):
    """Return, per rank, a list of (suffix, image_path, caption_path, part, num_parts)."""
    n = len(shards)
    if world <= n:
        return [
            [(s[0], s[1], s[2], 0, 1) for i, s in enumerate(shards) if i % world == rank]
            for rank in range(world)
        ]
    base, rem = divmod(world, n)
    per = [[] for _ in range(world)]
    rank = 0
    for i, s in enumerate(shards):
        count = base + (1 if i < rem else 0)
        for part in range(count):
            per[rank].append((s[0], s[1], s[2], part, count))
            rank += 1
    return per


# ---------------------------------------------------------------------- csv helpers
def base_name(name):
    return os.path.splitext(os.path.basename(str(name)))[0]


def _empty_keywords(value):
    s = (value or "").strip()
    return not s or s == "[]" or s.lower() == "none"


def read_csv_rows(path):
    rows = {}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0] == "file_name":
                continue
            rows[row[0]] = row[1] if len(row) > 1 else ""
    return rows


def write_csv_rows(path, rows):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "key_elements"])
        for name, keywords in rows.items():
            w.writerow([name, keywords])
    os.replace(tmp, path)


def merge_csvs(path, extra_paths):
    rows = read_csv_rows(path)
    for p in extra_paths:
        rows.update(read_csv_rows(p))
    write_csv_rows(path, rows)
    for p in extra_paths:
        if os.path.exists(p):
            os.remove(p)


def recover_parts(out_root, suffix):
    """Merge leftover .part* CSVs from an interrupted run into the final CSV."""
    csv_path = os.path.join(out_root, f"mask-{suffix}.csv")
    parts = sorted(glob.glob(csv_path + ".part*"))
    if parts:
        merge_csvs(csv_path, parts)
        print(f"[resume] merged {len(parts)} part CSV(s) for shard {suffix}", flush=True)


def finalize_shard(out_root, suffix):
    csv_path = os.path.join(out_root, f"mask-{suffix}.csv")
    parts = sorted(glob.glob(csv_path + ".part*"))
    if parts:
        merge_csvs(csv_path, parts)
    elif os.path.exists(csv_path):
        # Compact duplicates (possible if a run was killed between CSV write and .npy save).
        merge_csvs(csv_path, [])


# -------------------------------------------------------------------- data streaming
def load_caption_map(path):
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(path).schema_arrow.names
    if "file_name" in schema:
        name_col = "file_name"
    elif "id" in schema:
        name_col = "id"
    else:
        raise KeyError(f"{path}: no file_name/id column")
    table = pq.read_table(path, columns=[name_col, "key_elements"])
    return dict(zip(table.column(0).to_pylist(), table.column(1).to_pylist()))


def build_done(csv_path, mask_dir, ext):
    if not os.path.exists(csv_path):
        return set()
    rows = read_csv_rows(csv_path)
    if not rows:
        return set()
    try:
        existing = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith("." + ext)}
    except FileNotFoundError:
        existing = set()
    done = set()
    for name, keywords in rows.items():
        if _empty_keywords(keywords) or base_name(name) in existing:
            done.add(name)
    return done


def stream_samples(image_path, caption_path, args, done, part, num_parts):
    """Yield (file_name, image_bytes, keywords) for this worker's share of a shard."""
    import pyarrow.parquet as pq

    caps = load_caption_map(caption_path)
    pf = pq.ParquetFile(image_path)
    columns = [args.name_col, args.image_col]
    use_entropy = args.min_entropy is not None and args.min_entropy > 0
    if use_entropy:
        if "entropy" not in pf.schema_arrow.names:
            raise KeyError(f"{image_path}: --min-entropy requires an 'entropy' column")
        columns.append("entropy")
    row_filter = None
    if num_parts > 1 and pf.num_row_groups >= num_parts:
        groups = [g for g in range(pf.num_row_groups) if g % num_parts == part]
    elif num_parts > 1:
        groups = None
        row_filter = (part, num_parts)
        if part == 0:
            print(f"[warn] {os.path.basename(image_path)} has only {pf.num_row_groups} row group(s) "
                  f"for {num_parts} workers; falling back to row-level splitting", flush=True)
    else:
        groups = None

    row = -1
    for rb in pf.iter_batches(batch_size=args.read_batch_size, row_groups=groups,
                              columns=columns):
        names = rb.column(0).to_pylist()
        blobs = rb.column(1).to_pylist()
        entropies = rb.column(2).to_pylist() if use_entropy else None
        for j, (name, blob) in enumerate(zip(names, blobs)):
            if row_filter is not None:
                row += 1
                if row % row_filter[1] != row_filter[0]:
                    continue
            if use_entropy:
                entropy = entropies[j]
                if entropy is None or not np.isfinite(entropy) or float(entropy) <= args.min_entropy:
                    continue
            if name in done:
                continue
            raw = caps.get(name)
            if raw is None:
                continue
            kws = parse_keywords(raw, args.max_keywords)
            if not kws:
                continue
            if isinstance(blob, dict):
                blob = blob.get("bytes")
            yield name, blob, kws


def decode(item):
    from PIL import Image
    name, blob, kws = item
    try:
        with Image.open(io.BytesIO(blob)) as im:
            image = im.convert("RGB")
        return name, image, kws
    except Exception as exc:
        print(f"[decode] {name}: {exc}", flush=True)
        return name, None, kws


def prefetch(iterable, fn, workers, depth):
    pool = ThreadPoolExecutor(workers)
    queue = deque()
    try:
        for item in iterable:
            queue.append(pool.submit(fn, item))
            if len(queue) >= depth:
                yield queue.popleft().result()
        while queue:
            yield queue.popleft().result()
    finally:
        pool.shutdown(wait=False)


def chunks(iterable, size):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# --------------------------------------------------------------------------- writer
class Writer:
    def __init__(self, csv_path, out_dir, threads, max_pending, packbits):
        self.out_dir = out_dir
        self.packbits = packbits
        self.ext = "npz" if packbits else "npy"
        os.makedirs(out_dir, exist_ok=True)
        is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        self.file = open(csv_path, "a", newline="", encoding="utf-8")
        self.csv = csv.writer(self.file)
        if is_new:
            self.csv.writerow(["file_name", "key_elements"])
        self.pool = ThreadPoolExecutor(threads)
        self.sem = threading.Semaphore(max_pending)
        self.n = 0

    def _save(self, path, arr):
        suffix = f".tmp.{threading.get_ident()}.npz" if self.packbits else f".tmp.{threading.get_ident()}.npy"
        tmp = path + suffix
        try:
            if self.packbits:
                np.savez(tmp, masks=np.packbits(arr, axis=-1), shape=np.array(arr.shape))
            else:
                np.save(tmp, arr)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _done(self, fut):
        try:
            exc = fut.exception()
            if exc is not None:
                print(f"[save] {exc}", flush=True)
        finally:
            self.sem.release()

    def save(self, file_name, keywords, arr):
        path = os.path.join(self.out_dir, base_name(file_name) + "." + self.ext)
        self.sem.acquire()
        fut = self.pool.submit(self._save, path, arr)
        fut.add_done_callback(self._done)
        self.csv.writerow([file_name, repr(list(keywords))])
        self.n += 1
        if self.n % 500 == 0:
            self.file.flush()

    def save_empty(self, file_name):
        self.csv.writerow([file_name, "[]"])
        self.n += 1
        if self.n % 500 == 0:
            self.file.flush()

    def close(self):
        self.pool.shutdown(wait=True)
        self.file.flush()
        self.file.close()


# ----------------------------------------------------------------------- GPU worker
def get_text(model, keyword, cache, limit, device):
    if cache is not None:
        hit = cache.get(keyword)
        if hit is not None:
            cache.move_to_end(keyword)
            return hit
    tx = model.backbone.forward_text([keyword], device=device)
    features, mask = tx["language_features"], tx["language_mask"]
    if cache is not None:
        cache[keyword] = (features, mask)
        if len(cache) > limit:
            cache.popitem(last=False)
    return features, mask


def run_chunk(model, processor, images, keywords_list, args, text_cache, device, torch):
    """Run one image chunk; returns a list of (valid_keywords, bool ndarray or None)."""
    from sam3.model.data_misc import FindStage, interpolate

    n_images = len(images)
    state = processor.set_image_batch(images)
    backbone_out = state["backbone_out"]

    target_sizes = []
    for image in images:
        w, h = image.size
        if args.mask_size and max(h, w) > args.mask_size:
            scale = args.mask_size / max(h, w)
            target_sizes.append((max(1, int(round(h * scale))), max(1, int(round(w * scale)))))
        else:
            target_sizes.append((h, w))

    # keyword -> image indices that need it
    inverted = {}
    for i, kws in enumerate(keywords_list):
        for kw in kws:
            inverted.setdefault(kw, []).append(i)

    per_image = [dict() for _ in range(n_images)]
    for keyword, indices in inverted.items():
        count = len(indices)
        features, mask = get_text(model, keyword, text_cache, args.text_cache_size, device)
        backbone_out["language_features"] = features.expand(-1, count, -1)
        backbone_out["language_mask"] = mask.expand(count, -1)

        find_stage = FindStage(
            img_ids=torch.tensor(indices, device=device, dtype=torch.long),
            text_ids=torch.arange(count, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = model._get_dummy_prompt(num_prompts=count)
        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        probs = out["pred_logits"].sigmoid()
        presence = out["presence_logit_dec"].sigmoid().unsqueeze(1)
        probs = (probs * presence).squeeze(-1)  # (count, queries)
        keep = probs > args.confidence_threshold
        counts = keep.sum(1).tolist()           # one sync for the whole keyword group
        pred_masks = out["pred_masks"]
        selected = pred_masks[keep]             # (sum(counts), Hm, Wm), one sync

        offset = 0
        for j, image_index in enumerate(indices):
            kept = counts[j]
            if kept == 0:
                continue
            masks = selected[offset:offset + kept]
            offset += kept
            th, tw = target_sizes[image_index]
            masks = interpolate(
                masks.unsqueeze(1), (th, tw), mode="bilinear", align_corners=False
            ).sigmoid().squeeze(1)
            # Multiple detections for the same keyword are OR-ed, matching the old script.
            per_image[image_index][keyword] = (masks > 0.5).any(0)

    results = []
    for i, kws in enumerate(keywords_list):
        valid, masks = [], []
        for kw in kws:
            m = per_image[i].get(kw)
            if m is not None:
                valid.append(kw)
                masks.append(m)
        if not masks:
            results.append(([], None))
            continue
        stacked = torch.stack(masks, 0)
        keep = stacked.flatten(1).any(1).cpu().numpy()
        if not keep.any():
            results.append(([], None))
            continue
        arr = stacked[torch.from_numpy(keep).to(device)].cpu().numpy()
        results.append(([k for k, f in zip(valid, keep) if f], arr))
    return results


def process_shard(rank, gpu, shard, args, model, processor, text_cache, pbar, budget, torch):
    suffix, image_path, caption_path, part, num_parts = shard
    out_dir = os.path.join(args.out_root, f"mask-{suffix}")
    csv_path = os.path.join(args.out_root, f"mask-{suffix}.csv")
    os.makedirs(out_dir, exist_ok=True)

    ext = "npz" if args.packbits else "npy"
    done = set() if args.no_resume else build_done(csv_path, out_dir, ext)
    writer_path = csv_path if num_parts == 1 else f"{csv_path}.part{part}"
    writer = Writer(writer_path, out_dir, args.save_threads, args.max_pending_saves, args.packbits)

    pbar.set_description(f"w{rank}/gpu{gpu} mask-{suffix}")
    stream = stream_samples(image_path, caption_path, args, done, part, num_parts)
    stream = prefetch(stream, decode, args.decode_threads, args.prefetch)

    stop = False
    try:
        for chunk in chunks(stream, args.batch_size):
            active = []
            for file_name, image, kws in chunk:
                if image is None or file_name in done:
                    continue
                active.append((file_name, image, kws))
            if not active:
                continue
            if budget[0]:
                active = active[:budget[0]]

            images = [x[1] for x in active]
            keywords_list = [x[2] for x in active]
            results = None
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    results = run_chunk(model, processor, images, keywords_list,
                                        args, text_cache, f"cuda:{gpu}", torch)
            except Exception as exc:
                print(f"[w{rank}] chunk failed ({exc}); falling back to single images", flush=True)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                results = []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for image, kws in zip(images, keywords_list):
                        try:
                            results.append(run_chunk(model, processor, [image], [kws],
                                                     args, text_cache, f"cuda:{gpu}", torch)[0])
                        except Exception as exc2:
                            print(f"[w{rank}] image failed ({exc2})", flush=True)
                            results.append(None)

            for (file_name, _image, _kws), result in zip(active, results):
                if result is None:
                    continue  # failed image: leave it for the next run
                valid, arr = result
                if arr is not None and len(valid):
                    writer.save(file_name, valid, arr)
                else:
                    writer.save_empty(file_name)
                done.add(file_name)
                pbar.update(1)

            if budget[0]:
                budget[0] -= len(active)
                if budget[0] <= 0:
                    stop = True
                    break
    finally:
        writer.close()
    return stop


def worker(rank, gpu_ids, args, assignments):
    import torch

    gpu = gpu_ids[rank % len(gpu_ids)]
    torch.cuda.set_device(gpu)
    torch.set_num_threads(args.torch_threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    sys.path.insert(0, SAM3_PATH)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from tqdm import tqdm

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        compile=args.compile,
    ).to(f"cuda:{gpu}").eval()
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=f"cuda:{gpu}",
        confidence_threshold=args.confidence_threshold,
    )
    text_cache = OrderedDict() if args.text_cache_size > 0 else None

    pbar = tqdm(desc=f"w{rank}/gpu{gpu}", position=rank, mininterval=5)
    budget = [args.limit]
    try:
        for shard in assignments[rank]:
            if process_shard(rank, gpu, shard, args, model, processor, text_cache, pbar, budget, torch):
                break
    finally:
        pbar.close()


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"not a directory: {data_dir}")
    args.out_root = os.path.abspath(args.out_dir or os.path.join(data_dir, "masks"))
    only = None
    if args.shards:
        only = {s.strip() for s in args.shards.split(",") if s.strip()}

    shards = discover_shards(data_dir, only)
    if not shards:
        raise SystemExit(f"no image-*/caption-* parquet pairs found in {data_dir}")
    os.makedirs(args.out_root, exist_ok=True)

    for suffix, _, _ in shards:
        recover_parts(args.out_root, suffix)

    # Resolve the checkpoint once before spawning, so workers do not race on HF.
    args.checkpoint = None
    try:
        sys.path.insert(0, SAM3_PATH)
        from sam3.model_builder import download_ckpt_from_hf
        args.checkpoint = download_ckpt_from_hf(version="sam3")
    except Exception as exc:
        print(f"[warn] checkpoint pre-download failed ({exc}); letting workers load it", flush=True)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise SystemExit("no GPUs selected")

    bad = [g for g in gpu_ids if g < 0 or g >= torch.cuda.device_count()]
    if bad:
        raise SystemExit(f"invalid GPU ids {bad}; visible devices: 0..{torch.cuda.device_count() - 1}")
    world = max(1, args.procs or len(gpu_ids))
    assignments = build_assignments(shards, world)
    per_gpu = {gpu: sum(1 for r in range(world) if gpu_ids[r % len(gpu_ids)] == gpu) for gpu in gpu_ids}
    print(f"{len(shards)} shard(s), {world} worker(s) over GPUs {gpu_ids}", flush=True)
    print(f"workers/GPU: {per_gpu}; effective image batch/GPU: "
          f"{ {gpu: per_gpu[gpu] * args.batch_size for gpu in gpu_ids} }", flush=True)
    print(f"output: {args.out_root}", flush=True)
    if args.packbits:
        print("[warn] --packbits writes .npz, but src/shards.py only reads .npy", flush=True)

    import torch.multiprocessing as mp
    mp.spawn(worker, args=(gpu_ids, args, assignments), nprocs=world, join=True)

    for suffix, _, _ in shards:
        finalize_shard(args.out_root, suffix)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
