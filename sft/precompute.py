"""Cache the frozen DINOv2 CLS and Qwen3 EOS features for SFT.

The SFT config freezes both towers, so their outputs never change. Running the backbones every
step is by far the dominant cost. This script runs them once and writes

    image.npy     (N, D_dino) float16   DINOv2 x_norm_clstoken
    text.npy      (N, D_text) float16   Qwen3 last-token (EOS) hidden state
    valid.npy     (N,) bool             False for images that could not be decoded
    train_idx.npy / val_idx.npy         same split as sft/data.py (split_seed, val_fraction)
    meta.json                           dims / models / column names / split settings

``sft/train.py`` then only runs the two linear projections (and the loss), which is orders of
magnitude faster and uses no backbone memory.

Usage:
    cd /home/susanket/esri-vlm
    python sft/precompute.py \
        --data_dir /data/susanket/vlm/sft_reshard \
        --output_dir /data/susanket/vlm/sft_features \
        --batch_size 256

Point ``data.data_dir`` at a directory containing training parquet(s) only; do not mix the test
split in here. Precomputed features use the same deterministic centre crop for every image, so
random crop / flip augmentation is no longer applied during training.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "sft")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import resolve_columns  # noqa: E402

SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DINOV2_REPO = "facebookresearch/dinov2"


# --------------------------------------------------------------------------- image decoding
def _cpu_decode(raw: bytes) -> Optional[torch.Tensor]:
    try:
        img = Image.open(io.BytesIO(raw))
        img.draft("RGB", (SIZE * 2, SIZE * 2))
        img = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
    except Exception:
        return None


def _gpu_resize(im: torch.Tensor) -> torch.Tensor:
    return F.interpolate(im.unsqueeze(0).float(), size=(SIZE, SIZE),
                         mode="bilinear", antialias=True, align_corners=False)[0]


def decode_batch(raws: List[bytes], device: torch.device,
                 cpu_pool: ThreadPoolExecutor) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = [None] * len(raws)
    is_cuda = device.type == "cuda"
    jpg = [i for i, b in enumerate(raws) if is_cuda and isinstance(b, (bytes, bytearray)) and b[:2] == b"\xff\xd8"]

    if jpg:
        datas = [torch.frombuffer(raws[i], dtype=torch.uint8) for i in jpg]
        try:
            decoded = decode_jpeg(datas, device=device, mode=ImageReadMode.RGB)
            for i, im in zip(jpg, decoded):
                out[i] = _gpu_resize(im)
        except Exception:
            for i, d in zip(jpg, datas):
                try:
                    out[i] = _gpu_resize(decode_jpeg(d, device=device, mode=ImageReadMode.RGB))
                except Exception:
                    out[i] = None

    rest = [i for i in range(len(raws)) if out[i] is None]
    for i, t in zip(rest, cpu_pool.map(_cpu_decode, [raws[i] for i in rest])):
        if t is not None:
            out[i] = t.to(device, non_blocking=True).float()
    return out


def load_image_bytes_col(batch: pa.RecordBatch):
    col = batch.column(0)
    if pa.types.is_struct(col.type):
        col = pc.struct_field(col, "bytes")
    return col.to_pylist()


# --------------------------------------------------------------------------- text / image towers
def embed_text(texts: List[str], tokenizer, model, device, context_length: int) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=context_length,
                    return_tensors="pt")
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()
           if k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        hidden = model(**enc).last_hidden_state
        last = enc["attention_mask"].sum(dim=1) - 1
        e = hidden[torch.arange(hidden.shape[0], device=device), last]
    return e.float().cpu().numpy().astype(np.float16, copy=False)


def embed_images(raws: List[bytes], dino, device, dtype, mean, std,
                 cpu_pool: ThreadPoolExecutor):
    decoded = decode_batch(raws, device, cpu_pool)
    valid = [d is not None for d in decoded]
    feats = np.zeros((len(raws), 0), dtype=np.float16)
    if not any(valid):
        return feats, valid
    keep = [i for i, d in enumerate(decoded) if d is not None]
    x = torch.stack([decoded[i] for i in keep]).div_(255.0)
    x = ((x - mean) / std).to(dtype)
    with torch.inference_mode():
        out = dino.forward_features(x)
        cls = out["x_norm_clstoken"].float().cpu().numpy()
    feats = np.zeros((len(raws), cls.shape[1]), dtype=np.float16)
    feats[keep] = cls.astype(np.float16, copy=False)
    return feats, valid


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory with training *.parquet")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_col", default="image_bytes")
    ap.add_argument("--caption_col", default=None)
    ap.add_argument("--image_path_col", default=None)
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--image_model_name", default="dinov2_vits14")
    ap.add_argument("--text_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256, help="images/texts per GPU step")
    ap.add_argument("--read_batch", type=int, default=2048, help="rows read from parquet at a time")
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--limit_rows", type=int, default=0, help="debug: only first N records")
    ap.add_argument("--cpu_threads", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no *.parquet files in {args.data_dir}")
    image_col, caption_col, image_path_col = resolve_columns(
        files, args.image_col, args.caption_col, args.image_path_col)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] {len(files)} parquet file(s), image={image_col or image_path_col} "
          f"caption={caption_col}")

    # ---- pass 1: captions (cheap) -> exact same records / split as sft/data.py
    t0 = time.time()
    captions: List[str] = []
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=65536, columns=[caption_col]):
            captions.extend(b.column(0).to_pylist())
    keep = np.array([c is not None and str(c).strip() != "" for c in captions], dtype=bool)
    kept_rows = np.flatnonzero(keep)
    n_total = len(captions)
    n = len(kept_rows)
    if args.limit_rows:
        kept_rows = kept_rows[:args.limit_rows]
        n = len(kept_rows)
    if n == 0:
        raise RuntimeError("no non-empty captions found")
    print(f"[precompute] {n} usable records out of {n_total} rows ({time.time() - t0:.1f}s)")

    perm = np.random.RandomState(args.split_seed).permutation(n)
    n_val = int(n * args.val_fraction)
    val_local, train_local = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    # kept_rows[local] -> global row; features are written in kept_rows order
    train_idx = train_local.copy()
    val_idx = val_local.copy()

    # ---- load towers once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if device.type == "cpu":
        dtype = torch.float32
    print(f"[precompute] loading {args.image_model_name} / {args.text_model_name} on {device} ({dtype})")

    try:
        dino = torch.hub.load(DINOV2_REPO, args.image_model_name, trust_repo=True)
    except TypeError:
        dino = torch.hub.load(DINOV2_REPO, args.image_model_name)
    dino = dino.eval().to(device, dtype)

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_model = AutoModel.from_pretrained(
        args.text_model_name, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    dino_dim = int(getattr(dino, "embed_dim", 384))
    text_dim = int(text_model.config.hidden_size)

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    img = np.lib.format.open_memmap(out_dir / "image.npy", mode="w+",
                                    dtype=np.float16, shape=(n, dino_dim))
    txt = np.lib.format.open_memmap(out_dir / "text.npy", mode="w+",
                                    dtype=np.float16, shape=(n, text_dim))
    valid = np.lib.format.open_memmap(out_dir / "valid.npy", mode="w+",
                                      dtype=np.bool_, shape=(n,))
    valid[:] = False

    # ---- pass 2: stream images in row order; write aligned features
    out_pos = 0
    global_row = 0
    t0 = time.time()
    kept_set = np.zeros(n_total, dtype=bool)
    kept_set[kept_rows] = True

    with ThreadPoolExecutor(args.cpu_threads) as cpu_pool:
        for fi, f in enumerate(files):
            pf = pq.ParquetFile(f)
            cols = [image_path_col] if image_path_col else [image_col]
            for b in pf.iter_batches(batch_size=args.read_batch, columns=cols):
                start = global_row
                global_row += b.num_rows
                sel_local = np.flatnonzero(kept_set[start:global_row])
                if sel_local.size == 0:
                    continue
                sel_global = start + sel_local

                if image_path_col:
                    root = Path(args.image_root) if args.image_root else f.parent
                    raws = []
                    for g in sel_global:
                        rel = b.column(0)[g - start].as_py()
                        try:
                            raws.append((root / rel).read_bytes())
                        except Exception:
                            raws.append(None)
                else:
                    col = load_image_bytes_col(b)
                    raws = [col[g - start] for g in sel_global]
                caps = [captions[g] for g in sel_global]

                for c0 in range(0, len(sel_global), args.batch_size):
                    c1 = min(c0 + args.batch_size, len(sel_global))
                    raws_c, caps_c = raws[c0:c1], caps[c0:c1]
                    feats, ok = embed_images(raws_c, dino, device, dtype, mean, std, cpu_pool)
                    txt[out_pos + c0:out_pos + c1] = embed_text(
                        caps_c, tokenizer, text_model, device, args.context_length)
                    if feats.shape[1]:
                        img[out_pos + c0:out_pos + c1] = feats
                    valid[out_pos + c0:out_pos + c1] = ok

                out_pos += len(sel_global)
                done = out_pos
                if done and (done % 10000 < len(sel_global) or done == n):
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"[precompute] {done}/{n} ({100.0 * done / n:.1f}%) {rate:.1f} rec/s")

    img.flush()
    txt.flush()
    valid.flush()

    valid_arr = np.asarray(valid)
    n_bad = int((~valid_arr).sum())
    train_idx = train_idx[valid_arr[train_idx]]
    val_idx = val_idx[valid_arr[val_idx]]
    if n_bad:
        print(f"[precompute] dropped {n_bad} undecodable image(s) from train/val")

    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    meta = {
        "num_records": int(n),
        "num_valid": int(n - n_bad),
        "image_dim": dino_dim,
        "text_dim": text_dim,
        "image_model_name": args.image_model_name,
        "text_model_name": args.text_model_name,
        "context_length": int(args.context_length),
        "val_fraction": float(args.val_fraction),
        "split_seed": int(args.split_seed),
        "image_col": image_col,
        "caption_col": caption_col,
        "image_path_col": image_path_col,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[precompute] done in {time.time() - t0:.1f}s -> {out_dir}")
    print(f"[precompute] train={len(train_idx)} val={len(val_idx)} "
          f"image=({n},{dino_dim}) text=({n},{text_dim})")


if __name__ == "__main__":
    main()
