"""One-off: collapse per-image DINOv2 .npy files ((1+P, D) float32, ~394 KB each) into one
CLS matrix per shard, (N, D) float32 (~1.5 KB per image).

Training with use_masks=False only ever reads token 0, but the current loader reads, validates
(np.isfinite over 98k floats), pins and ships the full (257, 384) tensor for every sample.

Usage:  python precompute_cls.py /data/susanket/vlm/images3 --workers 32
Writes: <root>/cls-<shard>.npy  (N, D) float32
        <root>/cls-<shard>.names.txt   one file stem per row, same order
"""
import argparse
import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def load_cls(p: Path):
    try:
        a = np.load(p, mmap_mode="r")          # mmap: only the first page of the file is read
        v = np.array(a[0], dtype=np.float32)
    except (EOFError, ValueError, OSError):
        return None
    return v if np.isfinite(v).all() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)

    for emb_dir in tqdm.tqdm(sorted(d for d in root.glob("image-*") if d.is_dir())):
        name = emb_dir.name[len("image-"):]
        files = sorted(p for p in emb_dir.iterdir() if p.suffix == ".npy")
        with ThreadPoolExecutor(args.workers) as ex:
            vecs = list(ex.map(load_cls, files))
        keep = [(f.stem, v) for f, v in zip(files, vecs) if v is not None]
        bad = len(files) - len(keep)
        if not keep:
            print(f"[{name}] nothing usable"); continue
        np.save(root / f"cls-{name}.npy", np.stack([v for _, v in keep]))
        (root / f"cls-{name}.names.txt").write_text("\n".join(s for s, _ in keep))
        print(f"[{name}] {len(keep)} rows, {bad} skipped (corrupt/non-finite)")


if __name__ == "__main__":
    main()