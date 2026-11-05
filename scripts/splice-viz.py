"""
Visualize SpLiCE decompositions like Fig. 2 of the paper: image on the left, non-zero concepts
with their weights as horizontal bars on the right.

    python /home/susanket/esri-vlm/scripts/splice-viz.py --parquet /data/susanket/vlm/images3/image-00.parquet --out_dir /data/susanket/vlm/splice/ --save /data/susanket/vlm/splice/grid3.png                  # first 6 images
    python viz_splice.py --parquet image.parquet --idx 3 17 42 --topk 12               # specific rows
    python viz_splice.py --parquet image.parquet --random 8 --save grid.png            # random sample
    python viz_splice.py --parquet image.parquet --idx 5 --show                        # interactive window
    python viz_splice.py --parquet image.parquet --objects buildings forests           # specific objects
"""
import argparse
import io
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image


def to_bytes(v):
    if isinstance(v, dict):
        v = v.get("bytes")
    return bytes(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="image.parquet")
    ap.add_argument("--image_col", default="image_bytes")
    ap.add_argument("--out_dir", default="splice_out", help="folder produced by splice_georsclip.py")
    ap.add_argument("--idx", type=int, nargs="*", help="row indices to show")
    ap.add_argument("--random", type=int, default=100, help="show N random rows instead")
    ap.add_argument("--n", type=int, default=0, help="number of rows if neither --idx nor --random")
    ap.add_argument("--topk", type=int, default=20, help="max concepts per image (all non-zeros are in the file)")
    ap.add_argument("--objects", type=str, nargs="*", help="specific object names to show scores for (overrides topk)")
    ap.add_argument("--cols", type=int, default=4, help="image+bars panels per grid row")
    ap.add_argument("--save", default=None, help="output png (default: <out_dir>/viz.png)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dec = pd.read_parquet(Path(args.out_dir) / "decompositions.parquet")
    imgs = pd.read_parquet(args.parquet, columns=[args.image_col])

    # If the decompositions file contains a global vocabulary column (e.g., 'vocab' or 'all_concepts'),
    # you can check presence against it. If your file only stores non-zero concepts in row["concepts"],
    # we can track a global vocabulary if available, or assume row["concepts"] represents the available vocabulary.
    # Here we check against row["concepts"] or a hypothetical global vocabulary column if your pipeline stores it.
    
    if args.idx:
        rows = args.idx
    elif args.random:
        rows = np.random.default_rng(0).choice(len(dec), min(args.random, len(dec)), replace=False).tolist()
    else:
        rows = list(range(min(args.n, len(dec))))

    ncols = min(args.cols, len(rows))
    nrows = int(np.ceil(len(rows) / ncols))
    fig = plt.figure(figsize=(7.5 * ncols, 3.4 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.15)

    for k, r in enumerate(rows):
        # Change wspace here (e.g., from 0.08 to 0.18)
        inner = outer[k // ncols, k % ncols].subgridspec(1, 2, width_ratios=[1, 1.1], wspace=0.18)
        ax_i, ax_b = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

        img = Image.open(io.BytesIO(to_bytes(imgs.iloc[r][args.image_col]))).convert("RGB")
        ax_i.imshow(img)
        ax_i.axis("off")
        ax_i.set_title(f"row {r}", fontsize=9)

        row = dec.iloc[r]
        concept_to_weight = dict(zip(row["concepts"], row["weights"]))
        
        # If your decompositions parquet has a separate full vocabulary list per image/row, 
        # replace row["concepts"] in `vocab_set` with that column name (e.g. row["vocab"]).
        # Assuming row["concepts"] contains the evaluated vocabulary space for this row:
        vocab_set = set(row["vocab"]) if "vocab" in row else set(row["concepts"])

        if args.objects:
            # Keep only objects present in the row's vocabulary
            valid_objs = [obj for obj in args.objects if obj in vocab_set]
            # Sort by weight descending (zero weights will naturally fall at the bottom or order stably)
            valid_objs = sorted(valid_objs, key=lambda obj: concept_to_weight.get(obj, 0.0), reverse=True)
            names = valid_objs
            wts = np.asarray([concept_to_weight.get(obj, 0.0) for obj in names])
        else:
            names = list(row["concepts"])[:args.topk]
            wts = np.asarray(row["weights"])[:args.topk]

        y = np.arange(len(names))
        ax_b.barh(y, wts, color="#4e64b6")
        ax_b.set_yticks(y)
        ax_b.set_yticklabels(names, fontsize=8)
        ax_b.invert_yaxis()
        for yi, w in zip(y, wts):
            ax_b.text(w, yi, f" {w:.3f}", va="center", fontsize=7)
        
        max_wt = wts.max() if len(wts) else 1
        ax_b.set_xlim(0, (max_wt if max_wt > 0 else 1) * 1.25)
        ax_b.tick_params(axis="x", labelsize=7)
        for s in ("top", "right"):
            ax_b.spines[s].set_visible(False)
        ax_b.set_title(f"L0={int(row['l0'])} (showing {len(names)})  recon cos={row['recon_cosine']:.2f}", fontsize=8)
        if len(names) == 0:
            msg = "no matching objects in vocab" if args.objects else "no concepts (try a lower l1_penalty)"
            ax_b.text(0.5, 0.5, msg, ha="center", transform=ax_b.transAxes, fontsize=8)

    save = args.save or str(Path(args.out_dir) / "viz.png")
    fig.savefig(save, dpi=150, bbox_inches="tight")
    print(f"Saved {save}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()