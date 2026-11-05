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
    ap.add_argument("--caption_col", default="caption", help="column name for image caption if available")
    ap.add_argument("--out_dir", default="splice_out", help="folder produced by splice_georsclip.py")
    ap.add_argument("--idx", type=int, nargs="*", help="row indices to show")
    ap.add_argument("--random", type=int, default=6, help="show N random rows instead")
    ap.add_argument("--n", type=int, default=0, help="number of rows if neither --idx nor --random")
    ap.add_argument("--topk", type=int, default=6, help="max concepts per image to display")
    ap.add_argument("--objects", type=str, nargs="*", help="specific object names to show scores for (overrides topk)")
    ap.add_argument("--cols", type=int, default=3, help="image+text panels per grid row (paper style uses 3)")
    ap.add_argument("--save", default=None, help="output png (default: <out_dir>/viz_paper.png)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dec = pd.read_parquet(Path(args.out_dir) / "decompositions.parquet")
    
    # Check for caption column safely without using unsupported 'nrows' argument in read_parquet
    imgs_meta = pd.read_parquet(args.parquet)
    has_caption = args.caption_col and args.caption_col in imgs_meta.columns

    img_cols = [args.image_col]
    if has_caption:
        img_cols.append(args.caption_col)

    imgs = pd.read_parquet(args.parquet, columns=img_cols)

    if args.idx:
        rows = args.idx
    elif args.random:
        rows = np.random.default_rng(0).choice(len(dec), min(args.random, len(dec)), replace=False).tolist()
    else:
        rows = list(range(min(args.n if args.n > 0 else 6, len(dec))))

    ncols = min(args.cols, len(rows))
    nrows = int(np.ceil(len(rows) / ncols))
    
    # Figure size tuned for paper layout (image + text list side-by-side per panel)
    fig = plt.figure(figsize=(5.2 * ncols, 3.2 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.2)

    for k, r in enumerate(rows):
        inner = outer[k // ncols, k % ncols].subgridspec(1, 2, width_ratios=[1, 1.1], wspace=0.08)
        ax_i, ax_t = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

        # Display Image
        img = Image.open(io.BytesIO(to_bytes(imgs.iloc[r][args.image_col]))).convert("RGB")
        ax_i.imshow(img)
        ax_i.axis("off")

        # Display Caption underneath image if available
        if has_caption:
            caption_text = str(imgs.iloc[r][args.caption_col])
            ax_i.text(
                0.0, -0.08, f'"{caption_text}"', 
                transform=ax_i.transAxes, 
                fontsize=8, 
                ha="left", 
                va="top", 
                wrap=True
            )

        # Retrieve concept decomposition for this row
        row = dec.iloc[r]
        concept_to_weight = dict(zip(row["concepts"], row["weights"]))
        vocab_set = set(row["vocab"]) if "vocab" in row else set(row["concepts"])

        if args.objects:
            valid_objs = [obj for obj in args.objects if obj in vocab_set]
            valid_objs = sorted(valid_objs, key=lambda obj: concept_to_weight.get(obj, 0.0), reverse=True)
            names = valid_objs
            wts = np.asarray([concept_to_weight.get(obj, 0.0) for obj in names])
        else:
            names = list(row["concepts"])[:args.topk]
            wts = np.asarray(row["weights"])[:args.topk]

        # Configure text-only axis to mimic Fig. 2 paper table style (Concept | Weight)
        ax_t.axis("off")
        
        if len(names) > 0:
            # Draw vertical divider line matching the paper style
            ax_t.axvline(x=0.62, ymin=0.05, ymax=0.95, color="black", linewidth=0.8)

            y_positions = np.linspace(0.88, 0.12, len(names))
            for yi, name, w in zip(y_positions, names, wts):
                # Concept name (left-aligned)
                ax_t.text(0.05, yi, str(name), fontsize=9, ha="left", va="center", transform=ax_t.transAxes)
                # Weight value (right-aligned before the divider)
                ax_t.text(0.58, yi, f"{w:.2f}", fontsize=9, ha="right", va="center", transform=ax_t.transAxes)
        else:
            msg = "no matching objects" if args.objects else "no concepts"
            ax_t.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_t.transAxes, fontsize=8)

    save = args.save or str(Path(args.out_dir) / "viz_paper.png")
    fig.savefig(save, dpi=300, bbox_inches="tight")
    print(f"Saved {save}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()