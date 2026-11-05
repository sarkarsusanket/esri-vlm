"""
Merge an image parquet with a caption parquet, embed a subset with the custom CLIP and GeoRSCLIP,
and plot the positive / negative image-text cosine-similarity distributions.

    python scripts/sim_dist.py \
        --image_parquet /data/susanket/vlm/images3/image-00.parquet \
        --caption_parquet /data/susanket/vlm/images3/caption-00.parquet \
        --n 10000 --select first --out sim_dist.png

Custom-model embeddings are read from the precomputed DINOv2 / Qwen directories
(image-00/*.npy and caption-00/summary/*.npy) when the checkpoint expects them; otherwise the
DINOv2 and Qwen backbones are loaded and run on the raw image bytes and summary text.
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_IMAGE_PARQUET = "/data/susanket/vlm/images3/image-00.parquet"
DEFAULT_CAPTION_PARQUET = "/data/susanket/vlm/images3/caption-00.parquet"
DEFAULT_CUSTOM_CKPT = ("/home/susanket/esri-vlm/lax2unwh/checkpoints/"
                       "epoch=244-step=1447215.ckpt")
DEFAULT_GEORSCLIP_CKPT = "/data/susanket/vlm/sota/ckpt/RS5M_ViT-B-32.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_parquet", default=DEFAULT_IMAGE_PARQUET)
    ap.add_argument("--caption_parquet", default=DEFAULT_CAPTION_PARQUET)
    ap.add_argument("--image_id_col", default=None, help="default: file_name, then id")
    ap.add_argument("--caption_id_col", default=None, help="default: file_name, then id")
    ap.add_argument("--caption_col", default=None, help="default: summary, then scene_caption/caption")
    ap.add_argument("--status_col", default=None, help="default: status if present")
    ap.add_argument("--image_col", default="image_bytes", help="column holding raw image bytes")
    ap.add_argument("--image_emb_dir", default=None, help="default: <image_parquet parent>/<stem>")
    ap.add_argument("--caption_emb_dir", default=None,
                    help="default: <caption_parquet parent>/<stem>")
    ap.add_argument("--caption_emb_subdir", default="summary")
    ap.add_argument("--n", type=int, default=10000, help="number of merged rows to evaluate")
    ap.add_argument("--select", choices=["first", "random"], default="first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", choices=["custom", "georsclip"],
                    default=["custom", "georsclip"])
    ap.add_argument("--custom_ckpt", default=DEFAULT_CUSTOM_CKPT)
    ap.add_argument("--custom_precomputed", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--georsclip_ckpt", default=DEFAULT_GEORSCLIP_CKPT)
    ap.add_argument("--georsclip_arch", default="ViT-B-32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024, help="rows per similarity chunk")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--sim_range", type=float, nargs=2, default=(-1.0, 1.0))
    ap.add_argument("--out", default="sim_dist.png")
    ap.add_argument("--save_merged", default=None, help="optional parquet with id + summary")
    return ap.parse_args()


def _columns(path):
    return pq.ParquetFile(path).schema_arrow.names


def _pick(cols, requested, candidates, name):
    if requested is not None:
        if requested not in cols:
            raise KeyError(f"{name} column {requested!r} not in parquet; available: {cols}")
        return requested
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"could not find a {name} column; available: {cols}")


def read_merged(image_parquet, caption_parquet, args):
    img_cols = _columns(image_parquet)
    cap_cols = _columns(caption_parquet)

    image_id = _pick(img_cols, args.image_id_col,
                     ["file_name", "id", "image_id", "key"], "image id")
    caption_id = _pick(cap_cols, args.caption_id_col,
                       ["file_name", "id", "image_id", "key"], "caption id")
    caption_col = _pick(cap_cols, args.caption_col,
                        ["summary", "scene_caption", "caption", "text"], "caption")
    status_col = args.status_col
    if status_col is None and "status" in cap_cols:
        status_col = "status"

    cap_read = [caption_id, caption_col] + ([status_col] if status_col else [])
    cap = pd.read_parquet(caption_parquet, columns=cap_read)
    if status_col is not None:
        status = cap[status_col].astype(str).str.lower()
        ok = status.isin(["ok", "success", "done"])
        if ok.any():
            cap = cap[ok]
    cap[caption_col] = cap[caption_col].astype(str).str.strip()
    cap = cap[cap[caption_col].notna() & (cap[caption_col] != "")]
    cap = cap.drop_duplicates(caption_id)
    cap_map = cap.set_index(caption_id)[caption_col]

    img_ids = pq.read_table(image_parquet, columns=[image_id]).column(0).to_pandas()
    valid = img_ids[img_ids.isin(cap_map.index)].reset_index(drop=True)

    if valid.empty:
        raise RuntimeError(
            f"no matching ids between {image_parquet} ({image_id}) and "
            f"{caption_parquet} ({caption_id}). If you meant the generated captions for "
            f"image-00.parquet, use {Path(image_parquet).parent / 'caption-00.parquet'}."
        )

    if args.n and args.n < len(valid):
        if args.select == "random":
            rng = np.random.default_rng(args.seed)
            idx = np.sort(rng.choice(len(valid), size=args.n, replace=False))
            chosen = valid.iloc[idx].reset_index(drop=True)
        else:
            chosen = valid.iloc[:args.n].reset_index(drop=True)
    else:
        chosen = valid

    merged = pd.DataFrame({"_id": chosen.astype(str).values})
    merged["summary"] = merged["_id"].map(cap_map).values
    return merged


def _to_bytes(v):
    if isinstance(v, dict):
        v = v.get("bytes")
    return bytes(v)


def load_raw_images(image_parquet, wanted, image_id, image_col, batch_size=2048):
    wanted = set(wanted)
    found = {}
    pf = pq.ParquetFile(image_parquet)
    for batch in pf.iter_batches(columns=[image_id, image_col], batch_size=batch_size):
        data = batch.to_pydict()
        for i, blob in zip(data[image_id], data[image_col]):
            i = str(i)
            if i in wanted and i not in found:
                try:
                    found[i] = Image.open(io.BytesIO(_to_bytes(blob))).convert("RGB")
                except Exception:
                    found[i] = None
        if len(found) >= len(wanted):
            break
    return found


def resolve_device(arg):
    if arg:
        if arg.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split(":", 1)[1]
            return torch.device("cuda:0")
        if arg == "cuda":
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return torch.device(arg)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def import_src_main():
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    import main as src_main
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return src_main


def custom_precomputed_mode(ckpt, args, image_emb_dir, caption_emb_dir):
    if args.custom_precomputed == "false":
        return False
    hp = dict(ckpt["hyper_parameters"])
    needed = [
        hp.get("precomputed_image", False) and image_emb_dir.is_dir(),
        hp.get("precomputed_text", False) and (caption_emb_dir / args.caption_emb_subdir).is_dir(),
    ]
    if args.custom_precomputed == "true":
        if not all(needed):
            raise FileNotFoundError(
                f"precomputed embeddings requested but missing: image={image_emb_dir}, "
                f"caption={caption_emb_dir / args.caption_emb_subdir}"
            )
        return True
    return all(needed)


def load_custom_model(ckpt, precomputed, device):
    src_main = import_src_main()
    hp = dict(ckpt["hyper_parameters"])
    hp.pop("_instantiator", None)
    hp["precomputed_image"] = bool(precomputed)
    hp["precomputed_text"] = bool(precomputed)
    hp["load_text_backbone"] = None if precomputed else True
    lm = src_main.CLIPLightningModule(**hp)
    lm.load_state_dict(ckpt["state_dict"])
    return lm.to(device).eval()


def encode_custom_precomputed(lm, ids, image_emb_dir, caption_emb_dir, subdir,
                              batch_size, device):
    img_out, txt_out = [], []
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        imgs = np.stack([np.load(image_emb_dir / f"{i}.npy").astype(np.float32) for i in batch])
        txts = []
        for i in batch:
            a = np.load(caption_emb_dir / subdir / f"{i}.npy")
            txts.append((a[0] if a.ndim == 2 else a).astype(np.float32))
        txts = np.stack(txts)
        with torch.inference_mode():
            ie = lm.model.encode_image(torch.from_numpy(imgs).to(device))
            te = lm.model.encode_text_emb(torch.from_numpy(txts).to(device))
            img_out.append(F.normalize(ie.float(), dim=-1).cpu().numpy())
            txt_out.append(F.normalize(te.float(), dim=-1).cpu().numpy())
        print(f"  custom precomputed {min(start + batch_size, len(ids))}/{len(ids)}")
    return np.concatenate(img_out), np.concatenate(txt_out)


def image_transform(image_size=224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def encode_custom_raw(lm, images, texts, batch_size, device):
    transform = image_transform(lm.hparams.image_resolution)
    img_out, txt_out = [], []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch_texts = texts[start:start + batch_size]
        x = torch.stack([transform(img) for img in batch_images]).to(device)
        tok = lm.tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=lm.hparams.context_length, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items() if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            ie = lm.model.encode_image(x)
            te = lm.model.encode_text(tok)
            img_out.append(F.normalize(ie.float(), dim=-1).cpu().numpy())
            txt_out.append(F.normalize(te.float(), dim=-1).cpu().numpy())
        print(f"  custom raw {min(start + batch_size, len(images))}/{len(images)}")
    return np.concatenate(img_out), np.concatenate(txt_out)


def load_georsclip(ckpt_path, arch, device):
    import open_clip
    model, _, transform = open_clip.create_model_and_transforms(arch, pretrained=None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  georsclip: {len(missing)} missing, {len(unexpected)} unexpected keys")
    tokenizer = open_clip.get_tokenizer(arch)
    return model.to(device).eval(), transform, tokenizer


def encode_georsclip(model, transform, tokenizer, images, texts, batch_size, device):
    img_out, txt_out = [], []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        batch_texts = texts[start:start + batch_size]
        x = torch.stack([transform(img) for img in batch_images]).to(device)
        tok = tokenizer(batch_texts).to(device)
        with torch.inference_mode():
            ie = model.encode_image(x)
            te = model.encode_text(tok)
            img_out.append(F.normalize(ie.float(), dim=-1).cpu().numpy())
            txt_out.append(F.normalize(te.float(), dim=-1).cpu().numpy())
        print(f"  georsclip {min(start + batch_size, len(images))}/{len(images)}")
    return np.concatenate(img_out), np.concatenate(txt_out)


def similarity_distribution(img_emb, txt_emb, bins, chunk):
    n = len(img_emb)
    pos_counts = np.zeros(len(bins) - 1, dtype=np.float64)
    all_counts = np.zeros(len(bins) - 1, dtype=np.float64)
    pos_sum = pos_sq = 0.0
    neg_sum = neg_sq = 0.0
    neg_count = 0

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sim = txt_emb @ img_emb[start:end].T
        rows = np.arange(start, end)
        cols = np.arange(end - start)
        pos = sim[rows, cols]

        pos_counts += np.histogram(pos, bins=bins)[0]
        all_counts += np.histogram(sim, bins=bins)[0]
        pos_sum += float(pos.sum())
        pos_sq += float((pos * pos).sum())
        neg_sum += float(sim.sum() - pos.sum())
        neg_sq += float((sim * sim).sum() - (pos * pos).sum())
        neg_count += sim.size - pos.size

    neg_counts = all_counts - pos_counts
    widths = np.diff(bins)

    def density(counts):
        total = counts.sum()
        return counts / (total * widths) if total else counts

    pos_mean = pos_sum / max(len(img_emb), 1)
    pos_std = np.sqrt(max(pos_sq / max(len(img_emb), 1) - pos_mean ** 2, 0.0))
    neg_mean = neg_sum / max(neg_count, 1)
    neg_std = np.sqrt(max(neg_sq / max(neg_count, 1) - neg_mean ** 2, 0.0))

    return {
        "pos_density": density(pos_counts),
        "neg_density": density(neg_counts),
        "pos_count": len(img_emb),
        "neg_count": neg_count,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "neg_mean": neg_mean,
        "neg_std": neg_std,
    }


def plot_results(results, bins, out):
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.6), squeeze=False)
    axes = axes[0]
    colors = {"pos": "#2a6fdb", "neg": "#e4572e"}

    for ax, (name, res) in zip(axes, results.items()):
        ax.fill_between(centers, res["pos_density"], step="mid", alpha=0.35,
                        color=colors["pos"])
        ax.plot(centers, res["pos_density"], drawstyle="steps-mid", lw=1.6,
                color=colors["pos"],
                label=f"positive (n={res['pos_count']:,}, μ={res['pos_mean']:.3f})")
        ax.fill_between(centers, res["neg_density"], step="mid", alpha=0.35,
                        color=colors["neg"])
        ax.plot(centers, res["neg_density"], drawstyle="steps-mid", lw=1.6,
                color=colors["neg"],
                label=f"negative (n={res['neg_count']:,}, μ={res['neg_mean']:.3f})")
        ax.axvline(res["pos_mean"], color=colors["pos"], ls="--", lw=1.0)
        ax.axvline(res["neg_mean"], color=colors["neg"], ls="--", lw=1.0)
        ax.set_title(name)
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("density")
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Matched vs. mismatched image-summary similarity", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    image_parquet = Path(args.image_parquet)
    caption_parquet = Path(args.caption_parquet)
    image_emb_dir = Path(args.image_emb_dir) if args.image_emb_dir else (
        image_parquet.parent / image_parquet.stem)
    caption_emb_dir = Path(args.caption_emb_dir) if args.caption_emb_dir else (
        caption_parquet.parent / caption_parquet.stem)

    print(f"Reading ids from {image_parquet} and captions from {caption_parquet}")
    merged = read_merged(image_parquet, caption_parquet, args)
    print(f"Selected {len(merged):,} matched image-summary pairs")

    if args.save_merged:
        merged.rename(columns={"_id": "file_name"}).to_parquet(args.save_merged, index=False)
        print(f"Saved merged rows to {args.save_merged}")

    ckpt = None
    precomputed = False
    if "custom" in args.models:
        ckpt = torch.load(args.custom_ckpt, map_location="cpu", weights_only=False)
        precomputed = custom_precomputed_mode(ckpt, args, image_emb_dir, caption_emb_dir)
        print(f"Custom model mode: {'precomputed embeddings' if precomputed else 'raw backbones'}")

    need_raw = "georsclip" in args.models or ("custom" in args.models and not precomputed)
    raw = {}
    if need_raw:
        img_cols = _columns(image_parquet)
        image_id = _pick(img_cols, args.image_id_col,
                         ["file_name", "id", "image_id", "key"], "image id")
        print(f"Reading raw images for {len(merged):,} rows from {image_parquet}")
        raw = load_raw_images(image_parquet, merged["_id"].tolist(), image_id, args.image_col)

    keep = pd.Series(True, index=merged.index)
    if need_raw:
        keep &= merged["_id"].map(lambda i: raw.get(i) is not None)
    if "custom" in args.models and precomputed:
        subdir = args.caption_emb_subdir
        keep &= merged["_id"].map(
            lambda i: (image_emb_dir / f"{i}.npy").is_file()
            and (caption_emb_dir / subdir / f"{i}.npy").is_file())
    merged = merged[keep].reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("no rows left after filtering for available images / embeddings")
    print(f"Using {len(merged):,} rows with all required inputs")

    ids = merged["_id"].tolist()
    texts = merged["summary"].tolist()
    images = [raw[i] for i in ids] if need_raw else None

    bins = np.linspace(args.sim_range[0], args.sim_range[1], args.bins + 1)
    results = {}

    if "custom" in args.models:
        lm = load_custom_model(ckpt, precomputed, device)
        if precomputed:
            img_emb, txt_emb = encode_custom_precomputed(
                lm, ids, image_emb_dir, caption_emb_dir, args.caption_emb_subdir,
                args.batch_size, device)
        else:
            img_emb, txt_emb = encode_custom_raw(
                lm, images, texts, args.batch_size, device)
        results["Custom CLIP"] = similarity_distribution(
            img_emb, txt_emb, bins, args.chunk)
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "georsclip" in args.models:
        model, transform, tokenizer = load_georsclip(
            args.georsclip_ckpt, args.georsclip_arch, device)
        img_emb, txt_emb = encode_georsclip(
            model, transform, tokenizer, images, texts, args.batch_size, device)
        results[f"GeoRSCLIP ({args.georsclip_arch})"] = similarity_distribution(
            img_emb, txt_emb, bins, args.chunk)

    for name, res in results.items():
        print(f"{name}: positive μ={res['pos_mean']:.4f} σ={res['pos_std']:.4f} | "
              f"negative μ={res['neg_mean']:.4f} σ={res['neg_std']:.4f}")

    plot_results(results, bins, args.out)


if __name__ == "__main__":
    main()
