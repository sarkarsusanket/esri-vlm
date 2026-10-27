"""Pretty-print exactly what the dataloader feeds the model.

Run::

    python tests/test_dataloader.py                      # config.yaml, first 2 train samples
    python tests/test_dataloader.py --n-samples 4 --batch-size 4
    python tests/test_dataloader.py --split val          # deterministic validation samples
    python tests/test_dataloader.py --use-masks          # force regions on even if config says off

It reads the same config.yaml as training (unless --config / --data-dir are given), builds the
real CLIPDataModule, then prints:

  * for a few raw samples: the image (ASCII preview + stats), the summary, the positive dense
    sentences, the hard negatives, the image keywords and the masked regions;
  * for one collated batch: every tensor key/shape and the decoded text behind the token groups,
    plus which dense sentences are positives for which image and how the keyword / region groups
    line up.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from data import CLIPDataModule  # noqa: E402

DEFAULT_CONFIG = SRC / "config.yaml"
RAMP = " .:-=+*#%@"


# --------------------------------------------------------------------- helpers
def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def ascii_image(img: Image.Image, width: int = 64) -> str:
    w = max(8, width)
    h = max(1, int(round(w * img.height / img.width * 0.5)))  # chars are ~2x taller than wide
    px = np.asarray(img.convert("L").resize((w, h)))
    step = 256 / len(RAMP)
    return "\n".join("".join(RAMP[min(len(RAMP) - 1, int(v / step))] for v in row) for row in px)


def decode_group(collator, tokens) -> list:
    ids, mask = tokens["input_ids"], tokens["attention_mask"]
    return [collator.tokenizer.decode(ids[j][mask[j].bool()], skip_special_tokens=True)
            for j in range(ids.shape[0])]


def fmt_tensor(t) -> str:
    if hasattr(t, "is_floating_point"):
        if t.dtype == torch.bool:
            return f"bool sum={int(t.sum())}"
        if t.is_floating_point():
            return f"{t.dtype} min={t.min().item():.3f} max={t.max().item():.3f} mean={t.mean().item():.3f}"
        return f"{t.dtype}"
    t = np.asarray(t)
    if t.dtype == np.bool_:
        return f"bool sum={int(t.sum())}"
    if np.issubdtype(t.dtype, np.floating):
        return f"{t.dtype} min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}"
    return f"{t.dtype}"


# ------------------------------------------------------------------ config glue
def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def datamodule_kwargs(cfg: dict, args) -> dict:
    m, d = cfg.get("model", {}), cfg.get("data", {})
    return dict(
        data_dir=args.data_dir or d.get("data_dir"),
        image_col=d.get("image_col", "image_bytes"),
        batch_size=args.batch_size,
        num_workers=0,  # keep workers out of the way so printing/ordering is deterministic
        val_fraction=d.get("val_fraction", 0.1),
        split_seed=d.get("split_seed", 0),
        image_resolution=m.get("image_resolution", 224),
        patch_size=m.get("vision_patch_size", 14),
        context_length=m.get("context_length", 256),
        tokenizer_name=m.get("tokenizer_name", "Qwen/Qwen3-Embedding-0.6B"),
        precomputed_image=m.get("precomputed_image", False),
        precomputed_text=m.get("precomputed_text", False),
        use_masks=args.use_masks if args.use_masks is not None else m.get("use_masks", False),
        use_dense=m.get("w_dense", 1.0) > 0,
        use_keywords=(args.use_keywords if args.use_keywords is not None
                      else m.get("use_keywords", True)),
        use_hard_negatives=m.get("use_hard_negatives", True),
        dense_max_sentences=d.get("dense_max_sentences", 8),
        hard_neg_per_image=d.get("hard_neg_per_image", 3),
        hard_neg_mut_count=d.get("hard_neg_mut_count", 1),
        hard_neg_hardness=d.get("hard_neg_hardness", 0.5),
        keyword_subsets=d.get("keyword_subsets", 4),
        keyword_negs=d.get("keyword_negs", 4),
        keyword_hard_negs=d.get("keyword_hard_negs", 3),
        keyword_hard_neg_swap=d.get("keyword_hard_neg_swap", 1),
        mask_subdir=d.get("mask_subdir", "masks"),
        max_regions_per_image=d.get("max_regions_per_image", 8),
        region_template=d.get("region_template", "{}"),
        keyword_emb_dir=d.get("keyword_emb_dir"),
        crop_scale=tuple(d.get("crop_scale", (0.5, 1.0))),
        hflip_prob=d.get("hflip_prob", 0.0),
        row_group_cache=d.get("row_group_cache", 4),
    )


# ------------------------------------------------------------------- printing
def source_image(dataset, k: int) -> Image.Image:
    index = dataset.builder.index
    rec = index.records[int(dataset.indices[k])]
    return Image.open(io.BytesIO(index.image_bytes(rec))).convert("RGB")


def show_raw_sample(dataset, k: int, show_image: bool) -> None:
    s = dataset[k]
    index = dataset.builder.index
    rec = index.records[int(dataset.indices[k])]

    hr(f"[raw sample {k}] {rec.file_name}  (shard {index.shards[rec.shard].name}, row {rec.row})")
    print(f"  summary      : {s['summary']!r}")
    if "summary_emb" in s:
        e = s["summary_emb"]
        print(f"  summary_emb  : {tuple(e.shape)} {fmt_tensor(e)}  (precomputed Qwen EOS state)")

    if "image_emb" in s:
        e = s["image_emb"]
        print(f"  image_emb    : {tuple(e.shape)} {fmt_tensor(e)}  (precomputed DINOv2 [CLS + patches])")
    else:
        im = s["image"]
        print(f"  image        : {tuple(im.shape)} {fmt_tensor(im)}  (normalised CHW)")

    if show_image:
        try:
            print(f"  source image : {source_image(dataset, k)}")
            print(ascii_image(source_image(dataset, k)))
        except Exception as exc:  # pragma: no cover - preview is best-effort
            print(f"  (image preview unavailable: {exc})")

    dense_rows = s.get("dense_emb")
    n_dense = len(dense_rows) if dense_rows is not None else len(s["dense"])
    if dense_rows is not None and "cap_emb" in s:
        print(f"  cap_emb      : {tuple(s['cap_emb'].shape)} {fmt_tensor(s['cap_emb'])}  (whole paragraph)")
    print(f"  positives    : {n_dense} dense sentence(s)")
    for j, sent in enumerate(s["dense"]):
        print(f"      + [{j}] {sent}")
    if dense_rows is not None and not s["dense"] and n_dense:
        print(f"      (sentence texts unavailable; {n_dense} precomputed embedding rows)")
    neg_rows = s.get("hard_neg_emb")
    if s["hard_negs"]:
        print(f"  hard negs    : {len(s['hard_negs'])}")
        for j, neg in enumerate(s["hard_negs"]):
            print(f"      - [{j}] {neg}")
    elif neg_rows is not None and len(neg_rows):
        print(f"  hard negs    : {len(neg_rows)} counterfactual embedding(s) "
              f"{tuple(neg_rows.shape)} (cf_dense_caption rows; text not loaded)")

    if s["keywords"]:
        print(f"  keywords     : {s['keywords']}")

    if s["region_kws"]:
        print(f"  regions      : {len(s['region_kws'])}  cov {tuple(s['region_cov'].shape)}")
        for j, kw in enumerate(s["region_kws"]):
            cov = torch.as_tensor(s["region_cov"][j])
            cells = int((cov > 0).sum())
            print(f"      @ [{j}] {kw!r}  coverage mean={cov.mean():.3f} max={cov.max():.3f} "
                  f"cells={cells}/{cov.numel()}")


def show_batch(collator, samples) -> None:
    batch = collator(samples)
    B = len(samples)
    hr(f"[collated batch] {B} sample(s) -> {len(batch)} tensor group(s)")

    print("  tensors")
    for key, val in batch.items():
        if key.endswith("_tokens"):
            n = val["input_ids"].shape[0]
            print(f"    {key:14s} {n} strings, input_ids={tuple(val['input_ids'].shape)} "
                  f"attention_mask={tuple(val['attention_mask'].shape)}")
        else:
            print(f"    {key:14s} {tuple(val.shape)}  {fmt_tensor(val)}")

    if "global_tokens" in batch:
        print("\n  global summaries")
        for j, txt in enumerate(decode_group(collator, batch["global_tokens"])):
            print(f"    [{j}] {txt}")
    elif "global_emb" in batch:
        print(f"\n  global summaries: {tuple(batch['global_emb'].shape)} precomputed embeddings "
              f"(not tokenised)")

    if "dense_tokens" in batch:
        texts = decode_group(collator, batch["dense_tokens"])
        pos = batch["dense_pos"]
        print(f"\n  dense sentences ({len(texts)} unique columns; positives per image from dense_pos)")
        for u, txt in enumerate(texts):
            owners = [i for i in range(B) if bool(pos[i, u])]
            tag = f"pos for {owners}" if owners else "hard negative (no positive image)"
            print(f"    [{u}] {tag:38s} {txt}")
    elif "dense_emb" in batch:
        emb, pos, neg = batch["dense_emb"], batch["dense_pos"], batch["dense_neg"]
        print(f"\n  dense sentences ({emb.shape[0]} unique embedding columns, {tuple(emb.shape)}; "
              f"positives per image from dense_pos, cf negatives from dense_neg)")
        for u in range(emb.shape[0]):
            owners = [i for i in range(B) if bool(pos[i, u])]
            negs = [i for i in range(B) if bool(neg[i, u])]
            tag = []
            if owners:
                tag.append(f"+ {owners}")
            if negs:
                tag.append(f"- {negs}")
            if not tag:
                tag.append("(unused)")
            print(f"    [{u}] {'  '.join(tag)}")

    if "kw_tokens" in batch:
        texts = decode_group(collator, batch["kw_tokens"])
        pos, neg = batch["kw_pos"], batch["kw_neg"]
        print(f"\n  keywords ({len(texts)} unique strings; + positive / - negative per image)")
        for u, txt in enumerate(texts):
            owners = [i for i in range(B) if bool(pos[i, u])]
            negs = [i for i in range(B) if bool(neg[i, u])]
            tag = []
            if owners:
                tag.append(f"+ {owners}")
            if negs:
                tag.append(f"- {negs}")
            print(f"    [{u}] {'  '.join(tag):38s} {txt}")
    elif "kw_emb" in batch:
        pos, neg = batch["kw_pos"], batch["kw_neg"]
        print(f"\n  keywords ({batch['kw_emb'].shape[0]} unique summed-EOS columns; "
              f"+ positive / - negative per image)")
        for u in range(batch["kw_emb"].shape[0]):
            owners = [i for i in range(B) if bool(pos[i, u])]
            negs = [i for i in range(B) if bool(neg[i, u])]
            tag = []
            if owners:
                tag.append(f"+ {owners}")
            if negs:
                tag.append(f"- {negs}")
            print(f"    [{u}] {'  '.join(tag)}")

    if "region_tokens" in batch or "region_emb" in batch:
        texts = (decode_group(collator, batch["region_tokens"])
                 if "region_tokens" in batch else None)
        cov, txt = batch["region_cov"], batch["region_txt"]
        n_unique = len(texts) if texts is not None else batch["region_emb"].shape[0]
        print(f"\n  regions ({n_unique} unique keyword(s); grid {cov.shape[-1]}x{cov.shape[-1]})")
        for i in range(B):
            for k in range(txt.shape[1]):
                t = int(txt[i, k])
                if t < 0:
                    continue
                c = cov[i, k]
                label = repr(texts[t]) if texts is not None else f"emb[{t}]"
                print(f"    image {i}  @{k}  {label}  coverage mean={c.mean():.3f} "
                      f"cells={int((c > 0).sum())}/{c.numel()}")


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default="/data/susanket/vlm/images3", help="override data.data_dir from the config")
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--n-samples", type=int, default=2, help="raw samples to print in detail")
    ap.add_argument("--batch-size", type=int, default=4, help="samples collated into the printed batch")
    ap.add_argument("--use-masks", dest="use_masks", action="store_true", default=None,
                    help="force regions on")
    ap.add_argument("--no-masks", dest="use_masks", action="store_false",
                    help="force regions off")
    ap.add_argument("--use-keywords", dest="use_keywords", action="store_true", default=None,
                    help="force keyword groups on")
    ap.add_argument("--no-keywords", dest="use_keywords", action="store_false",
                    help="force keyword groups off")
    ap.add_argument("--no-image", dest="show_image", action="store_false", default=True,
                    help="skip the ASCII image preview")
    args = ap.parse_args()

    cfg = load_config(args.config)
    kw = datamodule_kwargs(cfg, args)
    if not kw["data_dir"] or not Path(kw["data_dir"]).exists():
        sys.exit(f"data_dir not found: {kw['data_dir']!r} (set data.data_dir or pass --data-dir)")

    hr(f"config {args.config}")
    for key in ("data_dir", "image_resolution", "patch_size", "context_length", "precomputed_image",
                "precomputed_text", "use_masks", "use_dense", "use_keywords", "use_hard_negatives",
                "keyword_emb_dir", "batch_size"):
        print(f"  {key:20s} {kw[key]}")

    print("\nbuilding index / tokenizer ... (this scans the caption shards)")
    dm = CLIPDataModule(**kw)
    dm.setup()
    index = dm.index
    print(f"index: {len(index.shards)} shards, {len(index)} usable samples")
    dataset = dm.train_dataset if args.split == "train" else dm.val_dataset
    print(f"{args.split} split: {len(dataset)} samples; tokenizer "
          f"{kw['tokenizer_name']} (pad={dm.collator.tokenizer.pad_token!r})")

    n = min(args.n_samples, len(dataset))
    for k in range(n):
        show_raw_sample(dataset, k, args.show_image)

    m = min(args.batch_size, len(dataset))
    show_batch(dm.collator, [dataset[k] for k in range(m)])


if __name__ == "__main__":
    main()
