"""Torch-free access to the sharded image / caption / mask repo.

Layout (xx = shard id)::

    image-xx.parquet          image_bytes, file_name
    caption-xx.parquet        file_name, summary, dense_caption, key_elements
    masks/mask-xx.csv         file_name, key_elements   (subset that has masks, in mask order)
    masks/mask-xx/<file_name>.npy   (K, H, W) masks, K == len(csv key_elements)
"""
from __future__ import annotations

import io
import random
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from utils.crops import center_square_box, masks_to_grid, sample_resized_crop_box
from utils.negatives import NegativeCaptionGenerator
from utils.text import parse_key_elements, split_sentences

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class Record(NamedTuple):
    shard: int
    row: int  # row inside image-xx.parquet
    file_name: str
    summary: str
    dense_caption: str
    key_elements: Tuple[str, ...]
    mask_keywords: Optional[Tuple[str, ...]]  # None -> no mask entry for this image


@dataclass
class ShardInfo:
    name: str
    image_path: Path
    mask_dir: Path
    rg_ends: np.ndarray  # cumulative row count at the end of every row group


class ShardIndex:
    def __init__(self, root: str, image_col: str = "image_bytes", use_masks: bool = False,
                 mask_subdir: str = "masks", row_group_cache: int = 4, verbose: bool = True):
        self.root = Path(root)
        self.image_col = image_col
        self.use_masks = use_masks
        self.row_group_cache = row_group_cache
        self.shards: List[ShardInfo] = []
        self.records: List[Record] = []
        self._pf: Dict[int, pq.ParquetFile] = {}
        self._cache: "OrderedDict[Tuple[int, int], object]" = OrderedDict()
        self._warned_mask = 0

        image_files = sorted(self.root.glob("image-*.parquet"))
        if not image_files:
            raise FileNotFoundError(f"no image-*.parquet found in {self.root}")

        n_missing_img = n_empty = n_masked = 0
        for p in image_files:
            name = p.stem[len("image-"):]
            cap_path = self.root / f"caption-{name}.parquet"
            if not cap_path.exists():
                warnings.warn(f"{cap_path.name} missing, skipping shard {name}")
                continue

            pf = pq.ParquetFile(p)
            fn_col = pf.read(columns=["file_name"]).column("file_name").to_pylist()
            row_of: Dict[str, int] = {}
            for r, fn in enumerate(fn_col):
                row_of.setdefault(str(fn), r)
            rg_ends = np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
            if verbose and rg_ends.size and int(np.diff(np.r_[0, rg_ends]).max()) > 4096:
                warnings.warn(f"{p.name}: row groups have >4096 rows; random access re-reads a whole "
                              f"row group per cache miss. Re-write with ~256-1024 rows/group for speed.")

            mask_map: Dict[str, Tuple[str, ...]] = {}
            if use_masks:
                csv_path = self.root / mask_subdir / f"mask-{name}.csv"
                if csv_path.exists():
                    mdf = pd.read_csv(csv_path)
                    mask_map = {str(f): tuple(parse_key_elements(k))
                                for f, k in zip(mdf["file_name"], mdf["key_elements"])}
                else:
                    warnings.warn(f"{csv_path} missing: shard {name} has no masks")

            sid = len(self.shards)
            self.shards.append(ShardInfo(name, p, self.root / mask_subdir / f"mask-{name}", rg_ends))

            caps = pd.read_parquet(cap_path, columns=["file_name", "summary", "dense_caption", "key_elements"])
            for fn, summ, dense, ke in zip(caps["file_name"], caps["summary"], caps["dense_caption"],
                                           caps["key_elements"]):
                fn = str(fn)
                row = row_of.get(fn)
                if row is None:
                    n_missing_img += 1
                    continue
                summ = summ if isinstance(summ, str) else ""
                if not summ.strip():
                    n_empty += 1
                    continue
                mk = mask_map.get(fn)
                n_masked += mk is not None
                self.records.append(Record(
                    sid, row, fn, summ.strip(), dense if isinstance(dense, str) else "",
                    tuple(parse_key_elements(ke)), mk))

        if verbose:
            print(f"[ShardIndex] {len(self.shards)} shards, {len(self.records)} samples "
                  f"(dropped: {n_missing_img} without image, {n_empty} without summary)"
                  + (f", {n_masked} with mask entries" if use_masks else ""))

    def __len__(self) -> int:
        return len(self.records)

    # --------------------------------------------------------------- pickling
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pf"], d["_cache"] = {}, OrderedDict()
        return d

    # ------------------------------------------------------------------ images
    def _row_group_column(self, shard: int, rg: int):
        key = (shard, rg)
        col = self._cache.get(key)
        if col is not None:
            self._cache.move_to_end(key)
            return col
        pf = self._pf.get(shard)
        if pf is None:
            pf = self._pf[shard] = pq.ParquetFile(self.shards[shard].image_path)
        col = pf.read_row_group(rg, columns=[self.image_col]).column(0)
        self._cache[key] = col
        while len(self._cache) > self.row_group_cache:
            self._cache.popitem(last=False)
        return col

    def image_bytes(self, rec: Record) -> bytes:
        ends = self.shards[rec.shard].rg_ends
        rg = int(np.searchsorted(ends, rec.row, side="right"))
        start = 0 if rg == 0 else int(ends[rg - 1])
        return self._row_group_column(rec.shard, rg)[rec.row - start].as_py()

    # ------------------------------------------------------------------- masks
    def load_masks(self, rec: Record) -> Optional[np.ndarray]:
        """(K, H, W) memory-mapped masks aligned with rec.mask_keywords, or None."""
        if not rec.mask_keywords:
            return None
        d = self.shards[rec.shard].mask_dir
        for cand in (rec.file_name, Path(rec.file_name).stem):
            p = d / f"{cand}.npy"
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                break
        else:
            return None
        k = len(rec.mask_keywords)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim == 3 and arr.shape[0] != k and arr.shape[-1] == k:
            arr = arr.transpose(2, 0, 1)
        if arr.ndim != 3 or arr.shape[0] != k:
            if self._warned_mask < 5:
                warnings.warn(f"{rec.file_name}: mask array {arr.shape} does not match {k} keywords; skipped")
                self._warned_mask += 1
            return None
        return arr


@dataclass
class SampleConfig:
    image_resolution: int = 224
    patch_size: int = 32
    train: bool = True
    crop_scale: Tuple[float, float] = (0.5, 1.0)
    hflip_prob: float = 0.0
    use_dense: bool = True
    dense_max_sentences: int = 8
    use_hard_negatives: bool = True
    hard_neg_per_image: int = 3
    hard_neg_mut_count: int = 1
    hard_neg_hardness: float = 0.5
    use_keywords: bool = True
    keywords_per_image: int = 3
    use_masks: bool = False
    max_regions_per_image: int = 8


class SampleBuilder:
    """Builds one training sample (numpy / python objects only)."""

    def __init__(self, index: ShardIndex, cfg: SampleConfig):
        self.index, self.cfg = index, cfg
        self.grid = cfg.image_resolution // cfg.patch_size
        self._gen: Optional[NegativeCaptionGenerator] = None

    def _generator(self) -> NegativeCaptionGenerator:
        if self._gen is None:  # rng is always passed explicitly, so one shared generator is fine
            self._gen = NegativeCaptionGenerator(seed=0, hardness=self.cfg.hard_neg_hardness)
        return self._gen

    def build(self, record_idx: int) -> dict:
        cfg, rec = self.cfg, self.index.records[record_idx]
        # train: fresh randomness every call; val: deterministic per sample
        rng = random.Random(random.getrandbits(64) if cfg.train else 7919 * record_idx + 17)

        img = Image.open(io.BytesIO(self.index.image_bytes(rec))).convert("RGB")
        iw, ih = img.size
        if cfg.train:
            box = sample_resized_crop_box(iw, ih, cfg.crop_scale, rng=rng)
            flip = rng.random() < cfg.hflip_prob
        else:
            box, flip = center_square_box(iw, ih), False
        l, t, cw, ch = box
        res = cfg.image_resolution
        img = img.crop((l, t, l + cw, t + ch)).resize((res, res), Image.Resampling.BICUBIC)
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        out = {"image": np.ascontiguousarray(arr.transpose(2, 0, 1)), "summary": rec.summary,
               "dense": [], "hard_negs": [], "neg_keywords": [], "region_kws": [],
               "region_cov": np.zeros((0, self.grid, self.grid), np.float32)}

        if cfg.use_dense:
            sents = split_sentences(rec.dense_caption)
            if len(sents) > cfg.dense_max_sentences:
                sents = [sents[i] for i in sorted(rng.sample(range(len(sents)), cfg.dense_max_sentences))]
            out["dense"] = sents
            if cfg.use_hard_negatives and sents:
                gen = self._generator()
                picks = rng.sample(sents, min(cfg.hard_neg_per_image, len(sents)))
                for s in picks:
                    r = gen.generate(s, mut_count=cfg.hard_neg_mut_count, rng=rng)
                    # a "negative" identical to the positive would be a false negative: drop it
                    if r["n_mutations"] > 0 and r["negative"].strip() != s.strip():
                        out["hard_negs"].append(r["negative"].strip())

        if cfg.use_keywords:
            pool = sorted({k for k in rec.key_elements if k})
            out["neg_keywords"] = rng.sample(pool, min(cfg.keywords_per_image, len(pool)))

        if cfg.use_masks and rec.mask_keywords:
            masks = self.index.load_masks(rec)
            if masks is not None:
                sel = [k for k, w in enumerate(rec.mask_keywords) if w]
                cov = masks_to_grid(masks, sel, box, (iw, ih), self.grid, flip)
                vis = [j for j in range(len(sel)) if cov[j].max() > 0]  # still visible after the crop
                if len(vis) > cfg.max_regions_per_image:
                    vis = sorted(rng.sample(vis, cfg.max_regions_per_image))
                out["region_kws"] = [rec.mask_keywords[sel[j]] for j in vis]
                out["region_cov"] = cov[vis]
        return out
