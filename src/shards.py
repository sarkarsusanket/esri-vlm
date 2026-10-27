"""Torch-free access to the sharded image / caption / mask repo.

Layout (xx = shard id)::

    image-xx.parquet          image_bytes, file_name   (image_bytes only needed in bytes mode)
    image-xx/<file_stem>.npy  precomputed DINOv2 [CLS + patch] tokens, (1 + P, D) float32
                              (see preprocess/generate_dino_emb.py)
    caption-xx.parquet        file_name, summary, dense_caption, key_elements,
                              cf_dense_caption (negated dense paragraph, see generate_cf.py)
    caption-xx/summary/<file_name>.npy          precomputed Qwen EOS embedding, (1, H) fp16
    caption-xx/translated_col/<file_name>.npy    optional foreign-language summary embedding,
                                                (1, H) fp16; only ~30% of the rows have one
    caption-xx/dense_caption/<file_name>.npy    row 0 = whole paragraph, rows 1..S = sentences,
                                                (1 + S, H) fp16 (see preprocess/precompute_eos.py)
    caption-xx/cf_dense_caption/<file_name>.npy same layout; sentence j negates dense sentence j
                                                (absent while the cf precompute has not reached it)
    masks/mask-xx.csv         file_name, key_elements   (subset that has masks, in mask order)
    masks/mask-xx/<file_name>.npy   (K, H, W) masks, K == len(csv key_elements)

Two ingestion modes, selected by ``ShardIndex(precomputed=...)``:

* ``precomputed=True``  samples carry ``image_emb`` loaded from ``image-xx/``; image bytes are
  never read and no crop / flip augmentation is possible (the embeddings were computed once
  from the whole square-resized image). Records without an .npy are skipped.
* ``precomputed=False`` samples carry a decoded, cropped ``image`` tensor as before.

``cls_only=True`` (requires ``precomputed=True``; use it whenever masks / region loss are off)
reads ONLY the DINOv2 CLS token from ``cls-xx.npy`` (N, D) + ``cls-xx.names.txt`` written by
``precompute_cls.py``, instead of opening a (1 + P, D) ``.npy`` per sample. ``image_emb`` is then (D,).

``precomputed_text=True`` makes samples carry precomputed Qwen text embeddings instead of raw
caption strings: ``summary_emb`` (H,), ``translated_emb`` (H,) for the ~30% of records that have a
foreign-language summary, and when dense captions are enabled ``cap_emb`` (H, the
whole paragraph), ``cf_cap_emb`` (H, the whole counterfactual paragraph), ``dense_emb`` (K, H) the
selected sentence rows and ``hard_neg_emb`` (M, H) the counterfactual rows of
``cf_dense_caption`` (negatives are never mutated on the fly in this mode; ``cf`` rows that are not
precomputed yet simply yield neither ``cf_cap_emb`` nor a negative). Records without a
``summary/<stem>.npy`` are skipped. Keyword / region strings stay in the samples; the collator
turns them into precomputed EOS sums when a ``keyword_eos/`` lookup is configured.
"""
from __future__ import annotations

import io
import random
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from utils.crops import center_square_box, masks_to_grid, sample_resized_crop_box
from utils.negatives import NegativeCaptionGenerator
from utils.text import parse_key_elements, split_sentences, split_sentences_exact

# ImageNet statistics, matching the DINOv2 image tower's preprocessing.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# columns of preprocess/precompute_eos.py (only `summary` is required for a record to be usable,
# it feeds the global loss; dense/cf embeddings are optional per record)
TEXT_EMB_COLS = ("summary", "dense_caption", "cf_dense_caption")
# optional extra text states that not every record / shard has (only ~30% of the captions have a
# translated summary, and later shards may not have the directory at all). Missing files simply
# mean the record has no translation, so they must not skip the sample or warn.
OPTIONAL_TEXT_EMB_COLS = ("translated_col",)


class SkipSample(Exception):
    """Raised when a record cannot be materialised (e.g. a corrupt/truncated .npy).

    The dataset catches this and moves on to another index instead of crashing the worker.
    """


class Record(NamedTuple):
    shard: int
    row: int  # row inside image-xx.parquet
    file_name: str
    summary: str
    dense_caption: str
    key_elements: Tuple[str, ...]
    mask_keywords: Optional[Tuple[str, ...]]  # None -> no mask entry for this image
    cls_row: int = -1  # row inside cls-xx.npy (cls_only mode), else -1


@dataclass
class ShardInfo:
    name: str
    image_path: Path
    mask_dir: Path
    rg_ends: np.ndarray  # cumulative row count at the end of every row group
    emb_dir: Path  # precomputed DINOv2 embeddings for this shard (image-xx/)
    cap_dir: Path  # caption-xx/ holding the precomputed Qwen text embeddings
    text_stems: Dict[str, Set[str]] = field(default_factory=dict)  # col -> file stems, precomputed_text only


class ShardIndex:
    def __init__(self, root: str, image_col: str = "image_bytes", use_masks: bool = False,
                 mask_subdir: str = "masks", row_group_cache: int = 4,
                 precomputed: bool = False, verbose: bool = True, cls_only: bool = False,
                 precomputed_text: bool = False):
        if cls_only and not precomputed:
            raise ValueError("cls_only=True requires precomputed=True")
        if cls_only and use_masks:
            raise ValueError("cls_only=True drops the patch tokens, which the region loss needs; "
                             "set use_masks=False or cls_only=False")
        self.root = Path(root)
        self.image_col = image_col
        self.use_masks = use_masks
        self.precomputed = precomputed
        self.cls_only = cls_only
        self.precomputed_text = precomputed_text
        self._cls: Dict[int, np.ndarray] = {}  # lazily memory-mapped cls-xx.npy per shard
        self.row_group_cache = row_group_cache
        self.shards: List[ShardInfo] = []
        self.records: List[Record] = []
        self._pf: Dict[int, pq.ParquetFile] = {}
        self._cache: "OrderedDict[Tuple[int, int], object]" = OrderedDict()
        self._warned_mask = 0

        image_files = sorted(self.root.glob("image-*.parquet"))
        if not image_files:
            raise FileNotFoundError(f"no image-*.parquet found in {self.root}")

        n_missing_img = n_empty = n_masked = n_missing_emb = n_missing_txt_emb = 0
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

            emb_dir = self.root / p.stem
            emb_stems: set = set()
            cls_row_of: Dict[str, int] = {}
            if cls_only:
                names_path = self.root / f"cls-{name}.names.txt"
                if not (names_path.exists() and (self.root / f"cls-{name}.npy").exists()):
                    raise FileNotFoundError(
                        f"cls_only=True but cls-{name}.npy / cls-{name}.names.txt are missing in "
                        f"{self.root}; run `python precompute_cls.py {self.root}` first")
                for r, stem in enumerate(names_path.read_text().split("\n")):
                    cls_row_of.setdefault(stem, r)
                emb_stems = set(cls_row_of)
            elif precomputed:
                if emb_dir.is_dir():
                    # single scandir beats a Path.exists() per record on million-row shards
                    emb_stems = {e.stem for e in emb_dir.iterdir() if e.suffix == ".npy"}
                else:
                    warnings.warn(f"{emb_dir.name}/ missing: shard {name} has no precomputed embeddings")

            cap_dir = self.root / f"caption-{name}"
            text_stems: Dict[str, Set[str]] = {}
            if precomputed_text:
                for col in TEXT_EMB_COLS:
                    d = cap_dir / col
                    if d.is_dir():
                        text_stems[col] = {e.stem for e in d.iterdir() if e.suffix == ".npy"}
                    else:
                        text_stems[col] = set()
                        warnings.warn(f"{d.relative_to(self.root)}/ missing: shard {name} has no "
                                      f"precomputed '{col}' embeddings (run preprocess/precompute_eos.py)")
                for col in OPTIONAL_TEXT_EMB_COLS:
                    d = cap_dir / col
                    text_stems[col] = ({e.stem for e in d.iterdir() if e.suffix == ".npy"}
                                       if d.is_dir() else set())

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
            self.shards.append(ShardInfo(name, p, self.root / mask_subdir / f"mask-{name}",
                                         rg_ends, emb_dir, cap_dir, text_stems))

            caps = pd.read_parquet(cap_path, columns=["file_name", "summary", "dense_caption", "key_elements"])
            for fn, summ, dense, ke in zip(caps["file_name"], caps["summary"], caps["dense_caption"],
                                           caps["key_elements"]):
                fn = str(fn)
                row = row_of.get(fn)
                if row is None:
                    n_missing_img += 1
                    continue
                if precomputed and Path(fn).stem not in emb_stems:
                    n_missing_emb += 1
                    continue
                if precomputed_text and Path(fn).stem not in text_stems["summary"]:
                    n_missing_txt_emb += 1
                    continue
                summ = summ if isinstance(summ, str) else ""
                if not summ.strip():
                    n_empty += 1
                    continue
                mk = mask_map.get(fn)
                n_masked += mk is not None
                self.records.append(Record(
                    sid, row, fn, summ.strip(), dense if isinstance(dense, str) else "",
                    tuple(parse_key_elements(ke)), mk,
                    cls_row_of.get(Path(fn).stem, -1) if cls_only else -1))

        if verbose:
            print(f"[ShardIndex] {len(self.shards)} shards, {len(self.records)} samples "
                  f"(dropped: {n_missing_img} without image, {n_empty} without summary"
                  + (f", {n_missing_emb} without precomputed embedding" if precomputed else "") + ")"
                  + (f", {n_missing_txt_emb} without precomputed summary embedding"
                     if precomputed_text else "")
                  + (f"; {n_masked} with mask entries" if use_masks else ""))
            if precomputed and n_missing_emb:
                print(f"[ShardIndex] dropped samples get embeddings once preprocess/generate_dino_emb.py "
                      f"finishes; re-instantiate the index to pick them up")
            if precomputed_text and n_missing_txt_emb:
                print(f"[ShardIndex] dropped samples get text embeddings once preprocess/precompute_eos.py "
                      f"finishes; re-instantiate the index to pick them up")

    def __len__(self) -> int:
        return len(self.records)

    # --------------------------------------------------------------- pickling
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pf"], d["_cache"], d["_cls"] = {}, OrderedDict(), {}
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

    # ------------------------------------------------------- precomputed DINOv2
    def dino_emb(self, rec: Record) -> np.ndarray:
        """(1 + P, D) float32 [CLS + patch] embedding saved by preprocess/generate_dino_emb.py.

        In ``cls_only`` mode: the (D,) float32 CLS token from the consolidated cls-xx.npy instead.
        """
        if self.cls_only:
            arr = self._cls.get(rec.shard)
            if arr is None:  # mmap: pages are shared between dataloader workers via the page cache
                arr = self._cls[rec.shard] = np.load(
                    self.root / f"cls-{self.shards[rec.shard].name}.npy", mmap_mode="r")
            return np.array(arr[rec.cls_row], dtype=np.float32)  # copy out of the mmap
        d = self.shards[rec.shard].emb_dir
        p = d / f"{Path(rec.file_name).stem}.npy"
        if not p.exists():
            for cand in (rec.file_name, Path(rec.file_name).stem):  # be lenient about odd file_name
                alt = d / f"{cand}.npy"
                if alt.exists():
                    p = alt
                    break
            else:
                raise FileNotFoundError(f"{p} not found (shard {self.shards[rec.shard].name})")
        try:
            emb = np.load(p)
        except (EOFError, ValueError, OSError) as e:
            # truncated/empty/corrupt file: let the dataset skip this record
            raise SkipSample(f"{p} could not be loaded ({e})") from e
        if not np.isfinite(emb).all():
            # a corrupt/partial embedding that still happens to be the right size: normalising it
            # would produce NaNs and poison the whole run, so skip the record instead
            raise SkipSample(f"{p} contains non-finite values")
        return emb

    # ---------------------------------------------- precomputed Qwen text embeddings
    def text_emb(self, rec: Record, col: str) -> Optional[np.ndarray]:
        """(rows, H) float16 EOS embedding(s) written by preprocess/precompute_eos.py for `col`.

        ``summary`` files hold a single row; the paragraph columns (``dense_caption`` /
        ``cf_dense_caption``) hold row 0 = whole paragraph + rows 1..S = sentences, with the cf
        sentence j negating the dense sentence j. Returns None when this record has no embedding
        for `col` (empty text, or the precompute has not reached it yet) and raises SkipSample
        for unreadable / non-finite files.
        """
        if not self.precomputed_text:
            raise ValueError("text_emb() requires ShardIndex(precomputed_text=True)")
        sd = self.shards[rec.shard]
        stems = sd.text_stems.get(col, set())
        p = None
        for cand in (Path(rec.file_name).stem, rec.file_name):  # be lenient about odd file_name
            if cand in stems:
                p = sd.cap_dir / col / f"{cand}.npy"
                break
        if p is None:
            return None
        try:
            emb = np.load(p)
        except (EOFError, ValueError, OSError) as e:
            # truncated/empty/corrupt file: let the dataset skip this record
            raise SkipSample(f"{p} could not be loaded ({e})") from e
        if emb.ndim != 2:
            raise SkipSample(f"{p} has shape {emb.shape}, expected (rows, hidden_dim)")
        if not np.isfinite(emb).all():
            raise SkipSample(f"{p} contains non-finite values")
        return emb


    # ------------------------------------------------------------------- masks
    def load_masks(self, rec: Record) -> Optional[np.ndarray]:
        """(K, H, W) memory-mapped masks aligned with rec.mask_keywords, or None."""
        if not rec.mask_keywords:
            return None
        d = self.shards[rec.shard].mask_dir
        for cand in (rec.file_name, Path(rec.file_name).stem):
            p = d / f"{cand}.npy"
            if p.exists():
                try:
                    arr = np.load(p, mmap_mode="r")
                except (EOFError, ValueError, OSError) as e:
                    if self._warned_mask < 5:
                        warnings.warn(f"{rec.file_name}: mask file {p.name} could not be loaded ({e}); skipped")
                        self._warned_mask += 1
                    return None
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
    # optional translated summary embeddings (`caption-xx/translated_col/`); only ~30% of the
    # records have one, so they are loaded per record and never required
    use_translated: bool = True
    # with precomputed_text the negatives are the precomputed cf_dense_caption sentences
    # (hard_neg_per_image still caps them; mut_count / hardness only affect the rule-based
    # on-the-fly mutation used when text embeddings are NOT precomputed)
    use_hard_negatives: bool = True
    hard_neg_per_image: int = 3
    hard_neg_mut_count: int = 1
    hard_neg_hardness: float = 0.5
    use_keywords: bool = True
    # keyword groups: random positive combinations, absent-keyword negatives and swapped-keyword
    # hard negatives are built across the batch inside ClipCollator (see data.py); with a
    # precomputed keyword lookup each combination is the sum of its individual keyword EOS states
    keyword_subsets: int = 4        # positive combos sampled per image
    keyword_negs: int = 4           # absent single keywords used as negatives per image
    keyword_hard_negs: int = 3      # hard negatives (positive combo with swapped keywords) per image
    keyword_hard_neg_swap: int = 1  # keywords replaced when building a hard negative
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

    def _add_text_embs(self, out: dict, rec: Record, cfg: SampleConfig, rng: random.Random) -> None:
        """Precomputed Qwen EOS embeddings: nothing is tokenised and no negative is mutated here.

        ``summary_emb`` is the (H,) summary state. With dense captions enabled ``cap_emb`` is the
        whole-paragraph state (the summary stands in when there is none) and ``cf_cap_emb`` the
        whole-paragraph counterfactual state (row 0 of ``cf_dense_caption``). ``dense_emb`` holds
        the selected sentence rows and ``hard_neg_emb`` the counterfactual rows of
        ``cf_dense_caption`` (row-aligned with ``dense_caption``); cf rows that are not precomputed
        yet simply produce neither ``cf_cap_emb`` nor a sentence negative for that record.
        ``dense`` keeps the positive sentences as strings for inspection, but only when the parquet
        text still splits into exactly the embedded rows.
        """
        summ = self.index.text_emb(rec, "summary")
        if summ is None:
            raise SkipSample(f"{rec.file_name}: no precomputed summary embedding")
        summary_emb = np.asarray(summ[0], dtype=np.float32)
        out["summary_emb"] = summary_emb

        # Translated summary (foreign-language EOS state). Optional: only some records have one,
        # so a missing file/column just means this row carries no multilingual pair.
        if cfg.use_translated:
            tr = self.index.text_emb(rec, "translated_col")
            if tr is not None and tr.shape[0] > 0:
                if tr.shape[-1] != summary_emb.shape[-1]:
                    raise SkipSample(f"{rec.file_name}: translated_col embedding has dim "
                                     f"{tr.shape[-1]}, summary has {summary_emb.shape[-1]}")
                out["translated_emb"] = np.asarray(tr[0], dtype=np.float32)

        if not cfg.use_dense:
            return

        dim = summary_emb.shape[0]
        out["cap_emb"] = summary_emb          # stands in while the paragraph has no embedding
        out["dense_emb"] = np.zeros((0, dim), np.float32)
        out["hard_neg_emb"] = np.zeros((0, dim), np.float32)

        dense = self.index.text_emb(rec, "dense_caption")
        if dense is None or dense.shape[0] == 0:
            return  # empty dense caption: this sample only carries the summary / keyword groups
        if dense.shape[-1] != dim:
            raise SkipSample(f"{rec.file_name}: dense_caption embedding has dim {dense.shape[-1]}, "
                             f"summary has {dim}")

        n_rows = dense.shape[0]  # row 0 = paragraph, rows 1..S = sentences (precompute_eos layout)
        if n_rows - 1 > cfg.dense_max_sentences:
            sel = sorted(rng.sample(range(1, n_rows), cfg.dense_max_sentences))
        else:
            sel = list(range(1, n_rows))
        out["cap_emb"] = np.asarray(dense[0], dtype=np.float32)
        out["dense_emb"] = np.asarray(dense[sel], dtype=np.float32)

        # positive sentences as text for inspection; the row indices stay authoritative on drift
        sents = split_sentences_exact(rec.dense_caption)
        if len(sents) + 1 == n_rows:
            out["dense"] = [sents[j - 1] for j in sel]

        # generate_cf.py preserves the sentence boundaries, so cf row 0 negates the whole paragraph
        # and cf row j negates dense sentence j; a row-count mismatch means the cf paragraph drifted
        cf = self.index.text_emb(rec, "cf_dense_caption")
        if cf is not None and cf.shape == dense.shape:
            out["cf_cap_emb"] = np.asarray(cf[0], dtype=np.float32)  # paragraph-level negative
            if cfg.use_hard_negatives and sel:
                neg_sel = rng.sample(sel, min(cfg.hard_neg_per_image, len(sel)))
                out["hard_neg_emb"] = np.asarray(cf[neg_sel], dtype=np.float32)

    def build(self, record_idx: int) -> dict:
        cfg, rec = self.cfg, self.index.records[record_idx]
        # train: fresh randomness every call; val: deterministic per sample
        rng = random.Random(random.getrandbits(64) if cfg.train else 7919 * record_idx + 17)

        precomputed = self.index.precomputed
        box = flip = iw = ih = None
        if precomputed:
            emb = self.index.dino_emb(rec)
            n_tok = emb.shape[0] if emb.ndim == 2 else -1
            if self.index.cls_only:
                if emb.ndim != 1:
                    raise ValueError(f"{rec.file_name}: cls_only embedding has shape {emb.shape}, expected (D,)")
            elif n_tok != self.grid * self.grid + 1:
                raise ValueError(
                    f"{rec.file_name}: precomputed embedding has shape {emb.shape}, expected "
                    f"({self.grid * self.grid + 1}, D) for image_resolution={cfg.image_resolution} / "
                    f"patch_size={cfg.patch_size}. Re-run preprocess/generate_dino_emb.py or fix the config.")
            out = {"image_emb": np.ascontiguousarray(emb, dtype=np.float32), "summary": rec.summary,
                   "dense": [], "dense_text": "", "hard_negs": [], "keywords": [], "region_kws": [],
                   "region_cov": np.zeros((0, self.grid, self.grid), np.float32)}
        else:
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
                   "dense": [], "dense_text": "", "hard_negs": [], "keywords": [], "region_kws": [],
                   "region_cov": np.zeros((0, self.grid, self.grid), np.float32)}

        if self.index.precomputed_text:
            # summary / dense / counterfactual embeddings come from caption-xx/ .npy files
            self._add_text_embs(out, rec, cfg, rng)
        elif cfg.use_dense:
            out["dense_text"] = rec.dense_caption.strip()
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
            # the keywords that are actually present in this image; ClipCollator turns them into
            # positive combinations, absent-keyword negatives and swapped-keyword hard negatives
            out["keywords"] = sorted({k for k in rec.key_elements if k})

        if cfg.use_masks and rec.mask_keywords:
            masks = self.index.load_masks(rec)
            if masks is not None:
                sel = [k for k, w in enumerate(rec.mask_keywords) if w]
                if precomputed:
                    # embeddings came from the whole square-resized image: no crop, no flip, so
                    # the region grid is just the full mask mapped onto the patch grid.
                    hm, wm = masks.shape[-2:]
                    cov = masks_to_grid(masks, sel, (0, 0, wm, hm), (wm, hm), self.grid, False)
                else:
                    cov = masks_to_grid(masks, sel, box, (iw, ih), self.grid, flip)
                vis = [j for j in range(len(sel)) if cov[j].max() > 0]  # still visible after the crop
                if len(vis) > cfg.max_regions_per_image:
                    vis = sorted(rng.sample(vis, cfg.max_regions_per_image))
                out["region_kws"] = [rec.mask_keywords[sel[j]] for j in vis]
                out["region_cov"] = cov[vis]
        return out