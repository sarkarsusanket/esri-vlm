"""SFT data: a directory of parquet files holding (image bytes, caption) pairs.

Each ``*.parquet`` in ``data_dir`` must contain an image-bytes column (``image_col``, default
``image_bytes``) or an image-path column (``image_path_col`` + ``image_root``) plus a caption
column (``caption_col``; auto-detected among common names when left null).

Sampling mirrors ``src/shards.py``: a small number of better image/caption pairs, random-resized
crop + optional flip at train time, a deterministic centre crop at validation time, ImageNet
normalisation. Random access re-reads a whole parquet row group on a cache miss, so writing the
parquets with ~256-1024 rows/group keeps it fast (a warning is printed otherwise).
"""
from __future__ import annotations

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "3")

import io
import json
import random
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import lightning.pytorch as pl  # noqa: E402

from utils.crops import center_square_box, sample_resized_crop_box  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CAPTION_CANDIDATES = ("caption", "captions", "summary", "dense_caption", "text",
                      "sentence", "caption_text")
IMAGE_CANDIDATES = ("image_bytes", "image", "jpg", "png", "image_data", "bytes")


# --------------------------------------------------------------------------- schema helpers
def _first_present(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    name_set = set(names)
    for c in candidates:
        if c in name_set:
            return c
    return None


def resolve_columns(files: Sequence[Path], image_col: Optional[str], caption_col: Optional[str],
                    image_path_col: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """Pick the image / caption / path columns, validating that every file exposes them."""
    schemas = [pq.ParquetFile(f).schema_arrow.names for f in files]
    common = set(schemas[0])
    for s in schemas[1:]:
        common &= set(s)

    cap = caption_col or _first_present(sorted(common), CAPTION_CANDIDATES)
    if cap is None or cap not in common:
        raise ValueError(f"no caption column found among {CAPTION_CANDIDATES}; columns present: "
                         f"{sorted(common)} (set data.caption_col explicitly)")

    path = image_path_col
    if path is None:
        img = image_col if (image_col and image_col in common) else _first_present(sorted(common), IMAGE_CANDIDATES)
        if img is None:
            raise ValueError(f"no image column found among {IMAGE_CANDIDATES}; columns present: "
                             f"{sorted(common)} (set data.image_col / data.image_path_col)")
    else:
        if path not in common:
            raise ValueError(f"image_path_col={path!r} not present in every parquet")
        img = None
    return img, cap, path


# --------------------------------------------------------------------------- transforms
class ImageTransform:
    """PIL -> normalised float32 (3, H, W) using the same scheme as ``src/shards.py``."""

    def __init__(self, train: bool, resolution: int, crop_scale: Tuple[float, float] = (0.5, 1.0),
                 hflip_prob: float = 0.0):
        self.train = train
        self.resolution = int(resolution)
        self.crop_scale = tuple(crop_scale)
        self.hflip_prob = float(hflip_prob)

    def __call__(self, img: Image.Image) -> np.ndarray:
        w, h = img.size
        if self.train:
            left, top, cw, ch = sample_resized_crop_box(w, h, self.crop_scale, rng=random)
            flip = random.random() < self.hflip_prob
        else:
            left, top, cw, ch = center_square_box(w, h)
            flip = False

        img = img.crop((left, top, left + cw, top + ch)).resize(
            (self.resolution, self.resolution), Image.Resampling.BICUBIC)
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return np.ascontiguousarray(arr.transpose(2, 0, 1))


# --------------------------------------------------------------------------- dataset
class ParquetCaptionDataset(Dataset):
    """(image, caption) pairs from a set of parquet files, addressing a shared record index.

    ``records`` is built once by :func:`build_records` and shared between the train / val splits;
    ``indices`` selects the subset this dataset exposes.
    """

    def __init__(self, files: Sequence[Path], records: List[Tuple[int, int, int, str, Optional[str]]],
                 rg_ends: List[np.ndarray], image_col: Optional[str], image_path_col: Optional[str],
                 image_root: Optional[str], caption_col: str, transform: ImageTransform,
                 indices: np.ndarray, row_group_cache: int = 4):
        self.files = [Path(f) for f in files]
        self.records = records
        self.rg_ends = rg_ends
        self.image_col = image_col
        self.image_path_col = image_path_col
        self.image_root = Path(image_root) if image_root else None
        self.caption_col = caption_col
        self.transform = transform
        self.indices = np.asarray(indices)
        self.row_group_cache = int(row_group_cache)
        self._pf: Dict[int, pq.ParquetFile] = {}
        self._cache: "OrderedDict[Tuple[int, int], Any]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_pf"], d["_cache"] = {}, OrderedDict()
        return d

    def _row_group_column(self, file_idx: int, rg: int):
        key = (file_idx, rg)
        col = self._cache.get(key)
        if col is not None:
            self._cache.move_to_end(key)
            return col
        pf = self._pf.get(file_idx)
        if pf is None:
            pf = self._pf[file_idx] = pq.ParquetFile(self.files[file_idx])
        col = pf.read_row_group(rg, columns=[self.image_col]).column(0)
        self._cache[key] = col
        while len(self._cache) > self.row_group_cache:
            self._cache.popitem(last=False)
        return col

    def _load_image(self, file_idx: int, rg: int, row_in_rg: int, path: Optional[str]) -> Image.Image:
        if path is not None:
            root = self.image_root or self.files[file_idx].parent
            return Image.open(root / path).convert("RGB")
        raw = self._row_group_column(file_idx, rg)[row_in_rg].as_py()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def __getitem__(self, i: int) -> Dict[str, Any]:
        file_idx, rg, row_in_rg, caption, path = self.records[int(self.indices[i])]
        img = self._load_image(file_idx, rg, row_in_rg, path)
        return {"image": self.transform(img), "caption": caption}


def build_records(files: Sequence[Path], caption_col: str,
                  image_path_col: Optional[str]) -> Tuple[List[Tuple[int, int, int, str, Optional[str]]],
                                                          List[np.ndarray]]:
    """Index every non-empty caption; only the caption (+ path) columns are read for this."""
    records: List[Tuple[int, int, int, str, Optional[str]]] = []
    rg_ends: List[np.ndarray] = []
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        ends = np.cumsum([pf.metadata.row_group(j).num_rows for j in range(pf.num_row_groups)])
        rg_ends.append(ends)
        if ends.size and int(np.diff(np.r_[0, ends]).max()) > 4096:
            warnings.warn(f"{Path(f).name}: row groups have >4096 rows; random access re-reads a "
                          f"whole row group per cache miss. Re-write with ~256-1024 rows/group.")
        cols = [caption_col] + ([image_path_col] if image_path_col else [])
        table = pf.read(columns=cols)
        caps = table.column(caption_col).to_pylist()
        paths = table.column(image_path_col).to_pylist() if image_path_col else None
        for row, cap in enumerate(caps):
            if cap is None or not str(cap).strip():
                continue
            rg = int(np.searchsorted(ends, row, side="right"))
            start = 0 if rg == 0 else int(ends[rg - 1])
            records.append((fi, rg, row - start, str(cap).strip(),
                            paths[row] if paths is not None else None))
    if not records:
        raise RuntimeError(f"no non-empty captions found in {len(files)} parquet file(s)")
    return records, rg_ends


class PrecomputedFeatureDataset(Dataset):
    """Indices into the cached feature matrices; the model owns the actual tensors.

    Returning only an index keeps the DataLoader trivial and allows the feature buffers to live
    on the GPU for the whole run (zero per-step host-to-device traffic).
    """

    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, int]:
        return {"idx": int(self.indices[i])}


# --------------------------------------------------------------------------- collator
class SFTCollator:
    """Stack images and tokenise captions into one batch (no python strings downstream)."""

    def __init__(self, tokenizer, context_length: int):
        self.tokenizer = tokenizer
        self.context_length = int(context_length)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([torch.from_numpy(s["image"]) for s in samples])
        enc = self.tokenizer([s["caption"] for s in samples], padding=True, truncation=True,
                             max_length=self.context_length, return_tensors="pt")
        tokens = {k: enc[k] for k in ("input_ids", "attention_mask")}
        return {"images": images, "tokens": tokens}


# --------------------------------------------------------------------------- datamodule
class SFTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_col: str = "image_bytes",
        caption_col: Optional[str] = None,
        image_path_col: Optional[str] = None,
        image_root: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        val_fraction: float = 0.05,
        split_seed: int = 0,
        crop_scale: Tuple[float, float] = (0.5, 1.0),
        hflip_prob: float = 0.0,
        row_group_cache: int = 4,
        drop_last: bool = True,
        precomputed_dir: Optional[str] = None,
        # linked from the model section (do not set under `data:`)
        image_resolution: int = 224,
        context_length: int = 256,
        tokenizer_name: str = "Qwen/Qwen3-Embedding-0.6B",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_set = None
        self.val_set = None
        self.collator = None

    def setup(self, stage: str = "fit") -> None:
        if self.train_set is not None:
            return
        h = self.hparams

        if h.precomputed_dir:
            root = Path(h.precomputed_dir)
            meta = json.loads((root / "meta.json").read_text())
            self.train_set = PrecomputedFeatureDataset(np.load(root / "train_idx.npy"))
            self.val_set = PrecomputedFeatureDataset(np.load(root / "val_idx.npy"))
            print(f"[sft.data] precomputed features {root}: "
                  f"{len(self.train_set)} train / {len(self.val_set)} val "
                  f"(image_dim={meta['image_dim']}, text_dim={meta['text_dim']})")
            return

        files = sorted(Path(h.data_dir).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no *.parquet files found in {h.data_dir}")

        image_col, caption_col, image_path_col = resolve_columns(
            files, h.image_col, h.caption_col, h.image_path_col)
        records, rg_ends = build_records(files, caption_col, image_path_col)

        perm = np.random.RandomState(h.split_seed).permutation(len(records))
        n_val = int(len(perm) * h.val_fraction)
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        common = dict(files=files, records=records, rg_ends=rg_ends, image_col=image_col,
                      image_path_col=image_path_col, image_root=h.image_root,
                      caption_col=caption_col, row_group_cache=h.row_group_cache)
        self.train_set = ParquetCaptionDataset(
            indices=train_idx,
            transform=ImageTransform(True, h.image_resolution, h.crop_scale, h.hflip_prob),
            **common)
        self.val_set = ParquetCaptionDataset(
            indices=val_idx,
            transform=ImageTransform(False, h.image_resolution),
            **common)

        tokenizer = AutoTokenizer.from_pretrained(h.tokenizer_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.collator = SFTCollator(tokenizer, h.context_length)

        print(f"[sft.data] {len(files)} parquet file(s), {len(records)} pairs "
              f"(train {len(train_idx)} / val {len(val_idx)}); "
              f"image_col={image_col or image_path_col} caption_col={caption_col}")

    def _loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        if self.collator is None:  # precomputed feature indices: no workers / no pinning
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
                              num_workers=0, drop_last=drop_last)
        workers = int(self.hparams.num_workers)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=workers,
                          shuffle=shuffle, pin_memory=True, drop_last=drop_last,
                          collate_fn=self.collator, persistent_workers=workers > 0,
                          prefetch_factor=(4 if workers > 0 else None))

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True, drop_last=self.hparams.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False, drop_last=False)
