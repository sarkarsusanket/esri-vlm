import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenizing happens inside dataloader workers

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizerFast

import lightning.pytorch as pl

from shards import SampleBuilder, SampleConfig, ShardIndex
from utils.text import build_unique


class AerialClipDataset(Dataset):
    """Thin torch wrapper around SampleBuilder (all the logic lives in shards.py)."""

    def __init__(self, builder: SampleBuilder, indices: np.ndarray):
        self.builder = builder
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = self.builder.build(int(self.indices[i]))
        sample["image"] = torch.from_numpy(sample["image"])
        return sample


class ClipCollator:
    """Turns a list of samples into ONE batch of tensors (no python strings), tokenised in the worker.

    Batch layout (B images)::

        images          (B, 3, H, W)
        global_tokens   (B, L)                         summary caption of every image
        dense_tokens    (Ud, L)                        UNIQUE dense sentences of the batch, followed by
        dense_pos       (B, Ud) bool                   the hard-negative sentences; pos[i, u] = sentence u
                                                       is one of image i's sentences (hard negatives
                                                       have no positive image)
        kw_tokens       (Uk, L)                        unique negated-keyword sentences
        kw_img, kw_txt  (N,)                           pair n = (image kw_img[n], sentence kw_txt[n])
        region_tokens   (Ur, L)                        unique keyword texts of the masked regions
        region_cov      (B, K, g, g) float             per-region fraction of every patch cell covered
        region_txt      (B, K) long                    index into region_tokens, -1 = padding

    Identical strings are merged ("de-duplicated") before they become columns, so a sentence /
    keyword shared by several images is a positive for all of them and never a false negative.
    A group is left out of the dict when the batch has nothing for it.
    """

    def __init__(self, tokenizer, context_length: int, grid: int,
                 keyword_template: str = "there is no {}", region_template: str = "{}"):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.grid = grid
        self.keyword_template = keyword_template
        self.region_template = region_template

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.context_length,
                              return_tensors="pt")["input_ids"]

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(samples)
        out: Dict[str, torch.Tensor] = {
            "images": torch.stack([s["image"] for s in samples]),
            "global_tokens": self.tokenize([s["summary"] for s in samples]),
        }

        # ---- dense sentences (+ hard negatives as extra columns without positives)
        if any(s["dense"] for s in samples):
            uniq, idx = build_unique([s["dense"] for s in samples])
            seen = set(uniq)
            hard: List[str] = []
            for s in samples:
                for h in s["hard_negs"]:
                    if h not in seen:  # a mutated sentence that equals a real one stays a real one
                        seen.add(h)
                        hard.append(h)
            pos = torch.zeros(B, len(uniq) + len(hard), dtype=torch.bool)
            for i, ix in enumerate(idx):
                if ix:
                    pos[i, ix] = True
            out["dense_tokens"] = self.tokenize(uniq + hard)
            out["dense_pos"] = pos

        # ---- negated keywords
        if any(s["neg_keywords"] for s in samples):
            uniq, idx = build_unique([[self.keyword_template.format(k) for k in s["neg_keywords"]] for s in samples])
            out["kw_tokens"] = self.tokenize(uniq)
            out["kw_img"] = torch.tensor([i for i, ix in enumerate(idx) for _ in ix], dtype=torch.long)
            out["kw_txt"] = torch.tensor([j for ix in idx for j in ix], dtype=torch.long)

        # ---- masked regions
        if any(s["region_kws"] for s in samples):
            fmt = [[self.region_template.format(k) for k in s["region_kws"]] for s in samples]
            uniq, _ = build_unique(fmt)
            table = {t: j for j, t in enumerate(uniq)}
            K = max(len(f) for f in fmt)
            cov = torch.zeros(B, K, self.grid, self.grid)
            txt = torch.full((B, K), -1, dtype=torch.long)
            for i, (s, f) in enumerate(zip(samples, fmt)):
                n = len(f)  # one slot per region, even if a keyword repeats inside one image
                if n:
                    cov[i, :n] = torch.from_numpy(s["region_cov"])
                    txt[i, :n] = torch.tensor([table[t] for t in f])
            out["region_tokens"] = self.tokenize(uniq)
            out["region_cov"] = cov
            out["region_txt"] = txt
        return out


class CLIPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_col: str = "image_bytes",
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        split_seed: int = 0,
        # linked from the model section in main.py (do not set them under `data:`)
        image_resolution: int = 224,
        patch_size: int = 32,
        context_length: int = 256,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        use_masks: bool = False,
        use_dense: bool = True,
        use_keywords: bool = True,
        use_hard_negatives: bool = True,
        # dense captions / hard negatives
        dense_max_sentences: int = 8,
        hard_neg_per_image: int = 3,
        hard_neg_mut_count: int = 1,
        hard_neg_hardness: float = 0.5,
        # negated keywords
        keywords_per_image: int = 3,
        keyword_template: str = "there is no {}",
        # regions
        mask_subdir: str = "masks",
        max_regions_per_image: int = 8,
        region_template: str = "{}",
        # augmentation
        crop_scale: Tuple[float, float] = (0.5, 1.0),
        hflip_prob: float = 0.0,
        # io
        row_group_cache: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.index: Optional[ShardIndex] = None

    def setup(self, stage: str = "fit"):
        if self.index is not None:
            return
        h = self.hparams
        self.index = ShardIndex(h.data_dir, image_col=h.image_col, use_masks=h.use_masks,
                                mask_subdir=h.mask_subdir, row_group_cache=h.row_group_cache)

        perm = np.random.RandomState(h.split_seed).permutation(len(self.index))
        n_val = int(len(perm) * h.val_fraction)
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        def cfg(train: bool) -> SampleConfig:
            return SampleConfig(
                image_resolution=h.image_resolution, patch_size=h.patch_size, train=train,
                crop_scale=tuple(h.crop_scale), hflip_prob=h.hflip_prob,
                use_dense=h.use_dense, dense_max_sentences=h.dense_max_sentences,
                use_hard_negatives=h.use_hard_negatives and h.use_dense,
                hard_neg_per_image=h.hard_neg_per_image, hard_neg_mut_count=h.hard_neg_mut_count,
                hard_neg_hardness=h.hard_neg_hardness,
                use_keywords=h.use_keywords, keywords_per_image=h.keywords_per_image,
                use_masks=h.use_masks, max_regions_per_image=h.max_regions_per_image)

        self.train_dataset = AerialClipDataset(SampleBuilder(self.index, cfg(True)), train_idx)
        self.val_dataset = AerialClipDataset(SampleBuilder(self.index, cfg(False)), val_idx)
        self.collator = ClipCollator(
            CLIPTokenizerFast.from_pretrained(h.tokenizer_name), h.context_length,
            h.image_resolution // h.patch_size, h.keyword_template, h.region_template)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True,
                          drop_last=True, collate_fn=self.collator)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True,
                          collate_fn=self.collator)
