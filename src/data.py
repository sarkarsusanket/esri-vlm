import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenizing happens inside dataloader workers

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import lightning.pytorch as pl

from shards import SampleBuilder, SampleConfig, ShardIndex, SkipSample
from utils.text import build_unique


class AerialClipDataset(Dataset):
    """Thin torch wrapper around SampleBuilder (logic in in shards.py)."""

    def __init__(self, builder: SampleBuilder, indices: np.ndarray):
        self.builder = builder
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        n = len(self.indices)
        for attempt in range(min(n, 64)):
            j = int(self.indices[(i + attempt) % n])
            try:
                sample = self.builder.build(j)
                break
            except SkipSample:
                continue  # unreadable file: ignore it and try the next record
        else:
            raise RuntimeError(f"no readable sample found among {n} records (all corrupt?)")
        key = "image_emb" if "image_emb" in sample else "image"
        sample[key] = torch.from_numpy(sample[key])
        return sample


class ClipCollator:
    """Turns a list of samples into ONE batch of tensors (no python strings), tokenised in the worker.

    Batch layout (B images)::

        images          (B, 3, H, W)                   raw pixels; only when image embeddings are NOT precomputed
        image_emb       (B, 1 + P, D)                  precomputed DINOv2 [CLS + patch] tokens; preferred
                                                        over `images` when present (only one of the two is set)
        global_tokens   (B, L)                         summary caption of every image
        cap_tokens      (B, L)                         the whole dense caption of every image
        dense_tokens    (Ud, L)                        UNIQUE dense sentences of the batch, followed by
        dense_pos       (B, Ud) bool                   the hard-negative sentences; pos[i, u] = sentence u
        dense_neg       (B, Ud) bool                   is one of image i's sentences
        kw_tokens       (Uk, L)                        UNIQUE keyword texts: positive combinations of the
        kw_pos          (B, Uk) bool                   image's keywords, absent single keywords and
        kw_neg          (B, Uk) bool                   swapped-keyword hard negatives
        region_tokens   (Ur, L)                        unique keyword texts of the masked regions
        region_cov      (B, K, g, g) float             per-region fraction of every patch cell covered
        region_txt      (B, K) long                    index into region_tokens, -1 = padding

    Dense: positives are de-duplicated so a sentence shared by several images is a positive for
    all of them (never a false negative); hard negatives are columns owned only by the images that
    produced them. Keywords: positives are combinations of the image's own keywords (marked by
    content, so a combination that is true for another image is a positive there too); negatives
    are absent keywords and swapped-keyword hard negatives owned by the image. A group is left out
    of the dict when the batch has nothing for it.

    Keyword sampling is seeded from the batch's keyword content, so it is deterministic for a
    fixed batch (nice for validation) but still varies across shuffled training batches.
    """

    def __init__(self, tokenizer, context_length: int, grid: int,
                 keyword_subsets: int = 4, keyword_negs: int = 4,
                 keyword_hard_negs: int = 3, keyword_hard_neg_swap: int = 1,
                 region_template: str = "{}"):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.grid = grid
        self.keyword_subsets = keyword_subsets
        self.keyword_negs = keyword_negs
        self.keyword_hard_negs = keyword_hard_negs
        self.keyword_hard_neg_swap = keyword_hard_neg_swap
        self.region_template = region_template

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.context_length,
                             return_tensors="pt")
        return {k: enc[k] for k in ("input_ids", "attention_mask")}

    def _keyword_batch(self, keywords: List[List[str]], rng: random.Random):
        """Build the batch's unique keyword texts with content-based positive / owned-negative masks.

        Returns (uniq_texts, pos (B,U) bool, neg (B,U) bool). ``keywords[i]`` are the keywords that
        are present in image i. A text is a positive for every image whose keyword set contains all
        of the text's keywords (so shared combinations never become false negatives); it is a
        negative only for the images that actually generated it as one.
        """
        B = len(keywords)
        union = sorted({k for ks in keywords for k in ks})
        text_of: Dict[str, frozenset] = {}
        generated = [set() for _ in range(B)]

        for i, present in enumerate(keywords):
            K = list(dict.fromkeys(present))
            Kset = set(K)
            absent = [k for k in union if k not in Kset]

            combos: List[List[str]] = []
            if K:
                for _ in range(self.keyword_subsets):
                    size = rng.randint(1, len(K))
                    sub = rng.sample(K, size)
                    text_of[" ".join(sub)] = frozenset(sub)
                    generated[i].add(" ".join(sub))
                    combos.append(sub)

            if absent:
                for k in rng.sample(absent, min(self.keyword_negs, len(absent))):
                    text_of.setdefault(k, frozenset([k]))
                    generated[i].add(k)

            for _ in range(self.keyword_hard_negs):
                if not combos or not absent:
                    break
                base = list(rng.choice(combos))
                for _ in range(self.keyword_hard_neg_swap):
                    if absent:
                        base[rng.randrange(len(base))] = rng.choice(absent)
                txt = " ".join(base)
                text_of[txt] = frozenset(base)
                generated[i].add(txt)

        uniq = list(text_of)
        pos = torch.zeros(B, len(uniq), dtype=torch.bool)
        neg = torch.zeros(B, len(uniq), dtype=torch.bool)
        for i, present in enumerate(keywords):
            Kset = set(present)
            for u, txt in enumerate(uniq):
                kws = text_of[txt]
                if kws and kws <= Kset:
                    pos[i, u] = True
                elif txt in generated[i]:
                    neg[i, u] = True
        return uniq, pos, neg

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(samples)
        out: Dict[str, torch.Tensor] = {
            "global_tokens": self.tokenize([s["summary"] for s in samples]),
        }
        if "image_emb" in samples[0]:
            out["image_emb"] = torch.stack([s["image_emb"] for s in samples])
        else:
            out["images"] = torch.stack([s["image"] for s in samples])

        # ---- dense: whole caption + individual sentences with owned hard negatives
        if any(s["dense"] for s in samples):
            out["cap_tokens"] = self.tokenize([s.get("dense_text") or s["summary"] for s in samples])
            uniq, idx = build_unique([s["dense"] for s in samples])
            positives = set(uniq)
            texts = list(uniq)
            hard_col: Dict[str, int] = {}
            owners: Dict[int, set] = {}
            for i, s in enumerate(samples):
                for h in s["hard_negs"]:
                    if h in positives:  # a mutated sentence that equals a real one stays a real one
                        continue
                    if h not in hard_col:
                        hard_col[h] = len(texts)
                        texts.append(h)
                    owners.setdefault(hard_col[h], set()).add(i)
            pos = torch.zeros(B, len(texts), dtype=torch.bool)
            for i, ix in enumerate(idx):
                for j in ix:
                    pos[i, j] = True
            neg = torch.zeros(B, len(texts), dtype=torch.bool)
            for u, imgs in owners.items():
                for i in imgs:
                    neg[i, u] = True
            out["dense_tokens"] = self.tokenize(texts)
            out["dense_pos"] = pos
            out["dense_neg"] = neg

        # ---- keywords: positive combinations, absent-keyword negatives, swapped hard negatives
        if any(s["keywords"] for s in samples):
            key = "|".join(sorted({k for s in samples for k in s["keywords"]}))
            rng = random.Random(int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "little"))
            uniq, pos, neg = self._keyword_batch([s["keywords"] for s in samples], rng)
            out["kw_tokens"] = self.tokenize(uniq)
            out["kw_pos"] = pos
            out["kw_neg"] = neg

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
        tokenizer_name: str = "Qwen/Qwen3-Embedding-0.6B",
        precomputed_image: bool = False,
        use_masks: bool = False,
        use_dense: bool = True,
        use_keywords: bool = True,
        use_hard_negatives: bool = True,
        # dense captions / hard negatives
        dense_max_sentences: int = 8,
        hard_neg_per_image: int = 3,
        hard_neg_mut_count: int = 1,
        hard_neg_hardness: float = 0.5,
        # keyword groups (positive combinations / absent-keyword negatives / swapped hard negatives)
        keyword_subsets: int = 4,
        keyword_negs: int = 4,
        keyword_hard_negs: int = 3,
        keyword_hard_neg_swap: int = 1,
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
                                mask_subdir=h.mask_subdir, row_group_cache=h.row_group_cache,
                                precomputed=h.precomputed_image)

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
                use_keywords=h.use_keywords, keyword_subsets=h.keyword_subsets,
                keyword_negs=h.keyword_negs, keyword_hard_negs=h.keyword_hard_negs,
                keyword_hard_neg_swap=h.keyword_hard_neg_swap,
                use_masks=h.use_masks, max_regions_per_image=h.max_regions_per_image)

        self.train_dataset = AerialClipDataset(SampleBuilder(self.index, cfg(True)), train_idx)
        self.val_dataset = AerialClipDataset(SampleBuilder(self.index, cfg(False)), val_idx)
        tokenizer = AutoTokenizer.from_pretrained(h.tokenizer_name, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.collator = ClipCollator(
            tokenizer, h.context_length, h.image_resolution // h.patch_size,
            h.keyword_subsets, h.keyword_negs, h.keyword_hard_negs, h.keyword_hard_neg_swap,
            h.region_template)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True,
                          drop_last=True, collate_fn=self.collator)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True,
                          collate_fn=self.collator)
