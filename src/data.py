import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "1,2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # tokenizing happens inside dataloader workers

import bisect
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import lightning.pytorch as pl

from shards import SampleBuilder, SampleConfig, ShardIndex, SkipSample
from utils.text import build_unique


class KeywordEmbeddings:
    """Memory-mapped ``preprocess/precompute_keyword.py`` lookup.

    ``keywords`` is sorted, so a keyword's row is found with a binary search and the backing
    ``keywords.npy`` matrix is never copied into the worker. ``sum_rows`` adds the raw EOS states
    of a keyword set, which is how the collator decomposes sampled keyword combinations.
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        with open(self.root / "keywords.json", "r", encoding="utf-8") as f:
            self.keywords: List[str] = json.load(f)
        self.embeddings = np.load(self.root / "keywords.npy", mmap_mode="r")
        if len(self.keywords) != len(self.embeddings):
            raise ValueError(f"{self.root}: {len(self.keywords)} keywords but "
                             f"{len(self.embeddings)} embedding rows")

    def __len__(self) -> int:
        return len(self.keywords)

    def row(self, keyword: str) -> int:
        i = bisect.bisect_left(self.keywords, keyword)
        if i >= len(self.keywords) or self.keywords[i] != keyword:
            raise KeyError(f"keyword {keyword!r} is not in {self.root}; rerun "
                           f"preprocess/precompute_keyword.py")
        return i

    def __getitem__(self, keyword: str) -> np.ndarray:
        return self.embeddings[self.row(keyword)]

    def sum_rows(self, keywords: Sequence[str]) -> np.ndarray:
        idx = [self.row(k) for k in keywords]
        return np.asarray(self.embeddings[idx], dtype=np.float32).sum(axis=0)

    def matrix(self, keyword_sets: Sequence[Sequence[str]]) -> np.ndarray:
        return np.stack([self.sum_rows(ks) for ks in keyword_sets]).astype(np.float32)


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
        image_emb       (B, 1 + P, D) or (B, D)        precomputed DINOv2 [CLS + patch] tokens, or CLS only
                                                        when masks are off (cls_only); preferred over
                                                        `images` when present (only one of the two is set)
        global_tokens   (B, L)                         summary caption; when text embeddings are NOT precomputed
        global_emb      (B, H)                         precomputed summary embedding (precompute_eos.py) instead
        cap_tokens      (B, L)                         whole dense caption (tokenised mode)
        cap_emb         (B, H)                         precomputed whole-paragraph embedding (summary stands in
                                                        for images without a dense caption)
        cf_cap_emb      (B, H)                         precomputed whole counterfactual paragraph (row 0 of
        cf_cap_valid    (B,) bool                      cf_dense_caption); cf_cap_valid marks which rows carry
                                                        a real state (the rest are placeholders, never read)
        translated_emb  (T, H)                         precomputed foreign-language summary embeddings, one row
        translated_img  (T,) long                      per record that has one (~30%); translated_img[t] is the
                                                         image row the translation belongs to
        dense_tokens    (Ud, L)                        UNIQUE dense sentences (tokenised mode), followed by
        dense_pos       (B, Ud) bool                   the hard-negative sentences; pos[i, u] = sentence u
        dense_neg       (B, Ud) bool                   is one of image i's sentences
        dense_emb       (Ud, H)                        the same group as precomputed embeddings: unique sentence
                                                        states (de-duplicated by content across the batch)
        kw_emb          (Uk, H)                        raw EOS state of each unique keyword set:
                                                        the sum of the individual keyword rows
        kw_tokens       (Uk, L)                        same group tokenised (only without a
                                                        precomputed keyword lookup)
        kw_pos          (B, Uk) bool                   image's keywords, absent single keywords and
        kw_neg          (B, Uk) bool                   swapped-keyword hard negatives
        region_emb      (Ur, H)                        raw EOS state of every unique region keyword
        region_tokens   (Ur, L)                        same group tokenised (fallback only)
        region_cov      (B, K, g, g) float             per-region fraction of every patch cell covered
        region_txt      (B, K) long                    index into region_emb / region_tokens, -1 = padding

    The summary / translated / dense / counterfactual groups come precomputed (`*_emb`, float32
    hidden states that the model projects) whenever the samples carry them. Keywords and regions
    use the precomputed individual keyword EOS states too when a lookup is configured; a sampled
    combination is represented by the sum of its keywords' EOS states, and the projectable sum is
    passed as `kw_emb` / `region_emb`. Dense: positives are de-duplicated (embedding content, so a
    sentence shared by several images is a positive for all of them - never a false negative); hard
    negatives are the precomputed counterfactual sentences, columns owned only by the images that
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
                 region_template: str = "{}",
                 keyword_emb_dir: Optional[Union[str, Path]] = None):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.grid = grid
        self.keyword_subsets = keyword_subsets
        self.keyword_negs = keyword_negs
        self.keyword_hard_negs = keyword_hard_negs
        self.keyword_hard_neg_swap = keyword_hard_neg_swap
        self.region_template = region_template
        self.keyword_emb_dir = keyword_emb_dir
        self._keyword_store: Optional[KeywordEmbeddings] = None
        self._keyword_store_loaded = keyword_emb_dir is None

    def _keywords(self) -> Optional[KeywordEmbeddings]:
        if not self._keyword_store_loaded:
            self._keyword_store = KeywordEmbeddings(self.keyword_emb_dir)
            self._keyword_store_loaded = True
        return self._keyword_store

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.context_length,
                             return_tensors="pt")
        return {k: enc[k] for k in ("input_ids", "attention_mask")}

    def _keyword_batch(self, keywords: List[List[str]], rng: random.Random):
        """Build the batch's unique keyword sets with content-based positive / owned-negative masks.

        Returns (uniq_sets, pos (B,U) bool, neg (B,U) bool). ``keywords[i]`` are the keywords that
        are present in image i. A set is a positive for every image whose keyword set contains all
        of its keywords (so shared combinations never become false negatives); it is a negative
        only for the images that actually generated it as one. The tuple order is canonical
        (sorted) so the collator can sum the individual EOS states deterministically.
        """
        B = len(keywords)
        present = [frozenset(ks) for ks in keywords]
        union = sorted({k for ks in keywords for k in ks})
        sets: Dict[frozenset, Tuple[str, ...]] = {}
        generated: List[Set[frozenset]] = [set() for _ in range(B)]

        for i, Kset in enumerate(present):
            K = sorted(Kset)
            absent = [k for k in union if k not in Kset]

            combos: List[Tuple[str, ...]] = []
            if K:
                for _ in range(self.keyword_subsets):
                    size = rng.randint(1, len(K))
                    sub = tuple(rng.sample(K, size))
                    key = frozenset(sub)
                    sets[key] = tuple(sorted(key))
                    generated[i].add(key)
                    combos.append(sub)

            if absent:
                for k in rng.sample(absent, min(self.keyword_negs, len(absent))):
                    key = frozenset((k,))
                    sets.setdefault(key, (k,))
                    generated[i].add(key)

            for _ in range(self.keyword_hard_negs):
                if not combos or not absent:
                    break
                base = list(rng.choice(combos))
                for _ in range(self.keyword_hard_neg_swap):
                    if absent:
                        base[rng.randrange(len(base))] = rng.choice(absent)
                key = frozenset(base)
                sets[key] = tuple(sorted(key))
                generated[i].add(key)

        uniq = sorted(sets, key=lambda key: tuple(sorted(key)))
        col = {key: j for j, key in enumerate(uniq)}
        pos = torch.zeros(B, len(uniq), dtype=torch.bool)
        neg = torch.zeros(B, len(uniq), dtype=torch.bool)

        postings: Dict[str, Set[int]] = {}
        for i, Kset in enumerate(present):
            for k in Kset:
                postings.setdefault(k, set()).add(i)
        for j, key in enumerate(uniq):
            if not key:
                continue
            owners = set.intersection(*(postings.get(k, set()) for k in key))
            for i in owners:
                pos[i, j] = True
        for i in range(B):
            for key in generated[i]:
                neg[i, col[key]] = True
        neg &= ~pos
        return [sets[k] for k in uniq], pos, neg

    @staticmethod
    def _emb_rows(sample: Dict[str, Any], key: str) -> np.ndarray:
        v = sample.get(key)
        return v if v is not None and len(v) else np.zeros((0, 0), np.float32)

    def _dense_emb_group(self, out: Dict[str, torch.Tensor], samples: List[Dict[str, Any]]) -> None:
        """Whole-caption + unique sentence embeddings with content positives / owned cf negatives.

        Columns are de-duplicated by embedding content (identical sentences share identical
        precomputed states), so a sentence shared by several images is a positive for all of them;
        a counterfactual column is owned only by the image that produced it, and one that collides
        with a real positive is dropped (it would be a false negative).
        """
        B = len(samples)
        if not any(len(self._emb_rows(s, "dense_emb")) for s in samples):
            return
        # the paragraph state stands in for samples without one (mirrors `dense_text or summary`)
        out["cap_emb"] = torch.from_numpy(np.stack([s.get("cap_emb", s["summary_emb"]) for s in samples]))

        # whole-paragraph counterfactual states: the paragraph-level negative for the cf-cap loss.
        # A missing state keeps a placeholder row (never read) flagged off by `cf_cap_valid`.
        cf_caps = [s.get("cf_cap_emb") for s in samples]
        if any(c is not None for c in cf_caps):
            ref = next(c for c in cf_caps if c is not None)
            out["cf_cap_emb"] = torch.from_numpy(np.stack([c if c is not None else ref for c in cf_caps]))
            out["cf_cap_valid"] = torch.tensor([c is not None for c in cf_caps], dtype=torch.bool)

        rows: List[np.ndarray] = []
        col_of: Dict[bytes, int] = {}

        def _add(key: bytes, row: np.ndarray) -> int:
            j = col_of.get(key)
            if j is None:
                j = col_of[key] = len(rows)
                rows.append(row)
            return j

        pos_keys: Set[bytes] = set()
        pos_pairs: List[Tuple[int, int]] = []
        for i, s in enumerate(samples):
            for row in self._emb_rows(s, "dense_emb"):
                r = np.ascontiguousarray(row, dtype=np.float32)
                k = r.tobytes()
                pos_keys.add(k)
                pos_pairs.append((i, _add(k, r)))

        neg_pairs: List[Tuple[int, int]] = []
        for i, s in enumerate(samples):
            for row in self._emb_rows(s, "hard_neg_emb"):
                r = np.ascontiguousarray(row, dtype=np.float32)
                k = r.tobytes()
                if k in pos_keys:
                    continue  # a cf twin identical to a real sentence is a positive, not a negative
                neg_pairs.append((i, _add(k, r)))

        if not rows:
            return
        out["dense_emb"] = torch.from_numpy(np.stack(rows))
        pos = torch.zeros(B, len(rows), dtype=torch.bool)
        neg = torch.zeros(B, len(rows), dtype=torch.bool)
        for i, j in pos_pairs:
            pos[i, j] = True
        for i, j in neg_pairs:
            neg[i, j] = True
        out["dense_pos"] = pos
        out["dense_neg"] = neg

    def _translated_emb_group(self, out: Dict[str, torch.Tensor], samples: List[Dict[str, Any]]) -> None:
        """Foreign-language summary states, stacked for the rows that actually have one.

        Missing translations are dropped rather than padded, so the loss gets a compact
        (T, H) bank plus `translated_img` mapping every row back to its image.
        """
        rows: List[np.ndarray] = []
        imgs: List[int] = []
        for i, s in enumerate(samples):
            v = s.get("translated_emb")
            if v is not None:
                rows.append(np.asarray(v, dtype=np.float32))
                imgs.append(i)
        if rows:
            out["translated_emb"] = torch.from_numpy(np.stack(rows))
            out["translated_img"] = torch.tensor(imgs, dtype=torch.long)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(samples)
        out: Dict[str, torch.Tensor] = {}
        if "summary_emb" in samples[0]:  # precomputed Qwen states (precompute_eos.py)
            out["global_emb"] = torch.from_numpy(np.stack([s["summary_emb"] for s in samples]))
        else:
            out["global_tokens"] = self.tokenize([s["summary"] for s in samples])
        if "image_emb" in samples[0]:
            out["image_emb"] = torch.stack([s["image_emb"] for s in samples])
        else:
            out["images"] = torch.stack([s["image"] for s in samples])

        # ---- dense: whole caption + individual sentences with owned hard negatives
        if "summary_emb" in samples[0]:
            self._dense_emb_group(out, samples)
            self._translated_emb_group(out, samples)
        elif any(s["dense"] for s in samples):
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
        keyword_store = self._keywords()
        if any(s["keywords"] for s in samples):
            key = "|".join(sorted({k for s in samples for k in s["keywords"]}))
            rng = random.Random(int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "little"))
            uniq, pos, neg = self._keyword_batch([s["keywords"] for s in samples], rng)
            if keyword_store is not None:
                out["kw_emb"] = torch.from_numpy(keyword_store.matrix(uniq))
            else:
                out["kw_tokens"] = self.tokenize([" ".join(kws) for kws in uniq])
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
            if keyword_store is not None:
                if self.region_template != "{}":
                    raise ValueError("precomputed keyword embeddings require region_template='{}'")
                out["region_emb"] = torch.from_numpy(keyword_store.matrix([(t,) for t in uniq]))
            else:
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
        precomputed_text: bool = False,
        use_masks: bool = False,
        use_dense: bool = True,
        use_translated: bool = True,
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
        # precomputed individual-keyword EOS states (preprocess/precompute_keyword.py);
        # relative paths are resolved against data_dir
        keyword_emb_dir: Optional[str] = None,
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
        keyword_emb_dir = None
        if h.use_keywords or h.use_masks:
            keyword_emb_dir = Path(h.keyword_emb_dir or "keyword_eos")
            if not keyword_emb_dir.is_absolute():
                keyword_emb_dir = Path(h.data_dir) / keyword_emb_dir
            missing = [p.name for p in (keyword_emb_dir / "keywords.npy",
                                        keyword_emb_dir / "keywords.json") if not p.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"precomputed keyword embeddings missing from {keyword_emb_dir}: {missing}; "
                    f"run `python preprocess/precompute_keyword.py {h.data_dir}` first")
        self.index = ShardIndex(h.data_dir, image_col=h.image_col, use_masks=h.use_masks,
                                mask_subdir=h.mask_subdir, row_group_cache=h.row_group_cache,
                                precomputed=h.precomputed_image,
                                # only the CLS token is used unless the region loss needs patches
                                cls_only=h.precomputed_image and not h.use_masks,
                                precomputed_text=h.precomputed_text)

        perm = np.random.RandomState(h.split_seed).permutation(len(self.index))
        n_val = int(len(perm) * h.val_fraction)
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        def cfg(train: bool) -> SampleConfig:
            return SampleConfig(
                image_resolution=h.image_resolution, patch_size=h.patch_size, train=train,
                crop_scale=tuple(h.crop_scale), hflip_prob=h.hflip_prob,
                use_dense=h.use_dense, dense_max_sentences=h.dense_max_sentences,
                use_translated=h.use_translated,
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
            h.region_template, keyword_emb_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True,
                          drop_last=True, collate_fn=self.collator)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True,
                          collate_fn=self.collator)