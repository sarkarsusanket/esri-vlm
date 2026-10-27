"""ClipCollator batch-layout tests (needs torch + lightning). Run: python tests/test_collator.py"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from data import ClipCollator, KeywordEmbeddings  # noqa: E402

H = 6  # fake hidden dim


class StubTokenizer:
    """Enough of the tokenizer surface for ClipCollator: one token per word, right-padded."""

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        ids = [[i + 1] for i, _ in enumerate(texts)]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 1, dtype=torch.long)}


def emb(v: float) -> np.ndarray:
    return np.full(H, v, np.float32)


def base(i: int, **kw):
    s = {"summary": f"summary {i}", "dense": [], "dense_text": "", "hard_negs": [],
         "keywords": [f"kw{i}a", f"kw{i}b"], "region_kws": [],
         "region_cov": np.zeros((0, 2, 2), np.float32),
         "image_emb": torch.full((5, 4), float(i))}  # dataset path already hands over tensors
    s.update(kw)
    return s


def test_precomputed_dense_group():
    """Shared positive sentences collapse to one column; cf negatives are owned, not global."""
    coll = ClipCollator(StubTokenizer(), context_length=8, grid=2)
    s0 = base(0, summary_emb=emb(1.0), cap_emb=emb(2.0),
              dense_emb=np.stack([emb(3.0), emb(4.0)]),      # A, B
              hard_neg_emb=np.stack([emb(5.0)]))             # C
    s1 = base(1, summary_emb=emb(1.1), cap_emb=emb(2.1),
              dense_emb=np.stack([emb(4.0), emb(6.0)]),      # B (shared), D
              hard_neg_emb=np.stack([emb(5.0), emb(4.0)]))   # C (shared) + B == positive -> dropped
    batch = coll([s0, s1])

    assert "global_emb" in batch and batch["global_emb"].shape == (2, H)
    assert "global_tokens" not in batch and "dense_tokens" not in batch
    assert "cap_emb" in batch and batch["cap_emb"].shape == (2, H)
    assert "dense_emb" in batch and batch["dense_emb"].shape == (4, H)  # A, B, D, C
    pos, neg = batch["dense_pos"], batch["dense_neg"]
    assert pos.shape == (2, 4) and neg.shape == (2, 4)
    col = {float(batch["dense_emb"][u, 0]): u for u in range(4)}
    assert pos[0, col[3.0]] and not pos[1, col[3.0]]         # A owned by image 0
    assert pos[0, col[4.0]] and pos[1, col[4.0]]             # B is a positive for both
    assert pos[1, col[6.0]] and not pos[0, col[6.0]]         # D owned by image 1
    assert neg[0, col[5.0]] and neg[1, col[5.0]]             # shared cf owned by both images
    assert not pos[:, col[5.0]].any()                        # C is never a positive
    assert not neg[0, col[4.0]]                              # B==positive was dropped for image 0
    assert int(pos.sum()) == 4 and int(neg.sum()) == 2       # A,B,B,D positives; C twice
    assert batch["kw_pos"].shape[0] == 2                     # keywords still tokenised
    assert batch["image_emb"].shape == (2, 5, 4)


def test_precomputed_cf_paragraph_group():
    """Whole-paragraph cf states ride along as (B, H) with an availability mask."""
    coll = ClipCollator(StubTokenizer(), context_length=8, grid=2)
    s0 = base(0, summary_emb=emb(1.0), cap_emb=emb(2.0),
              dense_emb=np.stack([emb(3.0)]), cf_cap_emb=emb(9.0))
    s1 = base(1, summary_emb=emb(1.1), cap_emb=emb(2.1),
              dense_emb=np.stack([emb(4.0)]))  # no cf paragraph
    batch = coll([s0, s1])
    assert batch["cf_cap_emb"].shape == (2, H)
    assert batch["cf_cap_valid"].tolist() == [True, False]
    assert float(batch["cf_cap_emb"][0, 0]) == 9.0  # real state kept for the valid row

    # no cf state anywhere -> the group is absent (loss silently skipped)
    batch2 = coll([base(2, summary_emb=emb(1.0), cap_emb=emb(2.0),
                        dense_emb=np.stack([emb(3.0)]))])
    assert "cf_cap_emb" not in batch2 and "cf_cap_valid" not in batch2


def test_precomputed_empty_dense_group():
    """No dense rows at all -> only the global group is built (no empty dense_emb)."""
    coll = ClipCollator(StubTokenizer(), context_length=8, grid=2)
    s0 = base(0, summary_emb=emb(1.0), dense_emb=np.zeros((0, H), np.float32),
              hard_neg_emb=np.zeros((0, H), np.float32))
    batch = coll([s0])
    assert "global_emb" in batch and batch["global_emb"].shape == (1, H)
    assert "dense_emb" not in batch and "cap_emb" not in batch
    assert "dense_pos" not in batch


def test_precomputed_translated_group():
    """Translations are a compact bank with a row->image mapping; missing ones are dropped."""
    coll = ClipCollator(StubTokenizer(), context_length=8, grid=2)
    s0 = base(0, summary_emb=emb(1.0), translated_emb=emb(7.0))
    s1 = base(1, summary_emb=emb(1.1))  # no translation
    s2 = base(2, summary_emb=emb(1.2), translated_emb=emb(8.0))
    batch = coll([s0, s1, s2])
    assert batch["translated_emb"].shape == (2, H)
    assert batch["translated_img"].tolist() == [0, 2]
    assert batch["translated_emb"][:, 0].tolist() == [7.0, 8.0]

    batch2 = coll([s1])
    assert "translated_emb" not in batch2 and "translated_img" not in batch2


def test_tokenised_mode_unchanged():
    """Samples without *_emb keys keep the old string path."""
    coll = ClipCollator(StubTokenizer(), context_length=8, grid=2)
    s0 = base(0, dense=["sent a.", "sent b."], dense_text="sent a. sent b.",
              hard_negs=["mut a."], summary="summary 0")
    s1 = base(1, dense=["sent b."], dense_text="sent b.", hard_negs=[], summary="summary 1")
    batch = coll([s0, s1])

    assert "global_tokens" in batch and "global_emb" not in batch
    assert "dense_tokens" in batch and "dense_emb" not in batch
    n_u = batch["dense_tokens"]["input_ids"].shape[0]
    assert n_u == 3  # "sent a.", "sent b.", "mut a." (negatives share the dense token group)
    assert batch["dense_pos"].shape == (2, n_u)
    assert batch["dense_neg"].shape == (2, n_u)
    assert batch["dense_neg"][0].sum() == 1                  # one mutated twin


def test_precomputed_keyword_sums():
    """Keyword combinations and region texts use the summed individual EOS rows."""
    words = ["kw0a", "kw0b", "kw1a", "kw1b", "other a", "other b", "other c", "road"]
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows = np.arange(len(words) * H, dtype=np.float32).reshape(len(words), H) + 1
        np.save(root / "keywords.npy", rows.astype(np.float16))
        with open(root / "keywords.json", "w") as f:
            json.dump(sorted(words), f)
        store = KeywordEmbeddings(root)
        coll = ClipCollator(StubTokenizer(), context_length=8, grid=2, keyword_emb_dir=root)
        s0 = base(0, region_kws=["road"], region_cov=np.ones((1, 2, 2), np.float32))
        s1 = base(1)
        batch = coll([s0, s1])
        assert "kw_emb" in batch and "kw_tokens" not in batch
        assert "region_emb" in batch and "region_tokens" not in batch

        samples = [s0, s1]
        key = "|".join(sorted({k for s in samples for k in s["keywords"]}))
        rng = random.Random(int.from_bytes(hashlib.md5(key.encode()).digest()[:8], "little"))
        uniq, pos, neg = coll._keyword_batch([s["keywords"] for s in samples], rng)
        expect = np.stack([store.sum_rows(kws) for kws in uniq])
        assert np.allclose(batch["kw_emb"].numpy(), expect)
        assert torch.equal(batch["kw_pos"], pos) and torch.equal(batch["kw_neg"], neg)
        assert np.allclose(batch["region_emb"][0].numpy(), store.sum_rows(("road",)))


def run_all():
    for fn in (test_precomputed_dense_group, test_precomputed_cf_paragraph_group,
               test_precomputed_translated_group, test_precomputed_empty_dense_group,
               test_tokenised_mode_unchanged, test_precomputed_keyword_sums):
        fn()
        print("PASS", fn.__name__)


if __name__ == "__main__":
    run_all()
