"""Torch-free tests for the shard / mask / sample pipeline. Run: python tests/test_data_io.py"""
import io, os, sys, tempfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shards import SampleBuilder, SampleConfig, ShardIndex, SkipSample  # noqa: E402
from utils.crops import masks_to_grid, center_square_box, sample_resized_crop_box  # noqa: E402
from utils.text import split_sentences_exact  # noqa: E402


def png(w, h, color):
    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, format="PNG")
    return b.getvalue()


DENSE = ("A red roof house has a swimming pool to the north. Several cars are parked along the road. "
         "The field is green. Dense forest covers the east side.")

# hidden dim of the fake precomputed Qwen EOS embeddings written by make_repo(with_text_emb=True)
TEXT_H = 4


def make_repo(root: Path, with_emb: bool = False, skip_emb=(), with_text_emb: bool = False,
              skip_text=()):
    (root / "masks").mkdir(parents=True)
    for sh, n in (("00", 10), ("01", 7)):
        names = [f"{sh}_{i}.jpg" for i in range(n)]
        pd.DataFrame({"image_bytes": [png(300 + 10 * i, 200, (i * 20 % 255, 10, 10)) for i in range(n)],
                      "file_name": names}).to_parquet(root / f"image-{sh}.parquet", row_group_size=4)
        if with_emb:  # preprocessed DINOv2 [CLS + patch] embs: grid 7 -> 49 patches, dim 8
            emb_dir = root / f"image-{sh}"
            emb_dir.mkdir()
            for fn in names:
                if fn in skip_emb:
                    continue
                rng = np.random.default_rng(abs(hash(fn)) % 2 ** 32)
                np.save(emb_dir / f"{Path(fn).stem}.npy", rng.standard_normal((50, 8)).astype(np.float32))
        if with_text_emb:  # precompute_eos.py layout: summary (1, H), dense/cf (1 + S, H) fp16
            n_rows = 1 + len(split_sentences_exact(DENSE))
            for col in ("summary", "dense_caption", "cf_dense_caption"):
                (root / f"caption-{sh}" / col).mkdir(parents=True, exist_ok=True)
            for i, fn in enumerate(names):
                if fn in skip_text:
                    continue
                rng = np.random.default_rng(abs(hash("txt" + fn)) % 2 ** 32)
                stem = Path(fn).stem
                dense = rng.standard_normal((n_rows, TEXT_H))
                np.save(root / f"caption-{sh}" / "summary" / f"{stem}.npy",
                        rng.standard_normal((1, TEXT_H)).astype(np.float16))
                np.save(root / f"caption-{sh}" / "dense_caption" / f"{stem}.npy", dense.astype(np.float16))
                np.save(root / f"caption-{sh}" / "cf_dense_caption" / f"{stem}.npy",
                        (dense + 100.0).astype(np.float16))
        # caption parquet in a different row order, one extra file without image, one empty summary
        cap_names = names[::-1] + ["ghost.jpg"]
        pd.DataFrame({"file_name": cap_names,
                      "summary": ["a summary"] * (n - 1) + ["   ", "x"],
                      "dense_caption": [DENSE] * (n + 1),
                      "key_elements": ["['red building', 'road', 'solar panel']"] * (n + 1)}
                     ).to_parquet(root / f"caption-{sh}.parquet")
        (root / "masks" / f"mask-{sh}").mkdir()
        rows = []
        for i, fn in enumerate(names):
            if i == 3:
                continue  # image without mask entry
            rows.append({"file_name": fn, "key_elements": "['road', 'red building']"})
            m = np.zeros((2, 400, 600), dtype=bool)  # NOTE: different resolution than the image
            m[0, :, :300] = True    # 'road' = left half
            m[1, 200:, 300:] = True  # 'red building' = bottom-right quadrant
            np.save(root / "masks" / f"mask-{sh}" / f"{fn}.npy", m)
        pd.DataFrame(rows).to_csv(root / "masks" / f"mask-{sh}.csv", index=False)


def test_masks_to_grid():
    m = np.zeros((2, 400, 600), bool)
    m[0, :, :300] = True
    m[1, 200:, 300:] = True
    # image 300x200 -> mask is 2x the image; full-image box
    cov = masks_to_grid(m, [0, 1], (0, 0, 300, 200), (300, 200), 4)
    assert cov.shape == (2, 4, 4)
    assert np.allclose(cov[0][:, :2], 1) and np.allclose(cov[0][:, 2:], 0)
    assert np.allclose(cov[1][2:, 2:], 1) and np.allclose(cov[1][:2], 0)
    # flip mirrors columns
    f = masks_to_grid(m, [0], (0, 0, 300, 200), (300, 200), 4, flip=True)
    assert np.allclose(f[0][:, 2:], 1) and np.allclose(f[0][:, :2], 0)
    # crop that straddles the boundary, non-divisible sizes -> fractions
    c = masks_to_grid(m, [0], (50, 0, 200, 200), (300, 200), 7)
    assert 0 <= c.min() and c.max() <= 1 and 0 < c.mean() < 1
    # crop smaller than grid falls back to nearest sampling without crashing
    t = masks_to_grid(m, [0, 1], (10, 10, 2, 2), (300, 200), 7)
    assert t.shape == (2, 7, 7)
    # mean coverage ~ true area fraction on a random mask
    rng = np.random.default_rng(0)
    r = rng.random((1, 333, 517)) > 0.7
    g = masks_to_grid(r, [0], (0, 0, 517, 333), (517, 333), 7)
    assert abs(g.mean() - r.mean()) < 0.02


def test_crop_boxes():
    import random
    rng = random.Random(0)
    for _ in range(500):
        l, t, w, h = sample_resized_crop_box(300, 200, (0.5, 1.0), rng=rng)
        assert 0 <= l and 0 <= t and l + w <= 300 and t + h <= 200 and w > 0 and h > 0
        assert w * h >= 0.45 * 300 * 200
    assert center_square_box(300, 200) == (50, 0, 200, 200)


def test_index_and_samples():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_repo(root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = ShardIndex(str(root), use_masks=True, row_group_cache=2)
        # per shard: n captions-with-image minus one empty summary ('ghost' has no image)
        assert len(idx) == (10 - 1) + (7 - 1), len(idx)
        # joined by file_name, not by position: bytes must decode to the matching image width
        for rec in idx.records:
            i = int(rec.file_name.split("_")[1].split(".")[0])
            assert Image.open(io.BytesIO(idx.image_bytes(rec))).size[0] == 300 + 10 * i, rec
        # LRU cache respects its cap; pickling drops handles
        assert len(idx._cache) <= 2
        import pickle
        idx2 = pickle.loads(pickle.dumps(idx))
        assert idx2.image_bytes(idx2.records[0]) == idx.image_bytes(idx.records[0])

        no_mask = [r for r in idx.records if r.mask_keywords is None]
        assert {r.file_name for r in no_mask} == {"00_3.jpg", "01_3.jpg"}

        cfg = SampleConfig(train=True, use_masks=True, crop_scale=(0.9, 1.0))
        b = SampleBuilder(idx, cfg)
        s = b.build(0)
        assert s["image"].shape == (3, 224, 224) and s["image"].dtype == np.float32
        assert len(s["dense"]) == 4 and s["summary"] == "a summary"
        assert 1 <= len(s["hard_negs"]) <= 3
        assert all(h not in s["dense"] for h in s["hard_negs"])
        assert set(s["keywords"]) == {"red building", "road", "solar panel"}
        assert s["region_cov"].shape == (len(s["region_kws"]), 7, 7) and set(s["region_kws"]) == {"road", "red building"}
        k = s["region_kws"].index("road")
        assert s["region_cov"][k][:, :3].mean() > 0.9 and s["region_cov"][k][:, 5:].mean() < 0.1  # road is on the left

        # sample without a mask entry -> no regions, everything else intact
        i_nm = idx.records.index(no_mask[0])
        s2 = b.build(i_nm)
        assert s2["region_kws"] == [] and s2["region_cov"].shape == (0, 7, 7)

        # validation is deterministic
        vb = SampleBuilder(idx, SampleConfig(train=False, use_masks=True))
        a, c = vb.build(2), vb.build(2)
        assert np.array_equal(a["image"], c["image"]) and a["hard_negs"] == c["hard_negs"] \
            and a["keywords"] == c["keywords"]
        # train is stochastic (hard negatives are sampled / re-mutated every call)
        outs = {tuple(b.build(1)["hard_negs"]) for _ in range(20)}
        assert len(outs) > 1

        # toggles really switch work off
        off = SampleBuilder(idx, SampleConfig(train=False, use_dense=False, use_keywords=False, use_masks=False)).build(0)
        assert off["dense"] == [] and off["hard_negs"] == [] and off["keywords"] == [] and off["region_kws"] == []


def test_precomputed_embeddings():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_repo(root, with_emb=True, skip_emb={"01_2.jpg"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = ShardIndex(str(root), precomputed=True, use_masks=True, row_group_cache=2)
        # 01_2 has no .npy -> dropped on top of the usual empty-summary / ghost drops
        assert len(idx) == (10 - 1) + (7 - 1) - 1, len(idx)
        assert "01_2.jpg" not in {r.file_name for r in idx.records}
        # nothing is read from the image parquet: bytes access is never touched
        for rec in idx.records:
            assert (idx.shards[rec.shard].emb_dir / f"{Path(rec.file_name).stem}.npy").exists()

        b = SampleBuilder(idx, SampleConfig(train=False, use_masks=True))
        s = b.build(0)
        assert "image" not in s and s["image_emb"].shape == (50, 8) and s["image_emb"].dtype == np.float32
        # full mask (no crop / no flip): 'road' still covers the left third of the patch grid
        k = s["region_kws"].index("road")
        assert s["region_cov"][k][:, :3].mean() > 0.9 and s["region_cov"][k][:, 5:].mean() < 0.1

        # a mismatching token count is rejected with a clear error
        i0 = next(i for i, r in enumerate(idx.records) if r.file_name == "00_1.jpg")
        np.save(idx.shards[0].emb_dir / "00_1.npy", np.zeros((17, 8), np.float32))
        try:
            b.build(i0)
            assert False, "expected ValueError for a wrong precomputed embedding shape"
        except ValueError:
            pass


def test_precomputed_text_samples():
    """Summary / dense / counterfactual .npy embeddings replace strings + on-the-fly mutation."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_repo(root, with_emb=True, with_text_emb=True, skip_text={"01_2.jpg"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = ShardIndex(str(root), precomputed=True, precomputed_text=True,
                             row_group_cache=2, verbose=False)
        # 01_2 has no summary .npy -> dropped on top of the usual empty-summary / ghost drops
        assert len(idx) == (10 - 1) + (7 - 1) - 1, len(idx)
        assert "01_2.jpg" not in {r.file_name for r in idx.records}

        b = SampleBuilder(idx, SampleConfig(train=False))
        s = b.build(0)
        assert s["summary_emb"].shape == (TEXT_H,) and s["summary_emb"].dtype == np.float32
        assert s["cap_emb"].shape == (TEXT_H,) and s["cap_emb"].dtype == np.float32
        assert s["cf_cap_emb"].shape == (TEXT_H,) and s["cf_cap_emb"].dtype == np.float32
        assert not np.array_equal(s["cf_cap_emb"], s["cap_emb"])  # whole cf paragraph, not the positive
        assert s["dense_emb"].shape == (4, TEXT_H)  # 4 sentences, dense_max_sentences=8
        assert s["hard_neg_emb"].shape == (3, TEXT_H)  # hard_neg_per_image=3 counterfactual rows
        assert len(s["dense"]) == 4  # sentence strings stay row-aligned with dense_emb
        assert s["keywords"] == ["red building", "road", "solar panel"]
        # cf rows are dense rows + 100 in the fixture -> they must not collide with positives
        assert not set(map(bytes, s["dense_emb"])) & set(map(bytes, s["hard_neg_emb"]))

        # fewer positives: dense_max_sentences caps the rows, negatives stay within them
        b2 = SampleBuilder(idx, SampleConfig(train=False, dense_max_sentences=2))
        s2 = b2.build(0)
        assert s2["dense_emb"].shape == (2, TEXT_H) and len(s2["dense"]) == 2
        assert s2["hard_neg_emb"].shape[0] <= 2

        # toggles switch the groups off (summary stays: it feeds the global loss)
        off = SampleBuilder(idx, SampleConfig(train=False, use_dense=False)).build(0)
        assert "dense_emb" not in off and "cap_emb" not in off
        assert off["summary_emb"].shape == (TEXT_H,)
        no_neg = SampleBuilder(idx, SampleConfig(train=False, use_hard_negatives=False)).build(0)
        assert no_neg["dense_emb"].shape == (4, TEXT_H) and len(no_neg["hard_neg_emb"]) == 0

        # validation is deterministic, training re-samples the negative rows
        assert np.array_equal(b.build(3)["dense_emb"], b.build(3)["dense_emb"])
        outs = {bytes(SampleBuilder(idx, SampleConfig(train=True)).build(1)["hard_neg_emb"])
                for _ in range(20)}
        assert len(outs) > 1

        # records whose cf embedding is not precomputed yet: positives but no negatives
        cf_path = root / "caption-00" / "cf_dense_caption" / "00_1.npy"
        cf_path.unlink()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx2 = ShardIndex(str(root), precomputed=True, precomputed_text=True, verbose=False)
        i1 = next(i for i, r in enumerate(idx2.records) if r.file_name == "00_1.jpg")
        s1 = SampleBuilder(idx2, SampleConfig(train=False)).build(i1)
        assert s1["dense_emb"].shape == (4, TEXT_H) and len(s1["hard_neg_emb"]) == 0
        assert "cf_cap_emb" not in s1  # no cf paragraph -> no paragraph-level negative either

        # a cf file with a drifted sentence count is ignored instead of misaligned
        np.save(cf_path, np.zeros((2, TEXT_H), np.float16))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx3 = ShardIndex(str(root), precomputed=True, precomputed_text=True, verbose=False)
        i1 = next(i for i, r in enumerate(idx3.records) if r.file_name == "00_1.jpg")
        assert len(SampleBuilder(idx3, SampleConfig(train=False)).build(i1)["hard_neg_emb"]) == 0

        # a corrupt (non-finite) text embedding skips the record instead of poisoning the run
        # (00_0.jpg is the dropped empty-summary row; 00_3 is a live record)
        np.save(root / "caption-00" / "summary" / "00_3.npy", np.full((1, TEXT_H), np.nan, np.float16))
        i0 = next(i for i, r in enumerate(idx.records) if r.file_name == "00_3.jpg")
        try:
            b.build(i0)
            assert False, "expected SkipSample for a non-finite summary embedding"
        except SkipSample:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
