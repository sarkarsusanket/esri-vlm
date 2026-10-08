"""Torch-free tests for the shard / mask / sample pipeline. Run: python tests/test_data_io.py"""
import io, os, sys, tempfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shards import SampleBuilder, SampleConfig, ShardIndex  # noqa: E402
from utils.crops import masks_to_grid, center_square_box, sample_resized_crop_box  # noqa: E402


def png(w, h, color):
    b = io.BytesIO()
    Image.new("RGB", (w, h), color).save(b, format="PNG")
    return b.getvalue()


DENSE = ("A red roof house has a swimming pool to the north. Several cars are parked along the road. "
         "The field is green. Dense forest covers the east side.")


def make_repo(root: Path):
    (root / "masks").mkdir(parents=True)
    for sh, n in (("00", 10), ("01", 7)):
        names = [f"{sh}_{i}.jpg" for i in range(n)]
        pd.DataFrame({"image_bytes": [png(300 + 10 * i, 200, (i * 20 % 255, 10, 10)) for i in range(n)],
                      "file_name": names}).to_parquet(root / f"image-{sh}.parquet", row_group_size=4)
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
        assert len(s["neg_keywords"]) == 3 and set(s["neg_keywords"]) <= {"red building", "road", "solar panel"}
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
            and a["neg_keywords"] == c["neg_keywords"]
        # train is stochastic
        outs = {tuple(b.build(1)["neg_keywords"]) + tuple(b.build(1)["hard_negs"]) for _ in range(20)}
        assert len(outs) > 1

        # toggles really switch work off
        off = SampleBuilder(idx, SampleConfig(train=False, use_dense=False, use_keywords=False, use_masks=False)).build(0)
        assert off["dense"] == [] and off["hard_negs"] == [] and off["neg_keywords"] == [] and off["region_kws"] == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
