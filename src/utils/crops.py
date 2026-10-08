"""Torch-free crop sampling + mask -> patch-grid coverage."""
from __future__ import annotations

import math
import random
from typing import Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # left, top, width, height (image pixel coords)


def sample_resized_crop_box(w: int, h: int, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                            rng: random.Random = random) -> Box:
    """Same sampling as torchvision RandomResizedCrop.get_params, but with python rng."""
    area = w * h
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * rng.uniform(scale[0], scale[1])
        ar = math.exp(rng.uniform(*log_ratio))
        cw = int(round(math.sqrt(target * ar)))
        ch = int(round(math.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            return rng.randint(0, w - cw), rng.randint(0, h - ch), cw, ch
    in_ratio = w / h
    if in_ratio < ratio[0]:
        cw, ch = w, int(round(w / ratio[0]))
    elif in_ratio > ratio[1]:
        ch, cw = h, int(round(h * ratio[1]))
    else:
        cw, ch = w, h
    return (w - cw) // 2, (h - ch) // 2, cw, ch


def center_square_box(w: int, h: int) -> Box:
    """Box equivalent to Resize(shorter side) + CenterCrop."""
    s = min(w, h)
    return (w - s) // 2, (h - s) // 2, s, s


def _edges(n: int, g: int) -> np.ndarray:
    # floor(x + .5) is monotone and shift invariant (np.round is banker's rounding).
    return np.floor(np.linspace(0, n, g + 1) + 0.5).astype(np.int64)


def masks_to_grid(masks, sel: Sequence[int], box: Box, image_size: Tuple[int, int],
                  grid: int, flip: bool = False) -> np.ndarray:
    """Fraction of each (grid x grid) cell of the crop covered by each selected mask.

    masks:      (K, Hm, Wm) array / np.memmap, any dtype (binarised with > 0).
    sel:        which of the K masks to use.
    box:        crop box in *image* pixel coordinates; image_size = (width, height).
                The box is rescaled if the mask resolution differs from the image's.
    returns:    float32 (len(sel), grid, grid) in [0, 1]; horizontally flipped if flip.
    """
    iw, ih = image_size
    Hm, Wm = masks.shape[-2:]
    left, top, cw, ch = box
    x0 = min(max(int(round(left * Wm / iw)), 0), Wm - 1)
    y0 = min(max(int(round(top * Hm / ih)), 0), Hm - 1)
    x1 = min(max(int(round((left + cw) * Wm / iw)), x0 + 1), Wm)
    y1 = min(max(int(round((top + ch) * Hm / ih)), y0 + 1), Hm)
    h, w = y1 - y0, x1 - x0

    if len(sel) == 0:
        return np.zeros((0, grid, grid), np.float32)
    m = np.stack([np.asarray(masks[k, y0:y1, x0:x1]) for k in sel]) > 0  # (S, h, w) bool

    if h >= grid and w >= grid:
        ys, xs = _edges(h, grid), _edges(w, grid)
        s = np.add.reduceat(m, ys[:-1], axis=1, dtype=np.int32)
        s = np.add.reduceat(s, xs[:-1], axis=2, dtype=np.int32)
        area = np.outer(np.diff(ys), np.diff(xs)).astype(np.float32)
        cov = s.astype(np.float32) / area
    else:  # crop smaller than the grid: nearest-neighbour sampling
        yi = np.minimum(((np.arange(grid) + 0.5) * h / grid).astype(int), h - 1)
        xi = np.minimum(((np.arange(grid) + 0.5) * w / grid).astype(int), w - 1)
        cov = m[:, yi][:, :, xi].astype(np.float32)
    if flip:
        cov = cov[:, :, ::-1]
    return np.ascontiguousarray(cov)
