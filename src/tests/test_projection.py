"""Projection-head tests. Run: python tests/test_projection.py"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from model import (DINOv2ImageTower, PROJECTION_TYPES, build_projection)  # noqa: E402


def _build_kwargs(projection_type: str):
    return dict(
        projection_type=projection_type,
        dropout=0.0,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
    )


def _tower_kwargs(projection_type: str):
    return dict(
        projection_type=projection_type,
        projection_dropout=0.0,
        projection_num_layers=2,
        projection_num_heads=4,
        projection_ff_dim=32,
    )


def test_projection_heads_preserve_shape_and_gradients():
    x = torch.randn(2, 3, 32, requires_grad=True)
    for projection_type in PROJECTION_TYPES:
        proj = build_projection(32, 16, **_build_kwargs(projection_type))
        y = proj(x)
        assert y.shape == (2, 3, 16), (projection_type, y.shape)
        assert torch.isfinite(y).all()
        loss = y.sum()
        loss.backward(retain_graph=True)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_image_tower_accepts_every_projection_type():
    features = torch.randn(2, 5, 384)  # DINOv2 ViT-S/14 CLS + 4 patches
    for projection_type in PROJECTION_TYPES:
        tower = DINOv2ImageTower(
            model_name="dinov2_vits14",
            embed_dim=16,
            freeze=True,
            precomputed=True,
            **_tower_kwargs(projection_type),
        )
        cls = tower(features)
        cls2, patches = tower(features, return_patches=True)
        assert cls.shape == (2, 16), (projection_type, cls.shape)
        assert torch.equal(cls, cls2)
        assert patches.shape == (2, 4, 16), (projection_type, patches.shape)


def test_linear_probe_keeps_checkpoint_key_layout():
    proj = build_projection(32, 16, projection_type="linear_probe")
    assert isinstance(proj, nn.Linear)
    assert set(proj.state_dict()) == {"weight"}


def test_unknown_projection_type_is_rejected():
    try:
        build_projection(32, 16, projection_type="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown projection_type")


def test_transformer_heads_must_divide_embed_dim():
    try:
        build_projection(32, 16, projection_type="transformer", num_heads=5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid num_heads")


def run_all():
    for fn in (test_projection_heads_preserve_shape_and_gradients,
               test_image_tower_accepts_every_projection_type,
               test_linear_probe_keeps_checkpoint_key_layout,
               test_unknown_projection_type_is_rejected,
               test_transformer_heads_must_divide_embed_dim):
        fn()
        print("PASS", fn.__name__)


if __name__ == "__main__":
    run_all()
