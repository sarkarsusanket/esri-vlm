"""Loss regression tests. Run: python tests/test_loss.py"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from loss import CrossLingualConsistencyLoss, SilhouetteLoss  # noqa: E402


def test_silhouette_gradient_is_finite():
    """The zero distance on the diagonal must not produce inf/NaN gradients."""
    z = F.normalize(torch.randn(10, 16), dim=-1).requires_grad_()
    labels = torch.arange(5).repeat_interleave(2)
    loss = SilhouetteLoss()(z, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()


def test_silhouette_duplicate_points_gradient_is_finite():
    z = F.normalize(torch.randn(6, 16), dim=-1)
    z = torch.cat([z, z]).requires_grad_()
    labels = torch.arange(6).repeat(2)
    loss = SilhouetteLoss()(z, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()


def test_cross_lingual_consistency():
    loss_fn = CrossLingualConsistencyLoss()
    english = F.normalize(torch.randn(4, 8), dim=-1)

    aligned = loss_fn(english, english)
    opposite = loss_fn(english, -english)
    empty = loss_fn(english, torch.empty(0, 8))

    assert torch.isclose(aligned, torch.zeros(()), atol=1e-6)
    assert torch.isclose(opposite, torch.tensor(2.0))
    assert empty == 0 and torch.isfinite(empty)


def run_all():
    for fn in (test_silhouette_gradient_is_finite, test_silhouette_duplicate_points_gradient_is_finite,
               test_cross_lingual_consistency):
        fn()
        print("PASS", fn.__name__)


if __name__ == "__main__":
    run_all()
