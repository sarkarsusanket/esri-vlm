"""Load a pretrained ``src/`` CLIP checkpoint for supervised fine-tuning.

The pretraining run may have used *precomputed* towers (``precomputed_image`` /
``precomputed_text``), in which case the DINOv2 backbone is not part of the checkpoint at all.
SFT runs on real pixels + real captions, so we always rebuild the towers with the backbones
loaded (from ``torch.hub`` / HuggingFace, i.e. the same frozen weights pretraining used) and
load everything else from the checkpoint with ``strict=False``.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "3")

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model import CLIP  # noqa: E402


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """``LightningModule`` stores the CLIP under ``self.model`` -> drop that prefix."""
    out = {}
    for k, v in state.items():
        if k.startswith("model."):
            out[k[len("model."):]] = v
    return out


def load_pretrained_clip(
    ckpt_path: str,
    image_resolution: Optional[int] = None,
    context_length: Optional[int] = None,
    freeze_image_tower: bool = True,
    freeze_text_tower: bool = True,
    text_cache_size: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[CLIP, AutoTokenizer, Dict[str, list]]:
    """Rebuild the CLIP towers from a pretraining checkpoint and load its weights.

    Architecture-defining values (``embed_dim``, ``text_model_name``, ``image_model_name``,
    ``text_pooling``, presence of a logit bias) are taken from the checkpoint's saved
    ``hyper_parameters``. ``image_resolution`` / ``context_length`` may be overridden (SFT can run
    at a different size than pretraining); everything else must match the checkpoint.

    Returns ``(model, tokenizer, report)``; ``report`` lists missing / unexpected keys plus the
    ``resolved`` architecture actually used.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))
    state = _strip_module_prefix(ckpt["state_dict"])

    text_name = hp.get("text_model_name", "Qwen/Qwen3-Embedding-0.6B")
    ckpt_res = int(hp.get("image_resolution", 224))
    ckpt_ctx = int(hp.get("context_length", 256))
    # SFT may legitimately run at a different resolution / context than pretraining.
    res = int(image_resolution) if image_resolution else ckpt_res
    ctx = int(context_length) if context_length else ckpt_ctx
    image_name = hp.get("image_model_name", "dinov2_vits14")
    use_logit_bias = any(k.endswith("logit_bias") for k in state)

    tokenizer = AutoTokenizer.from_pretrained(text_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CLIP(
        embed_dim=int(hp.get("embed_dim", 512)),
        image_resolution=res,
        text_model_name=text_name,
        image_model_name=image_name,
        text_pooling=hp.get("text_pooling", "last_eos"),
        context_length=ctx,
        freeze_text_tower=freeze_text_tower,
        freeze_image_tower=freeze_image_tower,
        precomputed_image=False,   # SFT always encodes real pixels
        precomputed_text=False,    # ... and real captions
        pad_token_id=tokenizer.pad_token_id,
        text_cache_size=text_cache_size,
        use_logit_bias=use_logit_bias,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    # The DINOv2 backbone is expected to be "missing" when pretraining only saved projections;
    # it is deliberately re-loaded from the hub (frozen) instead.
    backbone_missing = [k for k in missing if ".backbone." in k]
    other_missing = [k for k in missing if ".backbone." not in k]
    report = {"missing": other_missing, "missing_backbone": backbone_missing,
              "unexpected": list(unexpected),
              "resolved": {"text_model_name": text_name, "image_resolution": res,
                           "context_length": ctx}}

    if verbose:
        print(f"[sft.loader] loaded {ckpt_path}")
        print(f"  embed_dim={model.visual.proj.out_features} text={text_name} image={image_name} "
              f"logit_bias={use_logit_bias}")
        print(f"  matched keys={len(state) - len(unexpected)}  "
              f"missing(non-backbone)={len(other_missing)}  "
              f"missing(backbone, reloaded from hub)={len(backbone_missing)}  "
              f"unexpected={len(unexpected)}")
        if other_missing:
            warnings.warn(f"{len(other_missing)} checkpoint keys were missing, e.g. {other_missing[:5]}")
        if unexpected:
            warnings.warn(f"{len(unexpected)} unexpected keys in the checkpoint, e.g. {unexpected[:5]}")

    model.to(device)
    return model, tokenizer, report
