"""Supervised fine-tuning of a pretrained ``src/`` CLIP on (image, caption) parquet data.

    cd /home/susanket/esri-vlm
    python sft/train.py --model.ckpt_path=/path/to/epoch=XX.ckpt --data.data_dir=/path/to/parquets

The loss is a plain or multi-GPU-gathered image<->text contrastive loss (``clip`` softmax or
``siglip`` sigmoid); negatives come from the batch (and from the other ranks when
``model.gather_across_gpus=true``). Everything else (towers, tokenizer, resolution) is taken from
the checkpoint.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "3")

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
import lightning.pytorch as pl  # noqa: E402
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint  # noqa: E402
from lightning.pytorch.cli import LightningCLI  # noqa: E402

from loss import build_pair_loss  # noqa: E402
from sft.data import SFTDataModule  # noqa: E402
from sft.loader import load_pretrained_clip  # noqa: E402


# ---------------------------------------------------------------------------------- utils
def _all_gather(x: torch.Tensor) -> torch.Tensor:
    """Differentiable all-gather of ``x`` along dim 0 (no-op outside distributed runs)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return x
    if torch.distributed.get_world_size() == 1:
        return x
    from torch.distributed.nn.functional import all_gather  # autograd-aware
    return torch.cat(all_gather(x.contiguous()), dim=0)


def _block_diag_pos(size: int, block: int, device) -> torch.Tensor:
    pos = torch.zeros(size, size, dtype=torch.bool, device=device)
    for k in range(0, size, block):
        pos[k:k + block, k:k + block] = torch.eye(block, dtype=torch.bool, device=device)
    return pos


def _retrieval_acc(logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    i2t = pos.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean()
    t2i = pos.gather(0, logits.argmax(dim=0, keepdim=True).t()).float().mean()
    return 0.5 * (i2t + t2i)


# ---------------------------------------------------------------------------------- module
class SFTModel(pl.LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        image_resolution: Optional[int] = None,
        context_length: Optional[int] = None,
        loss_type: str = "siglip",
        freeze_image_tower: bool = True,
        freeze_text_tower: bool = True,
        text_cache_size: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        min_lr_ratio: float = 0.0,
        gather_across_gpus: bool = False,
        w_relational: float = 0.1,
        precomputed_features: Optional[str] = None,
    ):
        super().__init__()
        # image_resolution / context_length are resolved from the checkpoint and owned by the
        # datamodule's hparams; keep them out of the model's to avoid Lightning's hparam-merge clash.
        self.save_hyperparameters(ignore=["image_resolution", "context_length"])

        self.precomputed = bool(precomputed_features)
        if self.precomputed:
            self.model = None
            self.tokenizer = None
            self._init_projection_only(ckpt_path, precomputed_features)
        else:
            self.model, self.tokenizer, self.load_report = load_pretrained_clip(
                ckpt_path=ckpt_path,
                image_resolution=image_resolution,
                context_length=context_length,
                freeze_image_tower=freeze_image_tower,
                freeze_text_tower=freeze_text_tower,
                text_cache_size=text_cache_size,
            )
            self.resolved = self.load_report["resolved"]
        self._pair_loss = build_pair_loss(loss_type)

    def _init_projection_only(self, ckpt_path: str, features_dir: str) -> None:
        """Load only the two linear probes (+ logit scale/bias) and cache the features on-device.

        The frozen towers are not constructed at all, so startup is a checkpoint read instead of
        a DINOv2 hub download + Qwen3 0.6B ``from_pretrained``.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = dict(ckpt.get("hyper_parameters", {}))
        state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                 if k.startswith("model.")}

        img_w = state["visual.proj.weight"].float()
        txt_w = state["transformer.proj.weight"].float()
        self.image_proj = nn.Linear(img_w.shape[1], img_w.shape[0], bias=False)
        self.text_proj = nn.Linear(txt_w.shape[1], txt_w.shape[0], bias=False)
        with torch.no_grad():
            self.image_proj.weight.copy_(img_w)
            self.text_proj.weight.copy_(txt_w)

        self.logit_scale = nn.Parameter(state["logit_scale"].float().reshape(()).clone())
        if state.get("logit_bias") is not None:
            self.logit_bias = nn.Parameter(state["logit_bias"].float().reshape(()).clone())
        else:
            self.register_parameter("logit_bias", None)

        root = Path(features_dir)
        img = np.load(root / "image.npy", mmap_mode="r")
        txt = np.load(root / "text.npy", mmap_mode="r")
        if img.shape[1] != img_w.shape[1] or txt.shape[1] != txt_w.shape[1]:
            raise ValueError(
                f"feature dims image={img.shape} text={txt.shape} do not match checkpoint "
                f"projections ({tuple(img_w.shape)}, {tuple(txt_w.shape)})")
        if img.shape[0] != txt.shape[0]:
            raise ValueError(f"image/text feature rows differ: {img.shape[0]} vs {txt.shape[0]}")
        # Persistent=False: these buffers are data, never part of the checkpoint.
        self.register_buffer("_image_feats", torch.from_numpy(np.array(img)), persistent=False)
        self.register_buffer("_text_feats", torch.from_numpy(np.array(txt)), persistent=False)
        self.resolved = {
            "image_resolution": int(hp.get("image_resolution", 224)),
            "context_length": int(hp.get("context_length", 256)),
        }
        print(f"[sft] precomputed mode: {img.shape[0]} cached pairs, "
              f"image_dim={img.shape[1]} text_dim={txt.shape[1]}, backbones not loaded")

    # ------------------------------------------------------------------ forward pieces
    def _logits(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        if self.precomputed:
            logits = self.logit_scale.exp() * img @ txt.t()
            if self.logit_bias is not None:
                logits = logits + self.logit_bias
            return logits
        return self.model.logits(img, txt)

    def _encode(self, batch):
        if self.precomputed:
            idx = batch["idx"].to(self.device, non_blocking=True)
            img = self.image_proj(self._image_feats[idx].float())
            txt = self.text_proj(self._text_feats[idx].float())
        else:
            img = self.model.encode_image(batch["images"])
            txt = self.model.encode_text(batch["tokens"])
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)
        return img, txt

    @staticmethod
    def relational_loss(pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Squared difference between pre-/post-projector batch cosine-similarity matrices.

        ``mean_{i,j} (cos_pre(i,j) - cos_post(i,j))^2``; ``pre`` is detached (frozen backbone).
        """
        pre = F.normalize(pre.detach().to(post.dtype), dim=-1)
        post = F.normalize(post, dim=-1)
        return F.mse_loss(post @ post.t(), pre @ pre.t())

    def _shared_step(self, batch, stage: str):
        img, txt = self._encode(batch)
        b = img.shape[0]
        world = self.trainer.world_size if self.trainer is not None else 1

        if self.hparams.gather_across_gpus and world > 1:
            img_all, txt_all = _all_gather(img), _all_gather(txt)
            pos = _block_diag_pos(img_all.shape[0], b, img.device)
            logits = self._logits(img_all, txt_all)
        else:
            pos = torch.eye(b, dtype=torch.bool, device=img.device)
            logits = self._logits(img, txt)

        loss = self._pair_loss(logits, pos)
        if not torch.isfinite(loss):  # never let a bad batch kill the DDP run
            loss = 0.0 * (self.logit_scale.exp().sum() + img.sum() * 0.0)
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
            return loss

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=(stage == "train"),
                 prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", _retrieval_acc(logits, pos), on_epoch=True, on_step=False,
                 sync_dist=True)
        self.log("logit_scale", self.logit_scale.exp(), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    # ------------------------------------------------------------------ optim / schedule
    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [{"params": decay, "weight_decay": self.hparams.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=self.hparams.learning_rate)

        total = self.trainer.estimated_stepping_batches
        warmup = max(1, int(self.hparams.warmup_ratio * total))

        def lr_lambda(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return max(self.hparams.min_lr_ratio, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793))).item())

        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
                                 "interval": "step"}}


# ---------------------------------------------------------------------------------- callback
class SystemMetricsCallback(Callback):
    """Per-step GPU/CPU/perf metrics, logged through Lightning (and hence to W&B)."""

    def __init__(self, every_n_steps: int = 50):
        self.every_n_steps = max(1, int(every_n_steps))
        self._t0 = None
        self._step_t0 = None
        self._samples = 0

    def on_train_start(self, trainer, pl_module):
        self._t0 = self._step_t0 = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._samples += batch["images"].shape[0]
        if (batch_idx + 1) % self.every_n_steps:
            return
        now = time.time()
        dt = now - self._step_t0
        self._step_t0 = now
        m = {"perf/step_time_s": dt / self.every_n_steps,
             "perf/samples_per_s": self._samples / max(dt, 1e-9)}
        self._samples = 0
        try:
            import psutil
            m["system/cpu_percent"] = psutil.cpu_percent()
            m["system/ram_used_gb"] = psutil.virtual_memory().used / 1e9
        except Exception:
            pass
        try:
            for i in range(torch.cuda.device_count()):
                util = torch.cuda.utilization(i) if hasattr(torch.cuda, "utilization") else None
                free, total = torch.cuda.mem_get_info(i)
                if util is not None:
                    m[f"gpu/util_{i}"] = util
                m[f"gpu/mem_used_gb_{i}"] = (total - free) / 1e9
        except Exception:
            pass
        for k, v in m.items():
            pl_module.log(k, float(v), on_step=True, on_epoch=False, prog_bar=False)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._t0 is not None:
            pl_module.log("perf/epoch_time_s", time.time() - self._t0, on_epoch=True)
            self._t0 = time.time()


# ---------------------------------------------------------------------------------- CLI
class SFTCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--use_wandb", action="store_true",
                            help="Log to Weights & Biases (also settable via use_wandb in config.yaml)")
        parser.add_argument("--wandb_project", type=str, default="esri-vlm-sft")
        parser.add_argument("--wandb_entity", type=str, default=None)
        parser.add_argument("--wandb_tags", type=str, default=None,
                            help="Comma-separated list of tags")
        parser.add_argument("--wandb_offline", action="store_true")
        parser.add_argument("--wandb_every_n_steps", type=int, default=50)


def cli_main(default_config: str) -> None:
    cli = SFTCLI(
        model_class=SFTModel,
        datamodule_class=SFTDataModule,
        save_config_kwargs=dict(
            config_filename=default_config.replace(".yaml", "-latest.yaml"), overwrite=True),
        trainer_defaults={"log_every_n_steps": 10},
        parser_kwargs={"default_config_files": [default_config]},
        run=False,
    )
    config = dict(cli.config)

    # The checkpoint defines the towers, tokenizer, resolution and context length. Make the
    # datamodule follow it so captions and images can never be processed differently from the
    # tower they are fed to (override with --model.image_resolution / --model.context_length).
    data_hp, resolved = cli.datamodule.hparams, cli.model.resolved
    data_hp.image_resolution = resolved["image_resolution"]
    data_hp.context_length = resolved["context_length"]
    if cli.model.tokenizer is not None:
        data_hp.tokenizer_name = cli.model.tokenizer.name_or_path
    data_hp.precomputed_dir = cli.model.hparams.precomputed_features
    print(f"[sft] resolution={resolved['image_resolution']} context={resolved['context_length']} "
          f"tokenizer={data_hp.tokenizer_name} "
          f"precomputed={data_hp.precomputed_dir or 'off'}")

    if bool(config.get("use_wandb", False)):
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError as exc:  # pragma: no cover
            raise ImportError("use_wandb=true but `wandb` is not installed") from exc
        tags = config.get("wandb_tags")
        logger = WandbLogger(
            project=config.get("wandb_project") or "esri-vlm-sft",
            entity=config.get("wandb_entity") or None,
            name=f"SFT_{datetime.now():%Y%m%d-%H%M%S}",
            save_dir=cli.trainer.default_root_dir,
            offline=bool(config.get("wandb_offline", False)),
            log_model=False,
        )
        cli.trainer.logger = logger
        hparams = {f"model/{k}": v for k, v in dict(cli.model.hparams).items()}
        hparams.update({f"data/{k}": v for k, v in dict(cli.datamodule.hparams).items()})
        logger.log_hyperparams(hparams)

    ckpt_dir = Path(cli.trainer.default_root_dir) / "checkpoints"
    cli.trainer.callbacks.extend([
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min", save_top_k=1,
                        save_last=True, filename="epoch{epoch:02d}-valloss{val_loss:.4f}"),
        SystemMetricsCallback(every_n_steps=config.get("wandb_every_n_steps", 50)),
    ])
    print(f"[sft] checkpoints -> {ckpt_dir}")
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_main(str(Path(__file__).resolve().parent / "config.yaml"))
