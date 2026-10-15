import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import csv
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback
from transformers import AutoTokenizer

from model import CLIP
from loss import (
    build_pair_loss,
    CounterfactualMarginLoss,
    ThresholdLoss,
    UniformityLoss,
    SilhouetteLoss,
)
from data import CLIPDataModule

torch.set_float32_matmul_precision("high")


class CLIPLightningModule(pl.LightningModule):
    def __init__(
        self,
        # pretrained towers
        text_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        image_model_name: str = "dinov2_vits14",
        text_pooling: str = "last_eos",
        freeze_text_tower: bool = True,
        freeze_image_tower: bool = True,
        precomputed_image: bool = False,  # True -> datamodule feeds precomputed DINOv2 .npy embs
        # model architecture
        embed_dim: int = 512,
        image_resolution: int = 224,
        vision_patch_size: int = 14,
        context_length: int = 256,
        # training
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        # losses
        loss_type: Literal["clip", "siglip"] = "clip",
        use_masks: bool = False,  # True -> load masks and add the region-level loss
        use_hard_negatives: bool = True,  # mutated dense sentences as extra negatives (data side)
        use_keywords: bool = True,  # build positive/negative keyword groups (data side)
        w_global: float = 1.0,  # summary caption <-> image CLS
        w_dense: float = 1.0,  # individual dense sentences <-> image (multi-positive contrastive)
        w_dense_cap: float = 1.0,  # whole dense caption <-> image (GCL)
        w_region: float = 1.0,  # only used when use_masks is True
        # counterfactual margin loss (applied to the dense sentences and the keywords)
        w_counter: float = 1.0,
        counter_margin: float = 0.2,  # m in the margin term m * (1 - s(a+, a-))
        counter_tau: float = 0.1,  # temperature dividing the softplus argument
        # threshold (hinge-squared) loss on positive / negative similarities
        w_threshold: float = 1.0,
        threshold_pos: float = 0.5,  # t+ : positives are pulled above this
        threshold_neg: float = 0.1,  # t- : negatives are pushed below this
        # geometry regularisers
        w_uniform: float = 0.1,
        w_silhouette: float = 0.1,
        region_min_coverage: float = 0.1,  # a patch belongs to a region if >= this fraction is masked
        # tokenizer
        tokenizer_name: str = "Qwen/Qwen3-Embedding-0.6B",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # SigLIP: scale = log(10), learnable bias = -10 (paper init); CLIP: scale = log(1/0.07), no bias
        logit_kwargs = (dict(logit_scale_init=math.log(10.0), use_logit_bias=True, logit_bias_init=-10.0)
                        if loss_type == "siglip" else {})

        self.model = CLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            text_pooling=text_pooling,
            context_length=context_length,
            freeze_text_tower=freeze_text_tower,
            freeze_image_tower=freeze_image_tower,
            precomputed_image=precomputed_image,
            pad_token_id=self.tokenizer.pad_token_id,
            **logit_kwargs,
        )
        # print(self.model)

        self.pair_loss = build_pair_loss(loss_type)  # SigLIP or CLIP multi-positive contrastive
        self.counter_loss = CounterfactualMarginLoss(margin=counter_margin, tau=counter_tau)
        self.threshold_loss = ThresholdLoss(t_pos=threshold_pos, t_neg=threshold_neg)
        self.uniform_loss = UniformityLoss()
        self.silhouette_loss = SilhouetteLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self._last_step = None  # (stage, batch_idx, {component: value}, total) for the CSV callback

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.hparams.context_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items() if k in ("input_ids", "attention_mask")}

    def _region_embeddings(self, patches: torch.Tensor, cov: torch.Tensor):
        """Pool patch tokens inside every mask.

        patches (B, P, D), cov (B, K, g, g) with P == g*g. Returns L2-normalised region
        embeddings (B, K, D) and a (B, K) bool telling which regions are still visible in the crop.
        Weights are the masked fraction of every patch, ignoring patches below
        `region_min_coverage`; if that leaves nothing (small object) the single most covered patch is used.
        """
        c = cov.flatten(2).to(patches.dtype)  # (B, K, P)
        visible = c.amax(-1) > 0
        w = c * (c >= self.hparams.region_min_coverage)
        tiny = visible & (w.sum(-1) == 0)
        if tiny.any():
            top = F.one_hot(c.argmax(-1), c.shape[-1]).to(c.dtype)
            w = torch.where(tiny.unsqueeze(-1), top, w)
        p = F.normalize(patches, dim=-1)  # unit patches: no single high-norm token can dominate
        emb = torch.bmm(w, p) / w.sum(-1, keepdim=True).clamp_min(1e-6)
        return F.normalize(emb, dim=-1), visible

    def enc_text(self, tokens):
            return F.normalize(self.model.encode_text(tokens), dim=-1)

    @staticmethod
    def _positive_points(img: torch.Tensor, groups):
        """(z, labels) over the image CLS tokens plus every positive text embedding.

        Each positive text column is assigned to the first image it is a positive for, so all
        points of one image form a cluster (used by the silhouette loss). Returns (None, None)
        when there is nothing to cluster.
        """
        zs = [img]
        labels = [torch.arange(img.shape[0], device=img.device)]
        for text_emb, pos in groups:
            if text_emb.shape[0] == 0 or not bool(pos.any()):
                continue
            has = pos.any(0)
            owner = pos.to(torch.long).argmax(0)
            zs.append(text_emb[has])
            labels.append(owner[has])
        z, lab = torch.cat(zs, 0), torch.cat(labels, 0)
        if z.shape[0] < 3 or int(lab.max()) < 1:
            return None, None
        return z, lab

    def calculate_losses(self, batch, img, patches, h, B, use_regions) -> Tuple:
        losses: Dict[str, torch.Tensor] = {}
        eye = torch.eye(B, dtype=torch.bool, device=img.device)
        uniform_texts: List[torch.Tensor] = []
        pos_groups = []  # (text_emb, pos_mask) fed to the silhouette loss

        # 1) GCL: global contrastive loss (summary caption <-> image CLS)
        if h.w_global > 0 and "global_tokens" in batch:
            losses["global"] = self.pair_loss(self.model.logits(img, self.enc_text(batch["global_tokens"])), eye)

        # 2a) whole dense caption <-> image (GCL over the full caption)
        if h.w_dense_cap > 0 and "cap_tokens" in batch:
            losses["dense_cap"] = self.pair_loss(self.model.logits(img, self.enc_text(batch["cap_tokens"])), eye)

        # 2b) dense sentences: multi-positive contrastive + counterfactual margin + threshold
        if h.w_dense > 0 and "dense_tokens" in batch:
            dt = self.enc_text(batch["dense_tokens"])
            losses["dense"] = self.pair_loss(self.model.logits(img, dt), batch["dense_pos"])
            uniform_texts.append(dt)
            pos_groups.append((dt, batch["dense_pos"]))
            if h.w_counter > 0:
                losses["dense_counter"] = self.counter_loss(img, dt, batch["dense_pos"], batch["dense_neg"])
            if h.w_threshold > 0:
                losses["dense_threshold"] = self.threshold_loss(img, dt, batch["dense_pos"], batch["dense_neg"])

        # 3) keywords: positive combinations vs. absent / swapped-keyword negatives
        if h.use_keywords and "kw_tokens" in batch:
            kt = self.enc_text(batch["kw_tokens"])
            uniform_texts.append(kt)
            pos_groups.append((kt, batch["kw_pos"]))
            if h.w_counter > 0:
                losses["kw_counter"] = self.counter_loss(img, kt, batch["kw_pos"], batch["kw_neg"])
            if h.w_threshold > 0:
                losses["kw_threshold"] = self.threshold_loss(img, kt, batch["kw_pos"], batch["kw_neg"])

        # 4) region: masked patches <-> keyword text (only with masks)
        if use_regions:
            r_emb, visible = self._region_embeddings(patches, batch["region_cov"])
            valid = visible & (batch["region_txt"] >= 0)
            if valid.any():
                r_txt = self.enc_text(batch["region_tokens"])
                pos = F.one_hot(batch["region_txt"][valid], r_txt.shape[0]).bool()
                losses["region"] = self.pair_loss(self.model.logits(r_emb[valid], r_txt), pos)

        # 5) uniformity (images and texts averaged)
        if h.w_uniform > 0:
            u = self.uniform_loss(img)
            if uniform_texts:
                u = 0.5 * (u + self.uniform_loss(torch.cat(uniform_texts, 0)))
            losses["uniform"] = u

        # 6) silhouette (image CLS + positive texts, clustered per image)
        if h.w_silhouette > 0:
            z, lab = self._positive_points(img, pos_groups)
            if z is not None:
                losses["silhouette"] = self.silhouette_loss(z, lab)

        weights = {
            "global": h.w_global, "dense_cap": h.w_dense_cap, "dense": h.w_dense,
            "dense_counter": h.w_counter, "dense_threshold": h.w_threshold,
            "kw_counter": h.w_counter, "kw_threshold": h.w_threshold,
            "region": h.w_region, "uniform": h.w_uniform, "silhouette": h.w_silhouette,
        }
        total = sum(weights[k] * v for k, v in losses.items())
        return total, losses

    def common_step(self, batch, batch_idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.hparams
        images = batch["images"] if "images" in batch else batch["image_emb"]
        B = images.shape[0]
        use_regions = h.use_masks and h.w_region > 0 and "region_cov" in batch

        patches = None
        if use_regions:
            cls, patches = self.model.encode_image(images, return_patches=True)
        else:
            cls = self.model.encode_image(images)
        img = F.normalize(cls, dim=-1)

        return self.calculate_losses(batch, img, patches, h, B, use_regions)
    

    def _shared_step(self, batch, batch_idx, stage: str):
        total, losses = self.common_step(batch, batch_idx)
        if torch.is_tensor(total) and not torch.isfinite(total):
            # a bad sample slipped through the data-side checks: skip the batch instead of
            # backpropagating NaN/Inf and permanently poisoning the weights
            warnings.warn(f"non-finite {stage} loss at batch {batch_idx}; skipping batch")
            self._last_step = None
            return None
        bs = batch["images"].shape[0] if "images" in batch else batch["image_emb"].shape[0]
        on_step = stage == "train"
        self.log(f"{stage}_loss", total, prog_bar=True, on_epoch=True, on_step=on_step, batch_size=bs)
        for name, value in losses.items():
            self.log(f"{stage}_loss_{name}", value.detach(), on_epoch=True, on_step=on_step, batch_size=bs)
        # keep the raw per-iteration values for MetricsCSVCallback
        components = {name: float(value.detach()) for name, value in losses.items()}
        components["total"] = float(total.detach())
        self._last_step = (stage, batch_idx, components)
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": self.weight_decay},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return max(0.0, 1.0 - (step - self.warmup_steps) / 100000)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")
        # one source of truth: these are set under `model:` and forwarded to the datamodule
        parser.link_arguments("model.image_resolution", "data.image_resolution")
        parser.link_arguments("model.vision_patch_size", "data.patch_size")
        parser.link_arguments("model.context_length", "data.context_length")
        parser.link_arguments("model.tokenizer_name", "data.tokenizer_name")
        parser.link_arguments("model.precomputed_image", "data.precomputed_image")
        parser.link_arguments("model.use_masks", "data.use_masks")
        parser.link_arguments("model.use_hard_negatives", "data.use_hard_negatives")
        # a loss with weight 0 is switched off on the data side too (no tokenizing / negatives)
        parser.link_arguments("model.w_dense", "data.use_dense", compute_fn=lambda w: w > 0)
        parser.link_arguments("model.use_keywords", "data.use_keywords")


class MetricsCSVCallback(Callback):
    """Writes ITERATION-wise train/val losses (one row per batch) to metrics.csv.

    Columns: epoch, step, then one column per loss component (sorted) plus `total`. Rows are
    flushed as they are written, so the file can be tailed during training.
    """

    def __init__(self):
        super().__init__()
        self.metrics_path = None
        self._file = None
        self._writer = None
        self._cols = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        log_dir = trainer.log_dir or trainer.default_root_dir
        self.metrics_path = Path(log_dir) / "metrics.csv"
        self._file = open(self.metrics_path, "w", newline="")
        self._writer = csv.writer(self._file)

    def _write(self, trainer, pl_module, stage):
        if self._writer is None or not trainer.is_global_zero:
            return
        rec = getattr(pl_module, "_last_step", None)
        if rec is None:
            return
        st, _, comps = rec
        if st != stage:
            return
        if self._cols is None:
            self._cols = ["epoch", "step"] + sorted(comps)
            self._writer.writerow(self._cols)
        self._writer.writerow([trainer.current_epoch, trainer.global_step]
                              + [comps.get(c, "") for c in self._cols[2:]])
        self._file.flush()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._write(trainer, pl_module, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._write(trainer, pl_module, "val")


def cli_main(default_config_filename="./configs/default.yaml"):
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")

    cli = MyLightningCLI(
        model_class=CLIPLightningModule,
        datamodule_class=CLIPDataModule,
        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 16,
            "log_every_n_steps": 10,
        },
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"CLIP_{ts}"

    csv_callback = MetricsCSVCallback()

    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    cli.trainer.callbacks.append(csv_callback)

    dirname_cfg = Path(default_config_filename).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    config_fn = rf"/home/susanket/esri-vlm/src/config.yaml"

    if torch.cuda.is_available() and torch.cuda.get_device_name(device=0) == "NVIDIA A100 80GB PCIe":
        torch.set_float32_matmul_precision("highest")
        print("Superfast mode enabled (TF32)")
    else:
        torch.set_float32_matmul_precision("high")

    cli_main(config_fn)
