import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback
from transformers import CLIPTokenizer, CLIPTextModel

from model import CLIP
from loss import NegatedKeywordLoss, build_pair_loss
from data import CLIPDataModule

torch.set_float32_matmul_precision("high")


class CLIPLightningModule(pl.LightningModule):
    def __init__(
        self,
        # model architecture
        embed_dim: int = 512,
        image_resolution: int = 224,
        vision_layers: int = 12,
        vision_width: int = 64,
        vision_patch_size: int = 32,
        in_channels: int = 3,
        context_length: int = 256,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        # training
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        # losses
        loss_type: Literal["clip", "siglip"] = "clip",
        use_masks: bool = False,  # True -> load masks and add the region-level loss
        use_hard_negatives: bool = True,  # mutated dense sentences as extra negatives (data side)
        w_global: float = 1.0,
        w_dense: float = 1.0,
        w_region: float = 1.0,  # only used when use_masks is True
        w_keyword: float = 0.1,
        region_min_coverage: float = 0.1,  # a patch belongs to a region if >= this fraction is masked
        keyword_margin: Optional[float] = None,  # None: plain mean cosine; else mean(relu(cos - margin))
        # tokenizer
        tokenizer_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        vocab_size = self.tokenizer.vocab_size

        assert not use_masks or isinstance(vision_layers, int), "region loss needs the ViT image encoder"
        # SigLIP: scale = log(10), learnable bias = -10 (paper init); CLIP: scale = log(1/0.07), no bias
        logit_kwargs = (dict(logit_scale_init=math.log(10.0), use_logit_bias=True, logit_bias_init=-10.0)
                        if loss_type == "siglip" else {})

        self.model = CLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            vocab_size=vocab_size,
            context_length=context_length,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            **logit_kwargs,
        )
        # print(self.model)

        self.pair_loss = build_pair_loss(loss_type)
        self.keyword_loss = NegatedKeywordLoss(margin=keyword_margin)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def tokenize(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.hparams.context_length,
            return_tensors="pt",
        )
        return tokens.input_ids.to(self.device)

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

    def common_step(self, batch, batch_idx) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.hparams
        images = batch["images"]
        B = images.shape[0]
        use_regions = h.use_masks and h.w_region > 0 and "region_cov" in batch

        if use_regions:
            cls, patches = self.model.encode_image(images, return_patches=True)
        else:
            cls = self.model.encode_image(images)
        img = F.normalize(cls, dim=-1)

        def enc_text(tokens):
            return F.normalize(self.model.encode_text(tokens), dim=-1)

        losses: Dict[str, torch.Tensor] = {}

        # 1) global: summary caption <-> image CLS
        if h.w_global > 0:
            eye = torch.eye(B, dtype=torch.bool, device=img.device)
            losses["global"] = self.pair_loss(self.model.logits(img, enc_text(batch["global_tokens"])), eye)

        # 2) dense: every sentence <-> its image (+ mutated sentences as hard negatives)
        if h.w_dense > 0 and "dense_tokens" in batch:
            logits = self.model.logits(img, enc_text(batch["dense_tokens"]))
            losses["dense"] = self.pair_loss(logits, batch["dense_pos"])

        # 3) region: masked patches <-> keyword text (only with masks)
        if use_regions:
            r_emb, visible = self._region_embeddings(patches, batch["region_cov"])
            valid = visible & (batch["region_txt"] >= 0)
            if valid.any():
                r_txt = enc_text(batch["region_tokens"])
                pos = F.one_hot(batch["region_txt"][valid], r_txt.shape[0]).bool()
                losses["region"] = self.pair_loss(self.model.logits(r_emb[valid], r_txt), pos)

        # 4) keywords: image CLS must NOT match "there is no <keyword that is in the image>"
        if h.w_keyword > 0 and "kw_tokens" in batch:
            losses["keyword"] = self.keyword_loss(img, enc_text(batch["kw_tokens"]),
                                                  batch["kw_img"], batch["kw_txt"])

        weights = {"global": h.w_global, "dense": h.w_dense, "region": h.w_region, "keyword": h.w_keyword}
        total = sum(weights[k] * v for k, v in losses.items())
        return total, losses

    def _shared_step(self, batch, batch_idx, stage: str):
        total, losses = self.common_step(batch, batch_idx)
        bs = batch["images"].shape[0]
        self.log(f"{stage}_loss", total, prog_bar=True, on_epoch=True, batch_size=bs)
        for name, value in losses.items():
            self.log(f"{stage}_loss_{name}", value.detach(), on_epoch=True, batch_size=bs)
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
        parser.link_arguments("model.use_masks", "data.use_masks")
        parser.link_arguments("model.use_hard_negatives", "data.use_hard_negatives")
        # a loss with weight 0 is switched off on the data side too (no tokenizing / negatives)
        parser.link_arguments("model.w_dense", "data.use_dense", compute_fn=lambda w: w > 0)
        parser.link_arguments("model.w_keyword", "data.use_keywords", compute_fn=lambda w: w > 0)


class MetricsCSVCallback(Callback):
    """Writes epoch-level train/val losses to metrics.csv in the log directory."""

    def __init__(self):
        super().__init__()
        self.metrics_path = None
        self._writer = None
        self._file = None

    def _init_writer(self, log_dir, components):
        self._components = components
        self.metrics_path = Path(log_dir) / "metrics.csv"
        self._file = open(self.metrics_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._keys = ["train_loss", "val_loss"] + [f"{st}_loss_{n}" for n in self._components for st in ("train", "val")]
        self._writer.writerow(["epoch"] + self._keys)

    def on_fit_start(self, trainer, pl_module):
        log_dir = trainer.log_dir or trainer.default_root_dir
        hp = pl_module.hparams
        components = [n for n, on in (("global", hp.w_global > 0), ("dense", hp.w_dense > 0),
                                      ("region", hp.use_masks and hp.w_region > 0),
                                      ("keyword", hp.w_keyword > 0)) if on]
        self._init_writer(log_dir, components)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        row = []
        for k in self._keys:
            v = trainer.callback_metrics.get(k)
            row.append(v.item() if v is not None else "")
        self._writer.writerow([epoch] + row)
        self._file.flush()


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
