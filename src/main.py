import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

import csv
import math
import time
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
    RelationalLoss,
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
        precomputed_text: bool = False,   # True -> datamodule feeds precomputed Qwen EOS .npy embs
                                          # (summary / dense / counterfactual); keywords stay tokenised
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
        use_mrl: bool = False,  # Matryoshka: also apply every loss on the nested prefixes in mrl_dims
        mrl_dims: List[int] = [64, 128, 256],  # nested widths (full embed_dim is always added last)
        use_masks: bool = False,  # True -> load masks and add the region-level loss
        use_hard_negatives: bool = True,  # dense hard negatives (precomputed cf sentences when
                                          # precomputed_text, rule-based mutations otherwise; data side)
        use_keywords: bool = True,  # build positive/negative keyword groups (data side)
        w_global: float = 1.0,  # summary caption <-> image CLS
        w_dense: float = 1.0,  # individual dense sentences <-> image (multi-positive contrastive)
        w_dense_cap: float = 1.0,  # whole dense caption <-> image (GCL)
        w_dense_cap_cf: float = 1.0,  # whole dense caption (positive) vs. whole counterfactual
                                      # paragraph (negative) <-> image (needs precomputed cf states)
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
        # relational loss: keep the batch pairwise-similarity structure of the (frozen) backbone
        # space in the projected space, separately for images and captions:
        #   (1/B) sum_i [ cos_pre(i, j) - cos_post(i, j) ]   (see loss.RelationalLoss)
        w_relational: float = 0.1,
        relational_pairing: str = "perm",  # how j is chosen: "perm" | "shift" | "all"
        relational_squared: bool = False,  # squared structure-matching instead of the raw gap
        region_min_coverage: float = 0.1,  # a patch belongs to a region if >= this fraction is masked
        # tokenizer
        tokenizer_name: str = "Qwen/Qwen3-Embedding-0.6B",
        # LRU cache of frozen Qwen pooled states for repeated texts (keyword groups); 0 disables
        text_cache_size: int = 50000,
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
            precomputed_text=precomputed_text,
            pad_token_id=self.tokenizer.pad_token_id,
            text_cache_size=text_cache_size,
            **logit_kwargs,
        )
        # print(self.model)

        self.pair_loss = build_pair_loss(loss_type)  # SigLIP or CLIP multi-positive contrastive
        self.counter_loss = CounterfactualMarginLoss(margin=counter_margin, tau=counter_tau)
        self.threshold_loss = ThresholdLoss(t_pos=threshold_pos, t_neg=threshold_neg)
        self.uniform_loss = UniformityLoss()
        self.silhouette_loss = SilhouetteLoss()
        self.relational_loss = RelationalLoss(pairing=relational_pairing,
                                              squared=relational_squared)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

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

    def text_group(self, batch, emb_key: str, tok_key: str) -> torch.Tensor:
        """L2-normalised text embeddings for one loss group.

        Precomputed hidden states (`emb_key`, from preprocess/precompute_eos.py) are projected
        directly; otherwise the tokenised strings (`tok_key`) run through the Qwen tower.
        """
        if emb_key in batch:
            return F.normalize(self.model.encode_text_emb(batch[emb_key]), dim=-1)
        return self.enc_text(batch[tok_key])

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

    @staticmethod
    def _mrl_dims(h, full_dim: int) -> List[int]:
        """Nested prefix widths for Matryoshka training (full width always included, ascending)."""
        if not getattr(h, "use_mrl", False):
            return [full_dim]
        dims = {int(d) for d in (h.mrl_dims or []) if 0 < int(d) <= full_dim}
        dims.add(full_dim)
        return sorted(dims)

    def calculate_losses(self, batch, img, patches, h, B, use_regions) -> Tuple:
        full_dim = img.shape[-1]
        dims = self._mrl_dims(h, full_dim)
        mrl = bool(getattr(h, "use_mrl", False)) and len(dims) > 1

        eye = torch.eye(B, dtype=torch.bool, device=img.device)
        dev = img.device
        ar = torch.arange(B, device=dev)

        # ---- embeddings, computed once at full width and re-sliced for every MRL prefix ----
        emb: Dict[str, torch.Tensor] = {}
        if h.w_global > 0 and ("global_emb" in batch or "global_tokens" in batch):
            emb["global"] = self.text_group(batch, "global_emb", "global_tokens")
        if (h.w_dense_cap > 0 or h.w_dense_cap_cf > 0) and ("cap_emb" in batch or "cap_tokens" in batch):
            emb["cap"] = self.text_group(batch, "cap_emb", "cap_tokens")
        if h.w_dense_cap_cf > 0 and "cf_cap_emb" in batch and "cap_emb" in batch:
            emb["cf_cap"] = F.normalize(self.model.encode_text_emb(batch["cf_cap_emb"]), dim=-1)
        if h.w_dense > 0 and ("dense_emb" in batch or "dense_tokens" in batch):
            emb["dense"] = self.text_group(batch, "dense_emb", "dense_tokens")
        if h.use_keywords and "kw_tokens" in batch:
            emb["kw"] = self.enc_text(batch["kw_tokens"])
        region_txt = region_valid = region_pos = None
        if use_regions:
            r_emb, visible = self._region_embeddings(patches, batch["region_cov"])
            emb["region"] = r_emb
            region_txt = self.enc_text(batch["region_tokens"])
            # a region is a valid query only if it is visible in the crop and has a text target
            region_valid = visible & (batch["region_txt"] >= 0)
            region_pos = (F.one_hot(batch["region_txt"][region_valid], region_txt.shape[0]).bool()
                          if region_valid.any() else None)

        cf_valid = batch.get("cf_cap_valid")
        if cf_valid is None:
            cf_valid = torch.ones(B, dtype=torch.bool, device=dev)

        # ---- relational loss inputs: the *precomputed* pre-projector features ----------------
        # These are the image CLS / caption EOS states read from the .npy files by the dataset;
        # the backbone is never run again here. The post-projector vectors are reused from above.
        pre_img = pre_txt = rel_key = None
        if h.w_relational > 0:
            if "image_emb" in batch:
                pre_img = self.model.encode_image_pre(batch["image_emb"])  # CLS slice, detached
            # one whole-caption group per image: paragraph if available, else summary
            rel_key = "cap" if "cap_emb" in batch else ("global" if "global_emb" in batch else None)
            if rel_key is not None:
                pre_txt = batch[f"{rel_key}_emb"].detach()

        def sl(x: torch.Tensor, d: int) -> torch.Tensor:
            # Matryoshka prefix: keep the first d dims and re-normalise (identity at full width)
            return x if d >= x.shape[-1] else F.normalize(x[..., :d], dim=-1)

        def loss_at(d: int) -> Dict[str, torch.Tensor]:
            se = {k: sl(v, d) for k, v in emb.items()}
            i = sl(img, d)
            out: Dict[str, torch.Tensor] = {}
            uni: List[torch.Tensor] = []   # text embeddings fed to the uniformity loss
            pg: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (text, pos) fed to the silhouette loss

            # 1) GCL: global contrastive loss (summary caption <-> image CLS)
            if "global" in se:
                out["global"] = self.pair_loss(self.model.logits(i, se["global"]), eye)

            # 2a) whole dense caption <-> image (GCL over the full caption)
            if "cap" in se and h.w_dense_cap > 0:
                out["dense_cap"] = self.pair_loss(self.model.logits(i, se["cap"]), eye)

            # 2c) whole-paragraph contrastive: the dense paragraph is the positive, the whole
            # counterfactual paragraph (cf_dense_caption row 0) and the other in-batch paragraphs
            # are negatives. Rows without a cf state are dropped so a missing negative is never
            # mistaken for a present one.
            if "cap" in se and "cf_cap" in se and h.w_dense_cap_cf > 0 and bool(cf_valid.any()):
                bank = torch.cat([se["cap"], se["cf_cap"][cf_valid]], 0)  # (B + Bv, d); 0..B-1 positive
                pos = torch.zeros(B, bank.shape[0], dtype=torch.bool, device=dev)
                pos[ar, ar] = True
                out["dense_cap_cf"] = self.pair_loss(self.model.logits(i, bank), pos)
                uni.append(bank)

            # 2b) dense sentences: multi-positive contrastive + counterfactual margin + threshold
            if "dense" in se:
                dt = se["dense"]
                out["dense"] = self.pair_loss(self.model.logits(i, dt), batch["dense_pos"])
                uni.append(dt)
                pg.append((dt, batch["dense_pos"]))
                if h.w_counter > 0:
                    out["dense_counter"] = self.counter_loss(i, dt, batch["dense_pos"], batch["dense_neg"])
                if h.w_threshold > 0:
                    out["dense_threshold"] = self.threshold_loss(i, dt, batch["dense_pos"], batch["dense_neg"])

            # 3) keywords: positive combinations vs. absent / swapped-keyword negatives
            if "kw" in se:
                kt = se["kw"]
                uni.append(kt)
                pg.append((kt, batch["kw_pos"]))
                if h.w_counter > 0:
                    out["kw_counter"] = self.counter_loss(i, kt, batch["kw_pos"], batch["kw_neg"])
                if h.w_threshold > 0:
                    out["kw_threshold"] = self.threshold_loss(i, kt, batch["kw_pos"], batch["kw_neg"])

            # 4) region: masked patches <-> keyword text (only with masks)
            if "region" in se and region_pos is not None:
                rt = sl(region_txt, d)
                out["region"] = self.pair_loss(self.model.logits(se["region"][region_valid], rt), region_pos)

            # 5) uniformity (images and texts averaged)
            if h.w_uniform > 0:
                u = self.uniform_loss(i)
                if uni:
                    u = 0.5 * (u + self.uniform_loss(torch.cat(uni, 0)))
                out["uniform"] = u

            # 6) silhouette (image CLS + positive texts, clustered per image)
            if h.w_silhouette > 0:
                z, lab = self._positive_points(i, pg)
                if z is not None:
                    out["silhouette"] = self.silhouette_loss(z, lab)

            # 7) relational: match the pre-/post-projector pairwise structure (image + text)
            if h.w_relational > 0:
                rel = None
                if pre_img is not None:
                    rel = self.relational_loss(pre_img, i)
                if pre_txt is not None and rel_key is not None:
                    post_txt = se.get(rel_key)
                    if post_txt is None:  # caption group not otherwise used at this weight
                        post_txt = sl(F.normalize(self.model.encode_text_emb(pre_txt), dim=-1), d)
                    t = self.relational_loss(pre_txt, post_txt)
                    rel = t if rel is None else rel + t
                if rel is not None:
                    out["relational"] = rel
            return out

        weights = {
            "global": h.w_global, "dense_cap": h.w_dense_cap, "dense_cap_cf": h.w_dense_cap_cf,
            "dense": h.w_dense, "dense_counter": h.w_counter, "dense_threshold": h.w_threshold,
            "kw_counter": h.w_counter, "kw_threshold": h.w_threshold,
            "region": h.w_region, "uniform": h.w_uniform, "silhouette": h.w_silhouette,
            "relational": h.w_relational,
        }

        # final loss = plain sum when MRL is off; when it is on, the same weighted losses are
        # summed across every nested prefix (Matryoshka sum-of-losses). Per-dim losses are logged
        # with a `_d<dim>` suffix so metrics.csv shows every granularity.
        total = torch.zeros((), device=dev)
        flat: Dict[str, torch.Tensor] = {}
        for d in dims:
            for name, value in loss_at(d).items():
                total = total + weights[name] * value
                flat[f"{name}_d{d}" if mrl else name] = value
        return total, flat

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
    

    def _zero_loss(self) -> torch.Tensor:
        """Graph-connected 0 so a skipped batch still runs backward + DDP all-reduce (avoids hangs)."""
        p = next((p for p in self.model.parameters() if p.requires_grad), None)
        return p.sum() * 0.0 if p is not None else torch.zeros((), device=self.device)

    def _shared_step(self, batch, batch_idx, stage: str):
        total, losses = self.common_step(batch, batch_idx)
        if torch.is_tensor(total) and not torch.isfinite(total):
            # a bad sample slipped through the data-side checks: skip the batch instead of
            # backpropagating NaN/Inf and permanently poisoning the weights. Keeping a
            # graph-connected zero (instead of returning None) makes every DDP rank run the same
            # backward pass, so one poisoned rank cannot deadlock the others.
            warnings.warn(f"non-finite {stage} loss at batch {batch_idx}; skipping batch")
            total, losses = self._zero_loss(), {}
        bs = batch["images"].shape[0] if "images" in batch else batch["image_emb"].shape[0]
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
        parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from (e.g. last.ckpt or a specific .ckpt)")
        # Weights & Biases tracking (see `use_wandb` in config.yaml)
        parser.add_argument("--use_wandb", action="store_true",
                            help="Log losses, metrics, GPU utilisation, lr and hparams to W&B")
        parser.add_argument("--wandb_project", type=str, default="esri-vlm")
        parser.add_argument("--wandb_entity", type=str, default=None)
        parser.add_argument("--wandb_tags", type=str, default=None,
                            help="Comma-separated W&B tags")
        parser.add_argument("--wandb_offline", action="store_true",
                            help="Run W&B offline (sync later with `wandb sync`)")
        parser.add_argument("--wandb_every_n_steps", type=int, default=50,
                            help="How often to sample GPU/CPU/RAM metrics")
        # one source of truth: these are set under `model:` and forwarded to the datamodule
        parser.link_arguments("model.image_resolution", "data.image_resolution")
        parser.link_arguments("model.vision_patch_size", "data.patch_size")
        parser.link_arguments("model.context_length", "data.context_length")
        parser.link_arguments("model.tokenizer_name", "data.tokenizer_name")
        parser.link_arguments("model.precomputed_image", "data.precomputed_image")
        parser.link_arguments("model.precomputed_text", "data.precomputed_text")
        parser.link_arguments("model.use_masks", "data.use_masks")
        parser.link_arguments("model.use_hard_negatives", "data.use_hard_negatives")
        # a loss with weight 0 is switched off on the data side too (no tokenizing / negatives)
        parser.link_arguments("model.w_dense", "data.use_dense", compute_fn=lambda w: w > 0)
        parser.link_arguments("model.use_keywords", "data.use_keywords")


class MetricsCSVCallback(Callback):
    """Writes epoch-level train/val losses to metrics.csv in the log directory.

    One row per epoch: `epoch`, the total train/val loss, then every enabled loss component
    (the values Lightning aggregates over the epoch via ``self.log(..., on_epoch=True)``).
    """

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
        self._keys = ["train_loss", "val_loss"] + [
            f"{st}_loss_{n}" for n in self._components for st in ("train", "val")
        ]
        self._writer.writerow(["epoch"] + self._keys)

    def on_fit_start(self, trainer, pl_module):
        log_dir = trainer.log_dir or trainer.default_root_dir
        hp = pl_module.hparams
        components = [
            n for n, on in (
                ("global", hp.w_global > 0),
                ("dense_cap", hp.w_dense_cap > 0),
                ("dense_cap_cf", hp.w_dense_cap_cf > 0 and hp.precomputed_text),
                ("dense", hp.w_dense > 0),
                ("dense_counter", hp.w_dense > 0 and hp.w_counter > 0),
                ("dense_threshold", hp.w_dense > 0 and hp.w_threshold > 0),
                ("kw_counter", hp.use_keywords and hp.w_counter > 0),
                ("kw_threshold", hp.use_keywords and hp.w_threshold > 0),
                ("region", hp.use_masks and hp.w_region > 0),
                ("uniform", hp.w_uniform > 0),
                ("silhouette", hp.w_silhouette > 0),
                ("relational", hp.w_relational > 0),
            ) if on
        ]
        # Matryoshka: one column per (prefix width, component), matching the `_d<dim>` log keys
        dims = CLIPLightningModule._mrl_dims(hp, hp.embed_dim)
        if getattr(hp, "use_mrl", False) and len(dims) > 1:
            components = [f"{n}_d{d}" for d in dims for n in components]
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


class SystemMetricsCallback(Callback):
    """Logs GPU utilisation / memory, host CPU+RAM, throughput and learning rates.

    Everything goes through ``pl_module.log_dict`` so it reaches whatever logger Lightning
    has attached (W&B, TensorBoard, ...). GPU numbers are sampled from NVML through torch,
    i.e. they cover every process on the device, not just this rank.
    """

    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self._batch_start = None
        self._epoch_start = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero or pl_module.logger is None:
            return
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        pl_module.logger.log_metrics(
            {"model/total_params": total, "model/trainable_params": trainable}, step=0
        )

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._batch_start = time.perf_counter()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not trainer.is_global_zero or trainer.global_step % self.every_n_steps != 0:
            return
        total = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total += p.grad.detach().float().pow(2).sum().item()
        pl_module.log("grad_norm", total ** 0.5, on_step=True, on_epoch=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero or trainer.global_step % self.every_n_steps != 0:
            return
        logs: Dict[str, float] = {}
        if self._batch_start is not None:
            dt = time.perf_counter() - self._batch_start
            bs = batch["images"].shape[0] if "images" in batch else batch["image_emb"].shape[0]
            logs["perf/step_time_s"] = dt
            logs["perf/samples_per_s"] = bs / max(dt, 1e-9)
        for i, opt in enumerate(trainer.optimizers):
            for j, pg in enumerate(opt.param_groups):
                key = f"lr/group_{i}_{j}" if len(trainer.optimizers) > 1 else f"lr/group_{j}"
                logs[key] = pg["lr"]
        if torch.cuda.is_available():
            utils = []
            for i in range(torch.cuda.device_count()):
                try:
                    u = float(torch.cuda.utilization(i))
                    logs[f"gpu/{i}_util_percent"] = u
                    utils.append(u)
                except Exception:
                    pass
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    logs[f"gpu/{i}_mem_used_gb"] = (total - free) / 2**30
                    logs[f"gpu/{i}_mem_total_gb"] = total / 2**30
                    logs[f"gpu/{i}_mem_used_percent"] = 100.0 * (total - free) / max(total, 1)
                except Exception:
                    pass
                try:
                    logs[f"gpu/{i}_proc_alloc_gb"] = torch.cuda.memory_allocated(i) / 2**30
                    logs[f"gpu/{i}_proc_reserved_gb"] = torch.cuda.memory_reserved(i) / 2**30
                except Exception:
                    pass
            if utils:
                logs["gpu/util_mean"] = sum(utils) / len(utils)
                logs["gpu/util_max"] = max(utils)
        try:
            import psutil

            logs["system/cpu_percent"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            logs["system/ram_used_gb"] = (vm.total - vm.available) / 2**30
            logs["system/ram_total_gb"] = vm.total / 2**30
        except Exception:
            pass
        if logs:
            pl_module.log_dict(logs, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or self._epoch_start is None:
            return
        pl_module.log("perf/epoch_time_s", time.perf_counter() - self._epoch_start,
                      on_step=False, on_epoch=True)


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

    try:
        config = dict(cli.config)
    except Exception:
        config = {}

    use_wandb = bool(config.get("use_wandb", False))
    if use_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("use_wandb=true but `wandb` is not installed (`pip install wandb`)") from exc
        tags = config.get("wandb_tags")
        tags = [t.strip() for t in tags.split(",") if t.strip()] if isinstance(tags, str) and tags else None
        wandb_logger = WandbLogger(
            project=config.get("wandb_project") or "esri-vlm",
            entity=config.get("wandb_entity") or None,
            name=run_name,
            tags=tags,
            offline=bool(config.get("wandb_offline", False)),
            save_dir=cli.trainer.default_root_dir,
            log_model=False,
        )
        cli.trainer.logger = wandb_logger
        hparams = {f"model/{k}": v for k, v in dict(cli.model.hparams).items()}
        hparams.update({f"data/{k}": v for k, v in dict(cli.datamodule.hparams).items()})
        hparams["trainer/precision"] = str(cli.trainer.precision)
        wandb_logger.log_hyperparams(hparams)

    if cli.trainer.logger is not None:
        try:
            cli.trainer.logger.experiment.name = run_name
        except Exception:
            pass
        if not use_wandb:  # W&B already received the full merged hparams above
            cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    cli.trainer.callbacks.append(MetricsCSVCallback())
    cli.trainer.callbacks.append(
        SystemMetricsCallback(every_n_steps=config.get("wandb_every_n_steps", 50))
    )

    dirname_cfg = Path(default_config_filename).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    ckpt_path = None
    # LightningCLI stores parsed args in config
    if hasattr(cli, "config") and isinstance(cli.config, dict):
        ckpt_path = cli.config.get("ckpt_path")
    # Fallback to parser namespace if present
    if ckpt_path is None and hasattr(cli, "trainer") and hasattr(cli.trainer, "default_root_dir"):
        pass

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    config_fn = rf"/home/susanket/esri-vlm/src/config.yaml"

    if torch.cuda.is_available() and torch.cuda.get_device_name(device=0) == "NVIDIA A100 80GB PCIe":
        torch.set_float32_matmul_precision("highest")
        print("Superfast mode enabled (TF32)")
    else:
        torch.set_float32_matmul_precision("high")

    cli_main(config_fn)
