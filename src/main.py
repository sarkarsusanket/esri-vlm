import csv
from datetime import datetime
from pathlib import Path
from typing import List

import lightning.pytorch as pl
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import Callback
from transformers import CLIPTokenizer, CLIPTextModel

from model import CLIP
from loss import CLIPContrastiveLoss
from data import CLIPDataModule

torch.set_float32_matmul_precision("high")


class CLIPLightningModule(pl.LightningModule):
    def __init__(
        self,
        # model architecture
        embed_dim: int = 512,
        image_resolution: int = 224,
        vision_layers: List[int] = [3, 4, 6, 3],
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
        # tokenizer
        tokenizer_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        vocab_size = self.tokenizer.vocab_size

        self.model = CLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=tuple(vision_layers),
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            vocab_size=vocab_size,
            context_length=context_length,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

        self.loss_fun = CLIPContrastiveLoss()
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

    def common_step(self, batch, batch_idx):
        images = batch["image"]
        captions = batch["caption"]

        text_tokens = self.tokenize(captions)

        logits_per_image, logits_per_text = self.model(images, text_tokens)
        return self.loss_fun(logits_per_image, logits_per_text)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

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


class MetricsCSVCallback(Callback):
    """Writes epoch-level train/val losses to metrics.csv in the log directory."""

    def __init__(self):
        super().__init__()
        self.metrics_path = None
        self._writer = None
        self._file = None

    def _init_writer(self, log_dir):
        self.metrics_path = Path(log_dir) / "metrics.csv"
        self._file = open(self.metrics_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["epoch", "train_loss", "val_loss"])

    def on_fit_start(self, trainer, pl_module):
        log_dir = trainer.log_dir or trainer.default_root_dir
        self._init_writer(log_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        train_val = train_loss.item() if train_loss is not None else ""
        val_val = val_loss.item() if val_loss is not None else ""
        self._writer.writerow([epoch, train_val, val_val])
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
    config_fn = rf"D:\Code\esri-vlm\src\config.yaml"

    if torch.cuda.is_available() and torch.cuda.get_device_name(device=0) == "NVIDIA A100 80GB PCIe":
        torch.set_float32_matmul_precision("highest")
        print("Superfast mode enabled (TF32)")
    else:
        torch.set_float32_matmul_precision("high")

    cli_main(config_fn)
