import io
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import lightning.pytorch as pl


class ParquetCLIPDataset(Dataset):
    """Dataset for CLIP training from a parquet file with image_bytes and caption columns."""

    def __init__(
        self,
        parquet_path: str,
        image_transform: Optional[Callable] = None,
    ):
        self.df = pd.read_parquet(parquet_path)
        assert "image_bytes" in self.df.columns, "parquet must have 'image_bytes' column"
        assert "caption" in self.df.columns, "parquet must have 'caption' column"
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]

        image_bytes = row["image_bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)

        caption = str(row["caption"])

        return {"image": image, "caption": caption}


def get_clip_image_transform(image_resolution: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])


class CLIPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        parquet_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        image_resolution: int = 224,
        val_fraction: float = 0.1,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_resolution = image_resolution
        self.val_fraction = val_fraction
        self.save_hyperparameters()

    def setup(self, stage: str = "fit"):
        full_dataset = ParquetCLIPDataset(
            self.parquet_path,
            image_transform=get_clip_image_transform(self.image_resolution, is_train=True),
        )

        n_val = int(len(full_dataset) * self.val_fraction)
        n_train = len(full_dataset) - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )

        self.val_dataset.dataset = ParquetCLIPDataset(
            self.parquet_path,
            image_transform=get_clip_image_transform(self.image_resolution, is_train=False),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
