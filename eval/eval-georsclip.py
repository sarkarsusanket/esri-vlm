"""
GeoRSCLIP Cross-Modal Evaluation Script (Class-based Folders + Parquet Caption Datasets)

Usage:
    python eval_georsclip.py --ckpt_path path/to/checkpoint.pt --model_arch ViT-B-32
"""

import argparse
import io
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import open_clip

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENCHMARK_ROOT = rf"/data/susanket/vlm/eval"


# ==========================================
# Checkpoint & Model Loading
# ==========================================

def load_georsclip_model(ckpt_path: str, model_arch: str, device: torch.device):
    """Loads a GeoRSCLIP model and its corresponding OpenCLIP tokenizer and image transform."""
    print(f"Initializing OpenCLIP architecture: {model_arch}")
    model, _, image_transform = open_clip.create_model_and_transforms(model_arch, pretrained=None)
    
    print(f"Loading checkpoint weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract state dict from wrapper keys if necessary
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        
    # Clean up any potential DDP prefix ('module.')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    tokenizer = open_clip.get_tokenizer(model_arch)
    return model, tokenizer, image_transform


# ==========================================
# Datasets
# ==========================================

class ImageFolderFlat(Dataset):
    """Loads images from class subfolders in image-class/ directory using provided transform."""

    def __init__(self, root: str, transform, image_size: int = 224):
        self.root = Path(root)
        self.transform = transform
        self.image_size = image_size

        self.samples: List[Tuple[Path, str]] = []
        self.class_names: List[str] = sorted([
            d.name for d in self.root.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
                    self.samples.append((img_path, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_name = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))
        image = self.transform(image)
        return image, self.class_to_idx[cls_name]


class ParquetCaptionDataset(Dataset):
    """Loads image-caption pairs directly from a Parquet file containing image bytes and text."""

    def __init__(self, parquet_path: str, transform, image_size: int = 224):
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = str(row["caption"])
        img_bytes = row["image"]

        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))

        image = self.transform(image)
        return image, caption


# ==========================================
# Feature Extraction Helpers
# ==========================================

def encode_images_in_batches(model, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List]:
    """Computes normalized image embeddings using OpenCLIP model."""
    model.eval()
    all_embeddings, all_targets = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if isinstance(targets, torch.Tensor):
                all_targets.append(targets.numpy())
            else:
                all_targets.extend(targets)

    if len(all_targets) > 0 and isinstance(all_targets[0], np.ndarray):
        all_targets = np.concatenate(all_targets, axis=0)

    return np.concatenate(all_embeddings, axis=0), all_targets


def encode_prompts(model, tokenizer, prompts: List[str], device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Computes normalized text embeddings for a list of prompt strings using OpenCLIP tokenizer."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            tokens = tokenizer(batch_prompts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ==========================================
# Metrics Computation
# ==========================================

def compute_retrieval_metrics(sim_matrix: np.ndarray, is_1to1: bool = False, img_labels: np.ndarray = None):
    """Computes T2I and I2T Recall@1, 5, 10."""
    n_text, n_img = sim_matrix.shape

    # 1. Text-to-Image (T2I)
    t2i_r1, t2i_r5, t2i_r10 = 0.0, 0.0, 0.0
    for i in range(n_text):
        ranks = np.argsort(-sim_matrix[i, :])
        if is_1to1:
            matched = (ranks == i)
        else:
            matched = (img_labels[ranks] == i)
        
        hit_ranks = np.where(matched)[0]
        if len(hit_ranks) > 0:
            first_hit = hit_ranks[0]
            if first_hit < 1: t2i_r1 += 1.0
            if first_hit < 5: t2i_r5 += 1.0
            if first_hit < 10: t2i_r10 += 1.0

    t2i_res = {
        "T2I_R@1": t2i_r1 / n_text,
        "T2I_R@5": t2i_r5 / n_text,
        "T2I_R@10": t2i_r10 / n_text,
    }

    # 2. Image-to-Text (I2T)
    i2t_r1, i2t_r5, i2t_r10 = 0.0, 0.0, 0.0
    for j in range(n_img):
        ranks = np.argsort(-sim_matrix[:, j])
        if is_1to1:
            matched = (ranks == j)
        else:
            matched = (ranks == img_labels[j])

        hit_ranks = np.where(matched)[0]
        if len(hit_ranks) > 0:
            first_hit = hit_ranks[0]
            if first_hit < 1: i2t_r1 += 1.0
            if first_hit < 5: i2t_r5 += 1.0
            if first_hit < 10: i2t_r10 += 1.0

    i2t_res = {
        "I2T_R@1": i2t_r1 / n_img,
        "I2T_R@5": i2t_r5 / n_img,
        "I2T_R@10": i2t_r10 / n_img,
    }

    return {**t2i_res, **i2t_res}


def compute_classification_accuracy(sim_matrix: np.ndarray, img_labels: np.ndarray) -> Dict[str, float]:
    """Computes zero-shot image classification accuracy over classes."""
    logits = sim_matrix.T  
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

    top1_correct = 0
    top5_correct = 0
    n_samples = len(img_labels)

    for i in range(n_samples):
        ranking = np.argsort(-probs[i])
        target = img_labels[i]

        if target == ranking[0]:
            top1_correct += 1
        if target in ranking[:5]:
            top5_correct += 1

    return {
        "Acc@1": top1_correct / n_samples,
        "Acc@5": top5_correct / min(5, probs.shape[1]) if probs.shape[1] >= 5 else top1_correct / n_samples
    }


# ==========================================
# Evaluation Processors
# ==========================================

def evaluate_image_class(model, tokenizer, image_transform, dataset_dir: Path, device: torch.device, batch_size: int, image_size: int) -> Dict:
    dataset = ImageFolderFlat(str(dataset_dir), transform=image_transform, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_emb, img_labels = encode_images_in_batches(model, loader, device)

    prompts = [f"{cls.replace('_', ' ')}" for cls in dataset.class_names]
    txt_emb = encode_prompts(model, tokenizer, prompts, device)

    sim_matrix = txt_emb @ img_emb.T  # (num_classes, num_images)

    retrieval_metrics = compute_retrieval_metrics(sim_matrix, is_1to1=False, img_labels=img_labels)
    cls_metrics = compute_classification_accuracy(sim_matrix, img_labels)

    return {
        "num_images": len(dataset),
        "num_classes": len(dataset.class_names),
        **retrieval_metrics,
        **cls_metrics
    }


def evaluate_image_cap(model, tokenizer, image_transform, parquet_path: Path, device: torch.device, batch_size: int, image_size: int) -> Dict:
    dataset = ParquetCaptionDataset(str(parquet_path), transform=image_transform, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_emb, captions = encode_images_in_batches(model, loader, device)
    txt_emb = encode_prompts(model, tokenizer, captions, device)

    sim_matrix = txt_emb @ img_emb.T  # (N, N)
    retrieval_metrics = compute_retrieval_metrics(sim_matrix, is_1to1=True)

    return {
        "num_images": len(dataset),
        "T2I_R@1": retrieval_metrics["T2I_R@1"],
        "I2T_R@1": retrieval_metrics["I2T_R@1"]
    }


# ==========================================
# Console Table Printers
# ==========================================

def print_class_results_table(class_results: Dict[str, Dict]):
    if not class_results:
        return

    print("\n" + "=" * 115)
    print("IMAGE-CLASS BENCHMARKS EVALUATION RESULTS")
    print("=" * 115)
    header = f"{'Dataset':<20} | {'Images':<8} | {'Classes':<8} | {'Acc@1':<8} | {'Acc@5':<8} | {'T2I R@1':<8} | {'I2T R@1':<8}"
    print(header)
    print("-" * 115)

    class_avg = defaultdict(float)
    for ds_name, r in sorted(class_results.items()):
        print(f"{ds_name:<20} | {r['num_images']:<8,} | {r['num_classes']:<8} | {r['Acc@1']*100:6.2f}%  | {r['Acc@5']*100:6.2f}%  | {r['T2I_R@1']*100:6.2f}%  | {r['I2T_R@1']*100:6.2f}%")
        for k in ["Acc@1", "Acc@5", "T2I_R@1", "I2T_R@1"]:
            class_avg[k] += r[k]

    print("-" * 115)
    for k in class_avg:
        class_avg[k] /= len(class_results)
    print(f"{'AVERAGE':<20} | {'-':<8} | {'-':<8} | {class_avg['Acc@1']*100:6.2f}%  | {class_avg['Acc@5']*100:6.2f}%  | {class_avg['T2I_R@1']*100:6.2f}%  | {class_avg['I2T_R@1']*100:6.2f}%")
    print("=" * 115)


def print_cap_results_table(cap_results: Dict[str, Dict]):
    if not cap_results:
        return

    print("\n" + "=" * 65)
    print("IMAGE-CAPTION BENCHMARKS EVALUATION RESULTS")
    print("=" * 65)
    header = f"{'Dataset':<25} | {'Images':<8} | {'T2I R@1':<8} | {'I2T R@1':<8}"
    print(header)
    print("-" * 65)

    cap_avg = defaultdict(float)
    for ds_name, r in sorted(cap_results.items()):
        print(f"{ds_name:<25} | {r['num_images']:<8,} | {r['T2I_R@1']*100:6.2f}%  | {r['I2T_R@1']*100:6.2f}%")
        for k in ["T2I_R@1", "I2T_R@1"]:
            cap_avg[k] += r[k]

    print("-" * 65)
    for k in cap_avg:
        cap_avg[k] /= len(cap_results)
    print(f"{'AVERAGE':<25} | {'-':<8} | {cap_avg['T2I_R@1']*100:6.2f}%  | {cap_avg['I2T_R@1']*100:6.2f}%")
    print("=" * 65)


# ==========================================
# Main Execution Pipeline
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoRSCLIP checkpoint on class and caption evaluation suites.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to GeoRSCLIP .pt checkpoint file")
    parser.add_argument("--model_arch", type=str, default="ViT-B-32", help="OpenCLIP model architecture (e.g., ViT-B-32, ViT-L-14)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model, tokenizer, image_transform = load_georsclip_model(args.ckpt_path, args.model_arch, device)

    benchmark_root = Path(BENCHMARK_ROOT)
    class_dir = benchmark_root / "image-class"
    cap_dir = benchmark_root / "image-cap"

    class_results = {}
    cap_results = {}

    # 1. Evaluate image-class/
    if class_dir.exists():
        for ds in sorted([d for d in class_dir.iterdir() if d.is_dir()]):
            print(f"\n[Image-Class] Evaluating {ds.name}...", end=" ", flush=True)
            t0 = time.time()
            res = evaluate_image_class(model, tokenizer, image_transform, ds, device, args.batch_size, args.image_size)
            print(f"done ({time.time()-t0:.1f}s) | Acc@1: {res['Acc@1']*100:.2f}% | T2I R@1: {res['T2I_R@1']*100:.2f}% | I2T R@1: {res['I2T_R@1']*100:.2f}%")
            class_results[ds.name] = res

    # 2. Evaluate image-cap/
    if cap_dir.exists():
        parquet_files = sorted(list(cap_dir.glob("*.parquet")))
        for p_file in parquet_files:
            ds_name = p_file.stem
            print(f"\n[Image-Cap] Evaluating {ds_name}...", end=" ", flush=True)
            t0 = time.time()
            res = evaluate_image_cap(model, tokenizer, image_transform, p_file, device, args.batch_size, args.image_size)
            print(f"done ({time.time()-t0:.1f}s) | T2I R@1: {res['T2I_R@1']*100:.2f}% | I2T R@1: {res['I2T_R@1']*100:.2f}%")
            cap_results[ds_name] = res

    # 3. Print separate console tables
    print_class_results_table(class_results)
    print_cap_results_table(cap_results)


if __name__ == "__main__":
    main()