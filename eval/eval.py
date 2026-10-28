"""
Image-Text Cross-Modal Evaluation Script (Class-based Folders + Parquet Caption Datasets with MRL Support)

Usage:
    python eval/eval.py --ckpt_path /home/susanket/esri-vlm/esri-vlm/ddvi5ics/checkpoints/epoch=51-step=307164.ckpt --output /home/susanket/esri-vlm/esri-vlm/ddvi5ics/eval.html --mrl_dims 64 128 256
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "1,2")


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
from torchvision import transforms
from transformers import AutoTokenizer

sys.path.append(rf"/home/susanket/esri-vlm/src")
sys.path.append(rf"/home/susanket/esri-vlm")
from model import CLIP, build_model
from main import CLIPLightningModule
from sft.loader import load_pretrained_clip

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENCHMARK_ROOT = rf"/data/susanket/vlm/eval"




# ==========================================
# Checkpoint loading
# ==========================================

def is_sft_checkpoint(ckpt: dict) -> bool:
    """SFT checkpoints store the base pretraining checkpoint + precomputed features switch."""
    hp = ckpt.get("hyper_parameters", {})
    return "ckpt_path" in hp and "precomputed_features" in hp


def load_sft_clip(ckpt: dict, device: torch.device):
    """Rebuild a full CLIP from an SFT checkpoint (both full-backbone and precomputed modes)."""
    hp = dict(ckpt["hyper_parameters"])
    clip, tokenizer, _ = load_pretrained_clip(
        ckpt_path=hp["ckpt_path"],
        freeze_image_tower=True,
        freeze_text_tower=True,
        text_cache_size=0,
        device=str(device),
        verbose=False,
    )
    state = ckpt["state_dict"]

    if any(k.startswith("model.") for k in state):  # SFT ran with real backbones
        sub = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        clip.load_state_dict(sub, strict=False)
    else:  # precomputed SFT: transfer the trained probes into the real towers
        with torch.no_grad():
            clip.visual.proj.weight.copy_(state["image_proj.weight"])
            clip.transformer.proj.weight.copy_(state["text_proj.weight"])
            clip.logit_scale.copy_(state["_logit_scale"].reshape(()))
            if clip.logit_bias is not None and state.get("_logit_bias") is not None:
                clip.logit_bias.copy_(state["_logit_bias"].reshape(()))

    return clip.to(device).eval(), tokenizer


# ==========================================
# Datasets
# ==========================================

class ImageFolderFlat(Dataset):
    """Loads images from class subfolders in image-class/ directory."""

    def __init__(self, root: str, image_size: int = 224):
        self.root = Path(root)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

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

    def __init__(self, parquet_path: str, image_size: int = 224):
        self.df = pd.read_parquet(parquet_path)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

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

def get_dino_backbone(model: CLIP, device: torch.device):
    """Loads precomputed DINOv2 backbone if required by checkpoint."""
    if getattr(model.visual, "precomputed", False):
        from model import DINOV2_REPO
        try:
            dino_backbone = torch.hub.load(DINOV2_REPO, model.visual.model_name, trust_repo=True)
        except TypeError:
            dino_backbone = torch.hub.load(DINOV2_REPO, model.visual.model_name)
        return dino_backbone.eval().to(device)
    return None


def encode_images_in_batches(model: CLIP, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List]:
    """Computes normalized image embeddings."""
    model.eval()
    dino_backbone = get_dino_backbone(model, device)
    all_embeddings, all_targets = [], []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            if dino_backbone is not None:
                feats = dino_backbone.forward_features(images)
                images = torch.cat([feats["x_norm_clstoken"].unsqueeze(1), feats["x_norm_patchtokens"]], dim=1)
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


def encode_prompts(model: CLIP, tokenizer, prompts: List[str], device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Computes normalized text embeddings for a list of prompt strings."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=model.transformer.context_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            embeddings = model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ==========================================
# Metrics Computation
# ==========================================

def compute_retrieval_metrics(sim_matrix: np.ndarray, is_1to1: bool = False, img_labels: np.ndarray = None):
    n_text, n_img = sim_matrix.shape

    # 1. Text-to-Image (T2I)
    t2i_r1, t2i_r5, t2i_r10 = 0.0, 0.0, 0.0
    for i in range(n_text):
        ranks = np.argsort(-sim_matrix[i, :])
        matched = (ranks == i) if is_1to1 else (img_labels[ranks] == i)
        
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
        matched = (ranks == j) if is_1to1 else (ranks == img_labels[j])

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
# Evaluation Processors (with MRL breakdown)
# ==========================================

def evaluate_mrl_metrics(txt_emb_full: np.ndarray, img_emb_full: np.ndarray, mrl_dims: List[int], is_1to1: bool, img_labels: np.ndarray = None, evaluate_cls: bool = False) -> Dict[int, Dict]:
    mrl_results = {}
    max_dim = txt_emb_full.shape[1]

    for dim in mrl_dims:
        if dim > max_dim:
            continue
        
        txt_sub = txt_emb_full[:, :dim]
        img_sub = img_emb_full[:, :dim]
        
        txt_sub = txt_sub / np.linalg.norm(txt_sub, axis=-1, keepdims=True)
        img_sub = img_sub / np.linalg.norm(img_sub, axis=-1, keepdims=True)
        
        sim_matrix = txt_sub @ img_sub.T
        retrieval_metrics = compute_retrieval_metrics(sim_matrix, is_1to1=is_1to1, img_labels=img_labels)
        
        dim_res = {**retrieval_metrics}
        if evaluate_cls and img_labels is not None:
            cls_metrics = compute_classification_accuracy(sim_matrix, img_labels)
            dim_res.update(cls_metrics)
            
        mrl_results[dim] = dim_res

    return mrl_results


def evaluate_image_class(model: CLIP, tokenizer, dataset_dir: Path, device: torch.device, batch_size: int, image_size: int, mrl_dims: List[int]) -> Dict:
    dataset = ImageFolderFlat(str(dataset_dir), image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_emb, img_labels = encode_images_in_batches(model, loader, device)

    prompts = [f"{cls.replace('_', ' ')}" for cls in dataset.class_names]
    txt_emb = encode_prompts(model, tokenizer, prompts, device)

    sim_matrix_full = txt_emb @ img_emb.T
    retrieval_metrics_full = compute_retrieval_metrics(sim_matrix_full, is_1to1=False, img_labels=img_labels)
    cls_metrics_full = compute_classification_accuracy(sim_matrix_full, img_labels)

    mrl_metrics = evaluate_mrl_metrics(txt_emb, img_emb, mrl_dims, is_1to1=False, img_labels=img_labels, evaluate_cls=True)

    return {
        "num_images": len(dataset),
        "num_classes": len(dataset.class_names),
        **retrieval_metrics_full,
        **cls_metrics_full,
        "mrl_metrics": mrl_metrics
    }


def evaluate_image_cap(model: CLIP, tokenizer, parquet_path: Path, device: torch.device, batch_size: int, image_size: int, mrl_dims: List[int]) -> Dict:
    dataset = ParquetCaptionDataset(str(parquet_path), image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_emb, captions = encode_images_in_batches(model, loader, device)
    txt_emb = encode_prompts(model, tokenizer, captions, device)

    sim_matrix_full = txt_emb @ img_emb.T
    retrieval_metrics_full = compute_retrieval_metrics(sim_matrix_full, is_1to1=True)

    mrl_metrics = evaluate_mrl_metrics(txt_emb, img_emb, mrl_dims, is_1to1=True, evaluate_cls=False)

    return {
        "num_images": len(dataset),
        "T2I_R@1": retrieval_metrics_full["T2I_R@1"],
        "I2T_R@1": retrieval_metrics_full["I2T_R@1"],
        "mrl_metrics": mrl_metrics
    }


# ==========================================
# Report Generation
# ==========================================

def generate_html_report(
    checkpoint_path: str,
    benchmark_root: str,
    class_results: Dict[str, Dict],
    cap_results: Dict[str, Dict],
    mrl_dims: List[int],
    output_path: str,
):
    ckpt_name = Path(checkpoint_path).name

    # Helper function to build rows for a given dataset results dict and dimension type (full or MRL dim)
    def build_class_rows(get_metrics_fn):
        rows_html = ""
        for ds_name, r in sorted(class_results.items()):
            m = get_metrics_fn(r)
            if not m: continue
            rows_html += f"""
            <tr>
                <td class="ds-name">{ds_name}</td>
                <td>{r['num_images']:,}</td>
                <td>{r['num_classes']}</td>
                <td class="metric-highlight">{m.get('Acc@1', 0)*100:.2f}%</td>
                <td>{m.get('Acc@5', 0)*100:.2f}%</td>
                <td>{m.get('T2I_R@1', 0)*100:.2f}%</td>
                <td>{m.get('T2I_R@5', 0)*100:.2f}%</td>
                <td>{m.get('T2I_R@10', 0)*100:.2f}%</td>
                <td>{m.get('I2T_R@1', 0)*100:.2f}%</td>
                <td>{m.get('I2T_R@5', 0)*100:.2f}%</td>
                <td>{m.get('I2T_R@10', 0)*100:.2f}%</td>
            </tr>"""
        return rows_html

    def build_cap_rows(get_metrics_fn):
        rows_html = ""
        for ds_name, r in sorted(cap_results.items()):
            m = get_metrics_fn(r)
            if not m: continue
            rows_html += f"""
            <tr>
                <td class="ds-name">{ds_name}</td>
                <td>{r['num_images']:,}</td>
                <td class="metric-highlight">{m.get('T2I_R@1', 0)*100:.2f}%</td>
                <td class="metric-highlight">{m.get('I2T_R@1', 0)*100:.2f}%</td>
            </tr>"""
        return rows_html

    def compute_class_avg(get_metrics_fn, keys):
        avg = defaultdict(float)
        count = len(class_results)
        if count == 0: return avg
        for r in class_results.values():
            m = get_metrics_fn(r)
            for k in keys:
                avg[k] += m.get(k, 0)
        for k in avg: avg[k] /= count
        return avg

    def compute_cap_avg(get_metrics_fn, keys):
        avg = defaultdict(float)
        count = len(cap_results)
        if count == 0: return avg
        for r in cap_results.values():
            m = get_metrics_fn(r)
            for k in keys:
                avg[k] += m.get(k, 0)
        for k in avg: avg[k] /= count
        return avg

    class_keys = ["Acc@1", "Acc@5", "T2I_R@1", "T2I_R@5", "T2I_R@10", "I2T_R@1", "I2T_R@5", "I2T_R@10"]
    cap_keys = ["T2I_R@1", "I2T_R@1"]

    # Generate Full Dimension Section
    full_class_rows = build_class_rows(lambda r: r)
    full_class_avg = compute_class_avg(lambda r: r, class_keys)
    full_cap_rows = build_cap_rows(lambda r: r)
    full_cap_avg = compute_cap_avg(lambda r: r, cap_keys)

    sections_html = f"""
    <div class="section-title"><span>Full Dimension Evaluation</span> <span class="badge">Base Embeddings</span></div>
    
    <div class="card">
        <h3>Image-Class Benchmarks</h3>
        <table>
            <thead>
                <tr><th>Dataset</th><th>Images</th><th>Classes</th><th>Acc@1</th><th>Acc@5</th><th>T2I R@1</th><th>T2I R@5</th><th>T2I R@10</th><th>I2T R@1</th><th>I2T R@5</th><th>I2T R@10</th></tr>
            </thead>
            <tbody>
                {full_class_rows}
                {f'<tr class="avg-row"><td>Average</td><td>-</td><td>-</td><td>{full_class_avg["Acc@1"]*100:.2f}%</td><td>{full_class_avg["Acc@5"]*100:.2f}%</td><td>{full_class_avg["T2I_R@1"]*100:.2f}%</td><td>{full_class_avg["T2I_R@5"]*100:.2f}%</td><td>{full_class_avg["T2I_R@10"]*100:.2f}%</td><td>{full_class_avg["I2T_R@1"]*100:.2f}%</td><td>{full_class_avg["I2T_R@5"]*100:.2f}%</td><td>{full_class_avg["I2T_R@10"]*100:.2f}%</td></tr>' if class_results else ''}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h3>Image-Caption Benchmarks</h3>
        <table>
            <thead>
                <tr><th>Dataset</th><th>Images</th><th>T2I R@1</th><th>I2T R@1</th></tr>
            </thead>
            <tbody>
                {full_cap_rows}
                {f'<tr class="avg-row"><td>Average</td><td>-</td><td>{full_cap_avg["T2I_R@1"]*100:.2f}%</td><td>{full_cap_avg["I2T_R@1"]*100:.2f}%</td></tr>' if cap_results else ''}
            </tbody>
        </table>
    </div>
    """

    # Generate Separate Tables for each MRL Dimension
    for dim in mrl_dims:
        dim_class_rows = build_class_rows(lambda r: r.get("mrl_metrics", {}).get(dim, {}))
        dim_class_avg = compute_class_avg(lambda r: r.get("mrl_metrics", {}).get(dim, {}), class_keys)
        
        dim_cap_rows = build_cap_rows(lambda r: r.get("mrl_metrics", {}).get(dim, {}))
        dim_cap_avg = compute_cap_avg(lambda r: r.get("mrl_metrics", {}).get(dim, {}), cap_keys)

        sections_html += f"""
        <div class="section-title" style="margin-top: 48px;"><span>MRL Dimension: {dim}</span> <span class="badge">Truncated Embeddings</span></div>
        
        <div class="card">
            <h3>Image-Class Benchmarks (Dim {dim})</h3>
            <table>
                <thead>
                    <tr><th>Dataset</th><th>Images</th><th>Classes</th><th>Acc@1</th><th>Acc@5</th><th>T2I R@1</th><th>T2I R@5</th><th>T2I R@10</th><th>I2T R@1</th><th>I2T R@5</th><th>I2T R@10</th></tr>
                </thead>
                <tbody>
                    {dim_class_rows}
                    {f'<tr class="avg-row"><td>Average</td><td>-</td><td>-</td><td>{dim_class_avg["Acc@1"]*100:.2f}%</td><td>{dim_class_avg["Acc@5"]*100:.2f}%</td><td>{dim_class_avg["T2I_R@1"]*100:.2f}%</td><td>{dim_class_avg["T2I_R@5"]*100:.2f}%</td><td>{dim_class_avg["T2I_R@10"]*100:.2f}%</td><td>{dim_class_avg["I2T_R@1"]*100:.2f}%</td><td>{dim_class_avg["I2T_R@5"]*100:.2f}%</td><td>{dim_class_avg["I2T_R@10"]*100:.2f}%</td></tr>' if class_results else ''}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h3>Image-Caption Benchmarks (Dim {dim})</h3>
            <table>
                <thead>
                    <tr><th>Dataset</th><th>Images</th><th>T2I R@1</th><th>I2T R@1</th></tr>
                </thead>
                <tbody>
                    {dim_cap_rows}
                    {f'<tr class="avg-row"><td>Average</td><td>-</td><td>{dim_cap_avg["T2I_R@1"]*100:.2f}%</td><td>{dim_cap_avg["I2T_R@1"]*100:.2f}%</td></tr>' if cap_results else ''}
                </tbody>
            </table>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM MRL Evaluation Report - {ckpt_name}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {{
        --bg-main: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --accent: #2563eb;
        --border: #e2e8f0;
    }}
    body {{
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-main);
        color: var(--text-primary);
        margin: 0;
        padding: 40px 20px;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    .header {{
        background: var(--card-bg);
        padding: 24px 32px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid var(--border);
        margin-bottom: 24px;
    }}
    .header h1 {{
        margin: 0 0 8px 0;
        font-size: 24px;
        font-weight: 700;
    }}
    .header .meta {{
        font-size: 14px;
        color: var(--text-secondary);
    }}
    .section-title {{
        font-size: 18px;
        font-weight: 600;
        margin: 32px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .card {{
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        overflow: hidden;
        margin-bottom: 24px;
        padding: 16px 24px;
    }}
    .card h3 {{
        margin-top: 8px;
        font-size: 15px;
        color: var(--text-secondary);
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 14px;
        margin-top: 10px;
    }}
    th {{
        background: #f1f5f9;
        padding: 12px 16px;
        font-weight: 600;
        color: #475569;
        border-bottom: 1px solid var(--border);
    }}
    td {{
        padding: 14px 16px;
        border-bottom: 1px solid var(--border);
    }}
    tr:hover td {{ background: #f8fafc; }}
    .ds-name {{ font-weight: 500; color: var(--text-primary); }}
    .metric-highlight {{ font-weight: 600; color: var(--accent); }}
    .avg-row td {{
        background: #eff6ff;
        font-weight: 700;
        color: #1e40af;
        border-top: 2px solid #bfdbfe;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        background: #e0e7ff;
        color: #3730a3;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>VLM MRL Evaluation Dashboard</h1>
        <div class="meta">
            <strong>Checkpoint:</strong> {ckpt_name} &nbsp;|&nbsp; 
            <strong>Benchmark Root:</strong> {benchmark_root} &nbsp;|&nbsp;
            <strong>Evaluated MRL Dimensions:</strong> {mrl_dims}
        </div>
    </div>

    {sections_html}
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport successfully saved to {output_path}")


# ==========================================
# Main Execution Pipeline
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP checkpoint on class and caption evaluation suites with MRL classification and retrieval support.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mrl_dims", type=int, nargs="+", default=[64, 128, 256], help="List of MRL dimensions to evaluate")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading checkpoint: {args.ckpt_path}")
    print(f"Evaluating MRL dimensions: {args.mrl_dims}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)

    if is_sft_checkpoint(ckpt):
        model, tokenizer = load_sft_clip(ckpt, device)
    elif "hyper_parameters" in ckpt:
        for hp_key in ["eval_downstream", "air_temp_data_path", "election_data_path", "_instantiator"]:
            ckpt['hyper_parameters'].pop(hp_key, None)

        model_hp = dict(ckpt["hyper_parameters"])
        model_hp["load_text_backbone"] = True  
        lightning_model = CLIPLightningModule(**model_hp).to(device)
        lightning_model.load_state_dict(ckpt["state_dict"])
        lightning_model.eval()
        model = lightning_model.model
        tokenizer = lightning_model.tokenizer
    else:
        model = build_model(ckpt["state_dict"]).to(device)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")

    model.eval()

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
            res = evaluate_image_class(model, tokenizer, ds, device, args.batch_size, args.image_size, args.mrl_dims)
            print(f"done ({time.time()-t0:.1f}s) | Acc@1: {res['Acc@1']*100:.2f}% | T2I R@1: {res['T2I_R@1']*100:.2f}%")
            for dim, mrl_res in res["mrl_metrics"].items():
                print(f"   -> MRL Dim {dim}: Acc@1: {mrl_res.get('Acc@1', 0)*100:.2f}% | T2I R@1: {mrl_res['T2I_R@1']*100:.2f}%")
            class_results[ds.name] = res

    # 2. Evaluate image-cap/
    if cap_dir.exists():
        parquet_files = sorted(list(cap_dir.glob("*.parquet")))
        for p_file in parquet_files:
            ds_name = p_file.stem
            print(f"\n[Image-Cap] Evaluating {ds_name}...", end=" ", flush=True)
            t0 = time.time()
            res = evaluate_image_cap(model, tokenizer, p_file, device, args.batch_size, args.image_size, args.mrl_dims)
            print(f"done ({time.time()-t0:.1f}s) | T2I R@1: {res['T2I_R@1']*100:.2f}%")
            for dim, mrl_res in res["mrl_metrics"].items():
                print(f"   -> MRL Dim {dim}: T2I R@1: {mrl_res['T2I_R@1']*100:.2f}%")
            cap_results[ds_name] = res

    # 3. Generate HTML Report
    output_path = args.output or f"eval_report_{Path(args.ckpt_path).stem}_mrl.html"
    generate_html_report(args.ckpt_path, BENCHMARK_ROOT, class_results, cap_results, args.mrl_dims, output_path)


if __name__ == "__main__":
    main()