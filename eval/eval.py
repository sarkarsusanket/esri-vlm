"""
Image-Text Cross-Modal Evaluation Script (Class-based Folders + Parquet Caption Datasets)

Usage:
    python eval.py --ckpt_path path/to/checkpoint.ckpt
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
from torchvision import transforms
from transformers import AutoTokenizer

sys.path.append(rf"/home/susanket/esri-vlm/src")
from model import CLIP, build_model
from main import CLIPLightningModule

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BENCHMARK_ROOT = rf"/data/susanket/vlm/eval"


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
    """
    Computes T2I and I2T Recall@1, 5, 10.
    
    sim_matrix shape: (N_text, N_image)
    - If is_1to1=True: Text i matches Image i.
    - If is_1to1=False: Text i represents class i, matching images with img_labels == i.
    """
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
    """
    Computes zero-shot image classification accuracy over classes.
    sim_matrix: (num_classes, num_images)
    """
    # Transpose similarity matrix to (num_images, num_classes)
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

def evaluate_image_class(model: CLIP, tokenizer, dataset_dir: Path, device: torch.device, batch_size: int, image_size: int) -> Dict:
    """Runs evaluation on a class-based folder dataset."""
    dataset = ImageFolderFlat(str(dataset_dir), image_size=image_size)
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


def evaluate_image_cap(model: CLIP, tokenizer, parquet_path: Path, device: torch.device, batch_size: int, image_size: int) -> Dict:
    """Runs evaluation on an image-caption parquet dataset."""
    dataset = ParquetCaptionDataset(str(parquet_path), image_size=image_size)
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
# Report Generation
# ==========================================

def generate_html_report(
    checkpoint_path: str,
    benchmark_root: str,
    class_results: Dict[str, Dict],
    cap_results: Dict[str, Dict],
    output_path: str,
):
    """Generates an aesthetic dashboard HTML report."""
    ckpt_name = Path(checkpoint_path).name

    # Calculate class averages
    class_avg = defaultdict(float)
    if class_results:
        for r in class_results.values():
            for k in ["T2I_R@1", "T2I_R@5", "T2I_R@10", "I2T_R@1", "I2T_R@5", "I2T_R@10", "Acc@1", "Acc@5"]:
                class_avg[k] += r[k]
        for k in class_avg:
            class_avg[k] /= len(class_results)

    # Calculate caption averages
    cap_avg = defaultdict(float)
    if cap_results:
        for r in cap_results.values():
            for k in ["T2I_R@1", "I2T_R@1"]:
                cap_avg[k] += r[k]
        for k in cap_avg:
            cap_avg[k] /= len(cap_results)

    # Class rows
    class_rows_html = ""
    for ds_name, r in sorted(class_results.items()):
        class_rows_html += f"""
        <tr>
            <td class="ds-name">{ds_name}</td>
            <td>{r['num_images']:,}</td>
            <td>{r['num_classes']}</td>
            <td class="metric-highlight">{r['Acc@1']*100:.2f}%</td>
            <td>{r['Acc@5']*100:.2f}%</td>
            <td>{r['T2I_R@1']*100:.2f}%</td>
            <td>{r['T2I_R@5']*100:.2f}%</td>
            <td>{r['T2I_R@10']*100:.2f}%</td>
            <td>{r['I2T_R@1']*100:.2f}%</td>
            <td>{r['I2T_R@5']*100:.2f}%</td>
            <td>{r['I2T_R@10']*100:.2f}%</td>
        </tr>"""

    # Caption rows
    cap_rows_html = ""
    for ds_name, r in sorted(cap_results.items()):
        cap_rows_html += f"""
        <tr>
            <td class="ds-name">{ds_name}</td>
            <td>{r['num_images']:,}</td>
            <td class="metric-highlight">{r['T2I_R@1']*100:.2f}%</td>
            <td class="metric-highlight">{r['I2T_R@1']*100:.2f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM Evaluation Report - {ckpt_name}</title>
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
        margin-bottom: 32px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 14px;
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
    tr:last-child td {{ border-bottom: none; }}
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
        <h1>VLM Evaluation Dashboard</h1>
        <div class="meta">
            <strong>Checkpoint:</strong> {ckpt_name} &nbsp;|&nbsp; 
            <strong>Benchmark Root:</strong> {benchmark_root}
        </div>
    </div>

    <!-- Image-Class Section -->
    <div class="section-title">
        <span>Image-Class Benchmarks</span>
        <span class="badge">Zero-Shot Classification & Retrieval</span>
    </div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Images</th>
                    <th>Classes</th>
                    <th>Acc@1</th>
                    <th>Acc@5</th>
                    <th>T2I R@1</th>
                    <th>T2I R@5</th>
                    <th>T2I R@10</th>
                    <th>I2T R@1</th>
                    <th>I2T R@5</th>
                    <th>I2T R@10</th>
                </tr>
            </thead>
            <tbody>
                {class_rows_html}
                {'<tr class="avg-row"><td>Average</td><td>-</td><td>-</td><td>' + f"{class_avg['Acc@1']*100:.2f}%</td><td>{class_avg['Acc@5']*100:.2f}%</td><td>{class_avg['T2I_R@1']*100:.2f}%</td><td>{class_avg['T2I_R@5']*100:.2f}%</td><td>{class_avg['T2I_R@10']*100:.2f}%</td><td>{class_avg['I2T_R@1']*100:.2f}%</td><td>{class_avg['I2T_R@5']*100:.2f}%</td><td>{class_avg['I2T_R@10']*100:.2f}%</td></tr>" if class_results else ''}
            </tbody>
        </table>
    </div>

    <!-- Image-Cap Section -->
    <div class="section-title">
        <span>Image-Caption Benchmarks</span>
        <span class="badge">Fine-Grained Retrieval</span>
    </div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Images</th>
                    <th>T2I R@1</th>
                    <th>I2T R@1</th>
                </tr>
            </thead>
            <tbody>
                {cap_rows_html}
                {'<tr class="avg-row"><td>Average</td><td>-</td><td>' + f"{cap_avg['T2I_R@1']*100:.2f}%</td><td>{cap_avg['I2T_R@1']*100:.2f}%</td></tr>" if cap_results else ''}
            </tbody>
        </table>
    </div>
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
    parser = argparse.ArgumentParser(description="Evaluate CLIP checkpoint on class and caption evaluation suites.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)

    if "hyper_parameters" in ckpt:
        for hp_key in ["eval_downstream", "air_temp_data_path", "election_data_path", "_instantiator"]:
            ckpt['hyper_parameters'].pop(hp_key, None)

        lightning_model = CLIPLightningModule(**ckpt["hyper_parameters"]).to(device)
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
            res = evaluate_image_class(model, tokenizer, ds, device, args.batch_size, args.image_size)
            print(f"done ({time.time()-t0:.1f}s) | Acc@1: {res['Acc@1']*100:.2f}% | T2I R@1: {res['T2I_R@1']*100:.2f}% | I2T R@1: {res['I2T_R@1']*100:.2f}%")
            class_results[ds.name] = res

    # 2. Evaluate image-cap/
    if cap_dir.exists():
        parquet_files = sorted(list(cap_dir.glob("*.parquet")))
        for p_file in parquet_files:
            ds_name = p_file.stem
            print(f"\n[Image-Cap] Evaluating {ds_name}...", end=" ", flush=True)
            t0 = time.time()
            res = evaluate_image_cap(model, tokenizer, p_file, device, args.batch_size, args.image_size)
            print(f"done ({time.time()-t0:.1f}s) | T2I R@1: {res['T2I_R@1']*100:.2f}% | I2T R@1: {res['I2T_R@1']*100:.2f}%")
            cap_results[ds_name] = res

    # 3. Generate HTML
    output_path = args.output or f"eval_report_{Path(args.ckpt_path).stem}.html"
    generate_html_report(args.ckpt_path, BENCHMARK_ROOT, class_results, cap_results, output_path)


if __name__ == "__main__":
    main()