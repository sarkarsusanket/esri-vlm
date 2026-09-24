"""
Image retrieval evaluation script.

Usage:
    python eval.py --ckpt_path path/to/checkpoint.ckpt --benchmark_root path/to/benchmark

Benchmark structure:
    benchmark_root/
        dataset_1/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img1.jpg
                ...
        dataset_2/
            ...
"""

import argparse
import base64
import io
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import CLIP, build_model
from main import CLIPLightningModule


IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]

BENCHMARK_ROOT = rf""


class ImageFolderFlat(Dataset):
    """Loads all images from a directory of class subfolders."""

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
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
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
        return image, self.class_to_idx[cls_name], str(path)


def compute_embeddings(model: CLIP, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized image embeddings for all samples."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def recall_at_k(sim_matrix: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute Recall@K: fraction of queries where correct class is in top-k."""
    correct = 0
    n = len(labels)
    for i in range(n):
        ranking = np.argsort(-sim_matrix[i])
        top_k_classes = labels[ranking[:k]]
        if labels[i] in top_k_classes:
            correct += 1
    return correct / n


def mean_reciprocal_rank(sim_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute MRR: mean of 1/rank of first correct result."""
    rr_sum = 0.0
    n = len(labels)
    for i in range(n):
        ranking = np.argsort(-sim_matrix[i])
        ranked_labels = labels[ranking]
        matches = np.where(ranked_labels == labels[i])[0]
        if len(matches) > 0:
            rr_sum += 1.0 / (matches[0] + 1)
    return rr_sum / n


def mean_average_precision(sim_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute MAP: mean of average precision across all queries."""
    ap_sum = 0.0
    n = len(labels)
    for i in range(n):
        ranking = np.argsort(-sim_matrix[i])
        ranked_labels = labels[ranking]
        relevant = (ranked_labels == labels[i]).astype(float)
        num_relevant = relevant.sum()
        if num_relevant == 0:
            continue
        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / num_relevant
        ap_sum += ap
    return ap_sum / n


def evaluate_dataset(model: CLIP, dataset: ImageFolderFlat, device: torch.device, batch_size: int = 64) -> Dict:
    """Evaluate retrieval on a single dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    emb, labels = compute_embeddings(model, loader, device)

    sim_matrix = emb @ emb.T

    n_classes = len(dataset.class_names)

    results = {
        "num_images": len(dataset),
        "num_classes": n_classes,
        "R@1": recall_at_k(sim_matrix, labels, 1),
        "R@5": recall_at_k(sim_matrix, labels, 5),
        "R@10": recall_at_k(sim_matrix, labels, 10),
        "MRR": mean_reciprocal_rank(sim_matrix, labels),
        "MAP": mean_average_precision(sim_matrix, labels),
    }
    return results


def generate_html_report(
    checkpoint_path: str,
    benchmark_root: str,
    dataset_results: Dict[str, Dict],
    output_path: str,
):
    """Generate a styled HTML report."""
    ckpt_name = Path(checkpoint_path).stem
    bench_name = Path(benchmark_root).name

    avg_metrics = defaultdict(float)
    for ds_results in dataset_results.values():
        for key in ["R@1", "R@5", "R@10", "MRR", "MAP"]:
            avg_metrics[key] += ds_results[key]
    n_datasets = len(dataset_results)
    for key in avg_metrics:
        avg_metrics[key] /= max(n_datasets, 1)

    rows_html = ""
    for ds_name in sorted(dataset_results.keys()):
        r = dataset_results[ds_name]
        rows_html += f"""
        <tr>
            <td>{ds_name}</td>
            <td>{r['num_images']}</td>
            <td>{r['num_classes']}</td>
            <td>{r['R@1']:.4f}</td>
            <td>{r['R@5']:.4f}</td>
            <td>{r['R@10']:.4f}</td>
            <td>{r['MRR']:.4f}</td>
            <td>{r['MAP']:.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evaluation Report - {ckpt_name}</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 1100px;
        margin: 40px auto;
        padding: 0 20px;
        background: #f8f9fa;
        color: #212529;
    }}
    h1 {{
        text-align: center;
        color: #1a1a2e;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
    }}
    .meta {{
        text-align: center;
        color: #666;
        margin-bottom: 30px;
        font-size: 14px;
    }}
    .meta strong {{ color: #333; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    th {{
        background: #0066cc;
        color: white;
        padding: 12px 16px;
        text-align: left;
        font-weight: 600;
    }}
    td {{
        padding: 10px 16px;
        border-bottom: 1px solid #eee;
    }}
    tr:hover td {{
        background: #f0f7ff;
    }}
    tr.avg-row td {{
        font-weight: 700;
        background: #e8f4fd;
        border-top: 2px solid #0066cc;
    }}
    .metric-val {{
        font-variant-numeric: tabular-nums;
    }}
</style>
</head>
<body>
    <h1>Image Retrieval Evaluation Report</h1>
    <div class="meta">
        <strong>Checkpoint:</strong> {Path(checkpoint_path).name}<br>
        <strong>Benchmark:</strong> {bench_name} ({n_datasets} datasets)<br>
        <strong>Total images:</strong> {sum(r['num_images'] for r in dataset_results.values())}
    </div>
    <table>
        <thead>
            <tr>
                <th>Dataset</th>
                <th>Images</th>
                <th>Classes</th>
                <th>R@1</th>
                <th>R@5</th>
                <th>R@10</th>
                <th>MRR</th>
                <th>MAP</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
            <tr class="avg-row">
                <td>Average</td>
                <td>{sum(r['num_images'] for r in dataset_results.values())}</td>
                <td>-</td>
                <td class="metric-val">{avg_metrics['R@1']:.4f}</td>
                <td class="metric-val">{avg_metrics['R@5']:.4f}</td>
                <td class="metric-val">{avg_metrics['R@10']:.4f}</td>
                <td class="metric-val">{avg_metrics['MRR']:.4f}</td>
                <td class="metric-val">{avg_metrics['MAP']:.4f}</td>
            </tr>
        </tbody>
    </table>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP checkpoint on image retrieval benchmark")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path (default: eval_report_<ckpt_name>.html)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)

    if "hyper_parameters" in ckpt:
        lightning_model = CLIPLightningModule(**ckpt["hyper_parameters"]).to(device)
        lightning_model.load_state_dict(ckpt["state_dict"])
        lightning_model.eval()
        model = lightning_model.model
    else:
        model = build_model(ckpt["state_dict"]).to(device)

    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    benchmark_root = Path(args.benchmark_root)
    dataset_dirs = sorted([d for d in benchmark_root.iterdir() if d.is_dir()])

    if not dataset_dirs:
        print(f"No dataset directories found in {benchmark_root}")
        return

    print(f"Found {len(dataset_dirs)} datasets: {[d.name for d in dataset_dirs]}")

    dataset_results = {}
    for ds_dir in dataset_dirs:
        print(f"\nEvaluating: {ds_dir.name} ...", end=" ", flush=True)
        t0 = time.time()

        dataset = ImageFolderFlat(str(ds_dir), image_size=args.image_size)
        if len(dataset) == 0:
            print("SKIP (no images)")
            continue

        results = evaluate_dataset(model, dataset, device, batch_size=args.batch_size)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) | images={results['num_images']} classes={results['num_classes']} "
              f"R@1={results['R@1']:.4f} R@5={results['R@5']:.4f} MAP={results['MAP']:.4f}")

        dataset_results[ds_dir.name] = results

    if not dataset_results:
        print("No datasets evaluated.")
        return

    output_path = args.output or f"eval_report_{Path(args.ckpt_path).stem}.html"
    generate_html_report(args.ckpt_path, args.benchmark_root, dataset_results, output_path)

    print("\n===== Summary =====")
    avg = defaultdict(float)
    for r in dataset_results.values():
        for k in ["R@1", "R@5", "R@10", "MRR", "MAP"]:
            avg[k] += r[k]
    n = len(dataset_results)
    for k in avg:
        avg[k] /= n
    print(f"  R@1  = {avg['R@1']:.4f}")
    print(f"  R@5  = {avg['R@5']:.4f}")
    print(f"  R@10 = {avg['R@10']:.4f}")
    print(f"  MRR  = {avg['MRR']:.4f}")
    print(f"  MAP  = {avg['MAP']:.4f}")


if __name__ == "__main__":
    main()
