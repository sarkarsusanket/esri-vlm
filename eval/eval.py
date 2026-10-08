"""
Image-Text cross-modal retrieval evaluation script.

Usage:
    python eval.py --ckpt_path path/to/checkpoint.ckpt
"""

import argparse
import os
import sys
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
from transformers import CLIPTokenizer

sys.path.append(rf"/home/susanket/esri-vlm/src")
from model import CLIP, build_model
from main import CLIPLightningModule


IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]

BENCHMARK_ROOT = rf"/data/susanket/vlm/eval"


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
        return image, self.class_to_idx[cls_name], str(path)


def compute_image_embeddings(model: CLIP, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_text_embeddings(model: CLIP, class_names: List[str], device: torch.device) -> np.ndarray:
    """Compute normalized text embeddings for class prompt strings."""
    model.eval()
    # Standard prompt template for satellite/remote sensing imagery classification
    prompts = [f"a satellite photo of {cls.replace('_', ' ')}" for cls in class_names]

    with torch.no_grad():
        # Tokenize text prompts
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        text_tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=model.transformer.context_length,
            return_tensors="pt",
        ).input_ids.to(device)

        # Adjust tokenize according to your model implementation
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

    return text_embeddings.cpu().numpy()


def recall_at_k(sim_matrix: np.ndarray, targets: np.ndarray, k: int) -> float:
    """
    Compute Recall@K for retrieval.
    sim_matrix shape: (num_queries, num_candidates)
    targets: ground truth class index for each candidate (or query depending on mode)
    """
    correct = 0
    num_queries = sim_matrix.shape[0]

    for i in range(num_queries):
        ranking = np.argsort(-sim_matrix[i])
        top_k_indices = ranking[:k]

        # Query class is i if query is text prompt, or query class is targets[i]
        query_target = i if len(targets) != num_queries else targets[i]
        candidate_classes = targets[top_k_indices] if len(targets) == sim_matrix.shape[1] else top_k_indices

        if query_target in candidate_classes:
            correct += 1

    return correct / num_queries


def mean_reciprocal_rank(sim_matrix: np.ndarray, targets: np.ndarray) -> float:
    """Compute MRR: mean of 1/rank of first correct match."""
    rr_sum = 0.0
    num_queries = sim_matrix.shape[0]

    for i in range(num_queries):
        ranking = np.argsort(-sim_matrix[i])

        query_target = i if len(targets) != num_queries else targets[i]
        candidate_classes = targets[ranking] if len(targets) == sim_matrix.shape[1] else ranking

        matches = np.where(candidate_classes == query_target)[0]
        if len(matches) > 0:
            rr_sum += 1.0 / (matches[0] + 1)

    return rr_sum / num_queries


def mean_average_precision(sim_matrix: np.ndarray, targets: np.ndarray) -> float:
    """Compute MAP across cross-modal queries."""
    ap_sum = 0.0
    num_queries = sim_matrix.shape[0]

    for i in range(num_queries):
        ranking = np.argsort(-sim_matrix[i])

        query_target = i if len(targets) != num_queries else targets[i]
        candidate_classes = targets[ranking] if len(targets) == sim_matrix.shape[1] else ranking

        relevant = (candidate_classes == query_target).astype(float)
        num_relevant = relevant.sum()

        if num_relevant == 0:
            continue

        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / num_relevant
        ap_sum += ap

    return ap_sum / num_queries


def evaluate_dataset(model: CLIP, dataset: ImageFolderFlat, device: torch.device, batch_size: int = 64) -> Dict:
    """Evaluate Text-to-Image (T2I) retrieval performance."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    img_emb, img_labels = compute_image_embeddings(model, loader, device)
    txt_emb = compute_text_embeddings(model, dataset.class_names, device)

    # Text-to-Image similarity matrix (num_classes, num_images)
    t2i_sim_matrix = txt_emb @ img_emb.T

    results = {
        "num_images": len(dataset),
        "num_classes": len(dataset.class_names),
        "R@1": recall_at_k(t2i_sim_matrix, img_labels, 1),
        "R@5": recall_at_k(t2i_sim_matrix, img_labels, 5),
        "R@10": recall_at_k(t2i_sim_matrix, img_labels, 10),
        "MRR": mean_reciprocal_rank(t2i_sim_matrix, img_labels),
        "MAP": mean_average_precision(t2i_sim_matrix, img_labels),
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
<title>T2I Retrieval Report - {ckpt_name}</title>
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
    tr:hover td {{ background: #f0f7ff; }}
    tr.avg-row td {{
        font-weight: 700;
        background: #e8f4fd;
        border-top: 2px solid #0066cc;
    }}
    .metric-val {{ font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
    <h1>Text-to-Image Retrieval Evaluation Report</h1>
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
    parser = argparse.ArgumentParser(description="Evaluate CLIP checkpoint on cross-modal retrieval benchmark")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to .ckpt checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
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
        for hp_key in ["eval_downstream", "air_temp_data_path", "election_data_path", "_instantiator"]:
            ckpt['hyper_parameters'].pop(hp_key, None)

        lightning_model = CLIPLightningModule(**ckpt["hyper_parameters"]).to(device)
        lightning_model.load_state_dict(ckpt["state_dict"])
        lightning_model.eval()
        model = lightning_model.model
    else:
        model = build_model(ckpt["state_dict"]).to(device)

    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    benchmark_root = Path(BENCHMARK_ROOT)
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
    generate_html_report(args.ckpt_path, BENCHMARK_ROOT, dataset_results, output_path)

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