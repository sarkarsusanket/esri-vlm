"""
Unified SOTA Remote Sensing VLM Evaluation Harness

Supported models (and how each is actually loaded):
    gitrsclip  - lcybuaa/Git-RSCLIP, a SigLIP model (HF AutoModel). Text must be padded to 64 tokens.
    georsclip  - RS5M_ViT-B-32.pt: open_clip ViT-B/32 (OpenAI arch, QuickGELU) + raw state_dict.
    skyclip    - Original SkyScript open_clip ViT-L-14 checkpoint (the BiliSakura HF port is
                 vision-encoder-only, so it cannot do text retrieval/zero-shot).
    remoteclip - RemoteCLIP-ViT-L-14.pt from HF hub (chendelong/RemoteCLIP), open_clip ViT-L-14.
    farclip    - FarSLIP (ZhenShiL/FarSLIP on HF hub), open_clip ViT-B-16 with QuickGELU.

Checkpoints are auto-downloaded if the --*_ckpt path is not given.

Usage:
    python eval-sota.py \
        --benchmark_root /data/susanket/vlm/eval \
        --models gitrsclip georsclip skyclip remoteclip farclip \
        --georsclip_ckpt /data/susanket/vlm/sota/RS5M_ViT-B-32.pt \
        --batch_size 64
"""

import argparse
import io
import os
import tqdm
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
# SKYCLIP_URL = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_L14_top50pct.zip"

# ==========================================
# Datasets (No Transform Applied Inside)
# ==========================================

def _decode_image(obj) -> Image.Image:
    """Handles raw bytes or HF-style {'bytes': ..., 'path': ...} dicts."""
    try:
        if isinstance(obj, dict):
            obj = obj.get("bytes")
        return Image.open(io.BytesIO(obj)).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))


class ImageFolderRaw(Dataset):
    """Loads raw PIL images from class subfolders."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.class_names: List[str] = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.samples: List[Tuple[Path, str]] = []
        for cls_name in self.class_names:
            for p in sorted((self.root / cls_name).iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, cls_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_name = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        return image, self.class_to_idx[cls_name]


class ParquetCaptionRaw(Dataset):
    """Loads raw PIL images and captions from image-cap/*.parquet files."""

    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return _decode_image(row["image"]), str(row["caption"])


def raw_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ==========================================
# Helpers
# ==========================================

def _to_tensor(out) -> torch.Tensor:
    """Newer transformers versions return ModelOutput objects from get_*_features()."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise TypeError(f"Cannot extract features from {type(out)}")


def _norm(t: torch.Tensor) -> np.ndarray:
    return F.normalize(t.float(), dim=-1).cpu().numpy()


def load_state_dict_flex(model: torch.nn.Module, ckpt_path: str):
    """
    Load an open_clip-style checkpoint (raw state_dict or {'state_dict': ...}, optional 'module.' prefix).
    Raises if real weights are missing, so we never silently evaluate a randomly-initialised model.
    """
    # weights_only=False: some of these .pt files contain non-tensor pickled objects (PyTorch>=2.6 default breaks).
    # Only use checkpoints you trust.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    msg = model.load_state_dict(sd, strict=False)
    missing = [k for k in msg.missing_keys if k != "logit_scale"]
    print(f"  loaded {os.path.basename(ckpt_path)}: {len(missing)} missing, {len(msg.unexpected_keys)} unexpected keys")
    if missing:
        raise RuntimeError(f"Checkpoint does not match architecture; missing e.g. {missing[:5]}")
    return msg


def hf_download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename)


def resolve_skyclip_ckpt(path: Optional[str]) -> str:
    """Accepts a .pt file, a directory/zip containing one, or downloads the official zip."""
    cache = Path(os.environ.get("SKYCLIP_CACHE", Path.home() / ".cache" / "skyclip"))
    cache.mkdir(parents=True, exist_ok=True)
    if path and path.endswith(".pt") and os.path.exists(path):
        return path
    if path and path.endswith(".zip") and os.path.exists(path):
        zip_path = Path(path)
    elif path and os.path.isdir(path):
        pts = sorted(Path(path).rglob("*.pt"))
        if pts:
            return str(pts[-1])
        raise FileNotFoundError(f"No .pt found in {path}")
    else:
        zip_path = cache / "SkyCLIP_ViT_L14_top50pct.zip"
        if not zip_path.exists():
            print(f"  downloading SkyCLIP checkpoint -> {zip_path}")
            urllib.request.urlretrieve(SKYCLIP_URL, zip_path)
    out_dir = cache / zip_path.stem
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    pts = sorted(out_dir.rglob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt found inside {zip_path}")
    return str(pts[-1])


# ==========================================
# Model Adapters (Unified API)
# ==========================================

class BaseVLMAdapter:
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class GitRSCLIPAdapter(BaseVLMAdapter):
    """SigLIP-architecture model. Text is padded to max_length=64 (as in its model card)."""

    def __init__(self, device: torch.device, model_id: str = "lcybuaa/Git-RSCLIP"):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs = {"pixel_values": inputs["pixel_values"].to(self.model.dtype)}
        return _norm(_to_tensor(self.model.get_image_features(**inputs)))

    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        inputs = {"input_ids": inputs["input_ids"]}
        return _norm(_to_tensor(self.model.get_text_features(**inputs)))


class OpenCLIPAdapter(BaseVLMAdapter):
    def __init__(self, model_name: str, ckpt_path: str, device: torch.device):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
        load_state_dict_flex(self.model, ckpt_path)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_images(self, images):
        x = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        return _norm(self.model.encode_image(x))

    @torch.no_grad()
    def encode_texts(self, texts):
        tok = self.tokenizer(texts).to(self.device)
        return _norm(self.model.encode_text(tok))


def build_georsclip(ckpt: str, device):
    # OpenAI-derived ViT-B/32 -> QuickGELU variant (official GeoRSCLIP loads on top of openai weights)
    return OpenCLIPAdapter("ViT-B-32-quickgelu", ckpt, device)


def build_skyclip(ckpt: Optional[str], device):
    return OpenCLIPAdapter("ViT-L-14-quickgelu", resolve_skyclip_ckpt(ckpt), device)


def build_remoteclip(ckpt: Optional[str], device):
    if not ckpt or not os.path.exists(ckpt):
        ckpt = hf_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt")
    # RemoteCLIP's official loader uses plain 'ViT-L-14'
    return OpenCLIPAdapter("ViT-L-14", ckpt, device)


def build_farclip(ckpt: Optional[str], device):
    if not ckpt or not os.path.exists(ckpt):
        ckpt = hf_download("ZhenShiL/FarSLIP", "FarSLIP2_ViT-B-16.pt")
    # FarSLIP is run with --force-quick-gelu
    return OpenCLIPAdapter("ViT-B-16-quickgelu", ckpt, device)


# ==========================================
# Metric Calculations
# ==========================================

def compute_retrieval_metrics(sim_matrix: np.ndarray, is_1to1: bool = False, img_labels: np.ndarray = None) -> Dict:
    n_text, n_img = sim_matrix.shape

    t2i = np.zeros(3)
    for i in range(n_text):
        order = np.argsort(-sim_matrix[i, :])
        matched = (order == i) if is_1to1 else (img_labels[order] == i)
        hits = np.flatnonzero(matched)
        if len(hits):
            t2i += [hits[0] < 1, hits[0] < 5, hits[0] < 10]

    i2t = np.zeros(3)
    for j in range(n_img):
        order = np.argsort(-sim_matrix[:, j])
        matched = (order == j) if is_1to1 else (order == img_labels[j])
        hits = np.flatnonzero(matched)
        if len(hits):
            i2t += [hits[0] < 1, hits[0] < 5, hits[0] < 10]

    t2i /= n_text
    i2t /= n_img
    return {
        "T2I_R@1": t2i[0], "T2I_R@5": t2i[1], "T2I_R@10": t2i[2],
        "I2T_R@1": i2t[0], "I2T_R@5": i2t[1], "I2T_R@10": i2t[2],
    }


def compute_classification_accuracy(sim_matrix: np.ndarray, img_labels: np.ndarray) -> Dict:
    logits = sim_matrix.T  # (num_images, num_classes)
    k = min(5, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    top1 = (topk[:, 0] == img_labels).mean()
    top5 = (topk == img_labels[:, None]).any(axis=1).mean()
    return {"Acc@1": float(top1), "Acc@5": float(top5)}


# ==========================================
# Benchmark Runner
# ==========================================

def run_evaluation(adapter: BaseVLMAdapter, benchmark_root: Path, batch_size: int = 64,
                   num_workers: int = 4) -> Tuple[Dict, Dict]:
    class_dir = benchmark_root / "image-class"
    cap_dir = benchmark_root / "image-cap"
    class_results, cap_results = {}, {}

    print(f"Started eval: {adapter}")

    def make_loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=raw_collate_fn, num_workers=num_workers)

    if class_dir.exists():
        for ds_path in tqdm.tqdm(sorted(d for d in class_dir.iterdir() if d.is_dir())):
            print(f"Eval for {ds_path}")
            dataset = ImageFolderRaw(str(ds_path))
            img_embs, img_labels = [], []
            for imgs, lbls in make_loader(dataset):
                img_embs.append(adapter.encode_images(imgs))
                img_labels.extend(lbls)
            img_embs = np.concatenate(img_embs, axis=0)
            img_labels = np.array(img_labels)

            prompts = [f"{c.replace('_', ' ')}" for c in dataset.class_names]
            txt_embs = adapter.encode_texts(prompts)

            sim = txt_embs @ img_embs.T
            class_results[ds_path.name] = {
                "num_images": len(dataset),
                "num_classes": len(dataset.class_names),
                **compute_retrieval_metrics(sim, is_1to1=False, img_labels=img_labels),
                **compute_classification_accuracy(sim, img_labels),
            }
            print(f"  [class] {ds_path.name}: Acc@1={class_results[ds_path.name]['Acc@1']*100:.2f}%")

    if cap_dir.exists():
        for parquet_path in tqdm.tqdm(sorted(cap_dir.glob("*.parquet"))):
            dataset = ParquetCaptionRaw(str(parquet_path))
            img_embs, captions = [], []
            for imgs, caps in make_loader(dataset):
                img_embs.append(adapter.encode_images(imgs))
                captions.extend(caps)
            img_embs = np.concatenate(img_embs, axis=0)

            # encode captions in batches (avoids OOM / very long padded batches)
            txt_embs = np.concatenate(
                [adapter.encode_texts(captions[i:i + 256]) for i in range(0, len(captions), 256)], axis=0)

            sim = txt_embs @ img_embs.T
            r = compute_retrieval_metrics(sim, is_1to1=True)
            cap_results[parquet_path.stem] = {
                "num_images": len(dataset), "T2I_R@1": r["T2I_R@1"], "I2T_R@1": r["I2T_R@1"],
            }
            print(f"  [cap] {parquet_path.stem}: T2I R@1={r['T2I_R@1']*100:.2f}% I2T R@1={r['I2T_R@1']*100:.2f}%")

    return class_results, cap_results


# ==========================================
# Comparison Dashboard HTML Generator
# ==========================================

def generate_html_comparison(all_model_results: Dict[str, Tuple[Dict, Dict]], output_path: str):
    model_names = list(all_model_results.keys())
    class_ds_names = sorted({d for c, _ in all_model_results.values() for d in c})
    cap_ds_names = sorted({d for _, c in all_model_results.values() for d in c})

    def fmt(res, key):
        return f"{res[key]*100:.2f}%" if key in res else "–"

    class_header = "<th>Dataset</th>" + "".join(f"<th colspan='3' class='model-group'>{m}</th>" for m in model_names)
    class_sub = "<th></th>" + "<th>Acc@1</th><th>T2I R@1</th><th>I2T R@1</th>" * len(model_names)
    class_rows = ""
    for ds in class_ds_names:
        class_rows += f"<tr><td class='ds-name'>{ds}</td>"
        for m in model_names:
            res = all_model_results[m][0].get(ds, {})
            class_rows += f"<td>{fmt(res,'Acc@1')}</td><td>{fmt(res,'T2I_R@1')}</td><td>{fmt(res,'I2T_R@1')}</td>"
        class_rows += "</tr>"

    cap_header = "<th>Dataset</th>" + "".join(f"<th colspan='2' class='model-group'>{m}</th>" for m in model_names)
    cap_sub = "<th></th>" + "<th>T2I R@1</th><th>I2T R@1</th>" * len(model_names)
    cap_rows = ""
    for ds in cap_ds_names:
        cap_rows += f"<tr><td class='ds-name'>{ds}</td>"
        for m in model_names:
            res = all_model_results[m][1].get(ds, {})
            cap_rows += f"<td>{fmt(res,'T2I_R@1')}</td><td>{fmt(res,'I2T_R@1')}</td>"
        cap_rows += "</tr>"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SOTA Remote Sensing VLM Benchmark Leaderboard</title>
<style>
body {{ font-family: system-ui, 'Inter', sans-serif; background:#f8fafc; color:#0f172a; margin:0; padding:40px 20px; }}
.container {{ max-width:1400px; margin:0 auto; }}
.header {{ background:#fff; padding:24px 32px; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:32px; }}
.header h1 {{ margin:0 0 8px; font-size:26px; }}
.section-title {{ font-size:20px; font-weight:600; margin:24px 0 16px; }}
.card {{ background:#fff; border-radius:12px; border:1px solid #e2e8f0; overflow-x:auto; margin-bottom:32px; }}
table {{ width:100%; border-collapse:collapse; text-align:center; font-size:14px; }}
th {{ background:#f1f5f9; padding:12px 14px; font-weight:600; color:#475569; border-bottom:1px solid #e2e8f0; }}
th.model-group {{ background:#e0e7ff; color:#3730a3; border-left:1px solid #cbd5e1; border-right:1px solid #cbd5e1; }}
td {{ padding:12px 14px; border-bottom:1px solid #e2e8f0; }}
.ds-name {{ font-weight:600; text-align:left; background:#fafafa; }}
tr:hover td {{ background:#f8fafc; }}
</style></head><body><div class="container">
<div class="header"><h1>SOTA Remote Sensing VLM Leaderboard</h1>
<p>Zero-shot classification and caption retrieval</p></div>
<div class="section-title">Image-Class Suite</div>
<div class="card"><table><thead><tr>{class_header}</tr><tr>{class_sub}</tr></thead><tbody>{class_rows}</tbody></table></div>
<div class="section-title">Image-Caption Suite</div>
<div class="card"><table><thead><tr>{cap_header}</tr><tr>{cap_sub}</tr></thead><tbody>{cap_rows}</tbody></table></div>
</div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nLeaderboard report saved to '{output_path}'")


# ==========================================
# Main Execution Pipeline
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple SOTA Remote Sensing VLMs")
    parser.add_argument("--benchmark_root", type=str, default="/data/susanket/vlm/eval")
    parser.add_argument("--models", nargs="+", default=["gitrsclip", "georsclip", "skyclip", "remoteclip"],
                        choices=["gitrsclip", "georsclip", "skyclip", "remoteclip", "farslip"])
    parser.add_argument("--georsclip_ckpt", type=str, default="/data/susanket/vlm/sota/RS5M_ViT-B-32.pt")
    parser.add_argument("--skyclip_ckpt", type=str, default="/data/susanket/vlm/sota/SkyCLIP-ViT-L-14.pth")
    parser.add_argument("--remoteclip_ckpt", type=str, default=None, help="auto-downloads from HF if omitted")
    parser.add_argument("--farclip_ckpt", type=str, default=None, help="auto-downloads FarSLIP2_ViT-B-16 if omitted")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_html", type=str, default="sota_leaderboard.html")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_root = Path(args.benchmark_root)
    all_model_results = {}

    for model_key in args.models:
        print(f"\n==========================================\n Loading SOTA Model: {model_key.upper()}\n"
              f"==========================================")
        adapter = None
        try:
            if model_key == "gitrsclip":
                adapter = GitRSCLIPAdapter(device)
            elif model_key == "georsclip":
                adapter = build_georsclip(args.georsclip_ckpt, device)
            elif model_key == "skyclip":
                adapter = build_skyclip(args.skyclip_ckpt, device)
            elif model_key == "remoteclip":
                adapter = build_remoteclip(args.remoteclip_ckpt, device)
            elif model_key == "farclip":
                adapter = build_farclip(args.farclip_ckpt, device)

            t0 = time.time()
            class_res, cap_res = run_evaluation(adapter, benchmark_root, args.batch_size, args.num_workers)
            print(f"Completed {model_key.upper()} evaluation in {time.time()-t0:.1f}s")
            all_model_results[model_key.upper()] = (class_res, cap_res)
        except Exception as e:
            print(f"Error evaluating model '{model_key}': {e}")
            traceback.print_exc()
        finally:
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_model_results:
        generate_html_comparison(all_model_results, args.output_html)


if __name__ == "__main__":
    main()