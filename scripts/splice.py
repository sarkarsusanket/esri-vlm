"""
SpLiCE (Sparse Linear Concept Embeddings) on top of GeoRSCLIP -- training free.

Reads a parquet of image bytes, embeds them with GeoRSCLIP (OpenCLIP ViT-B-32 + RS5M ckpt),
and decomposes every image embedding into a sparse, non-negative combination of text-concept
embeddings (Bhalla, Oesterling et al., NeurIPS 2024).

Pipeline (paper Sec. 4):
  1. concept dictionary  : text-encode a vocabulary of 1/2-word concepts, L2-normalise,
                           subtract the concept mean, L2-normalise  ->  C  (c x d)
  2. image embedding     : encode, L2-normalise, subtract image mean (modality-gap fix),
                           L2-normalise  ->  z  (d)
  3. sparse solve        : min_{w>=0}  ||C^T w - z||^2 + l1_penalty * ||w||_1
                           (same objective as the official repo's ADMM / sklearn solvers;
                            solved here with batched FISTA on the GPU)
  4. reconstruction      : z_hat = norm( norm(C^T w) + image_mean )

Usage
-----
    python splice.py --parquet image.parquet --out_dir splice_out
    python splice_georsclip.py --parquet image.parquet --l1_penalty 0.35 --topk 8
    python splice_georsclip.py --parquet image.parquet --vocab_file my_remote_sensing_concepts.txt

Outputs (in --out_dir)
----------------------
    decompositions.parquet     per image: l0, recon_cosine, concepts[], weights[] (all non-zeros, sorted)
    weights_sparse.npz         scipy CSR (n_images x n_concepts) of all weights
    concepts.txt               vocabulary, column order of weights_sparse.npz
    concept_histogram.csv      dataset-level concept summary (sum / mean weight, #images)
    image_mean.pt              image mean used for centering (reuse via --image_mean)


python /home/susanket/esri-vlm/scripts/splice.py --parquet /data/susanket/vlm/images3/image-00.parquet --out_dir /data/susanket/vlm/splice/ --vocab_file /data/susanket/vlm/splice/vocab_out/vocab_unigrams.txt
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', "3")

import argparse
import hashlib
import io
import math
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ==========================================
# Configuration defaults (same as the eval script)
# ==========================================
CKPT_PATH = "/data/susanket/vlm/sota/ckpt/RS5M_ViT-B-32.pt"
MODEL_ARCH = "ViT-B-32"
IMAGE_SIZE = 224

GITHUB_DATA = "https://raw.githubusercontent.com/AI4LIFE-GROUP/SpLiCE/main/data/vocab/"
CACHE_DIR = Path.home() / ".cache" / "splice_georsclip"


# ==========================================
# 1. Model
# ==========================================
def load_georsclip(ckpt_path: str, arch: str, device: torch.device):
    import open_clip
    print(f"Initializing OpenCLIP architecture: {arch}")
    model, _, image_transform = open_clip.create_model_and_transforms(arch, pretrained=None)

    print(f"Loading checkpoint weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            print(f"         e.g. missing: {missing[:3]}")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(arch)
    print("Model and tokenizer successfully loaded!")
    return model, image_transform, tokenizer


# ==========================================
# 2. Data
# ==========================================
class ParquetImageDataset(Dataset):
    """Reads a parquet with an image-bytes column (raw bytes or HF-style {'bytes': ...} dict)."""

    def __init__(self, parquet_path: str, transform, image_col: str = "image",
                 image_size: int = IMAGE_SIZE, limit: Optional[int] = None):
        self.df = pd.read_parquet(parquet_path)
        if image_col not in self.df.columns:
            raise KeyError(f"Column '{image_col}' not in parquet. Available: {list(self.df.columns)}")
        if limit:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
        self.image_col = image_col
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _to_bytes(v):
        if isinstance(v, dict):
            v = v.get("bytes")
        return bytes(v)

    def __getitem__(self, idx):
        ok = True
        try:
            img = Image.open(io.BytesIO(self._to_bytes(self.df.iloc[idx][self.image_col]))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size))
            ok = False
        return self.transform(img), ok


@torch.no_grad()
def embed_images(model, loader, device):
    embs, oks = [], []
    for images, ok in loader:
        e = model.encode_image(images.to(device, non_blocking=True))
        embs.append(F.normalize(e.float(), dim=-1).cpu())
        oks.append(ok)
    return torch.cat(embs), torch.cat(oks).numpy()


# ==========================================
# 3. Concept dictionary
# ==========================================
def _download(fname: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / fname
    if not target.exists():
        print(f"Downloading {fname} from the official SpLiCE repo ...")
        urllib.request.urlretrieve(GITHUB_DATA + fname, target)
    return target


def _read_lines(path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_vocabulary(vocab_file: Optional[str], n_uni: int, n_bi: int) -> List[str]:
    """
    Default = paper vocabulary: top-`n_uni` LAION unigrams + top-`n_bi` LAION bigrams
    (official files are sorted least -> most frequent, so we take the *last* N lines).
    A custom --vocab_file (one concept per line) replaces it entirely.
    """
    if vocab_file:
        vocab = _read_lines(vocab_file)
    else:
        uni = _read_lines(_download("laion.txt"))[-n_uni:]
        bi = _read_lines(_download("laion_bigrams.txt"))[-n_bi:]
        vocab = uni + bi
    vocab = list(dict.fromkeys(vocab))  # dedupe, keep order
    print(f"Concept vocabulary: {len(vocab):,} concepts")
    return vocab


@torch.no_grad()
def build_dictionary(model, tokenizer, vocab: List[str], device, cache_key: str, batch_size: int = 512):
    """Returns C (c x d): normalise -> subtract concept mean -> normalise. Cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(("\n".join(vocab) + cache_key).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"concepts_{h}.pt"
    if cache.exists():
        print(f"Loading cached concept embeddings: {cache}")
        return torch.load(cache, map_location="cpu")

    print("Encoding concept vocabulary with the text tower ...")
    embs = []
    for i in range(0, len(vocab), batch_size):
        tok = tokenizer(vocab[i:i + batch_size]).to(device)
        embs.append(F.normalize(model.encode_text(tok).float(), dim=-1).cpu())
    C = torch.cat(embs)
    C = F.normalize(C - C.mean(dim=0, keepdim=True), dim=-1)
    torch.save(C, cache)
    return C


# ==========================================
# 4. Sparse non-negative solver (batched FISTA on GPU)
# ==========================================
@torch.no_grad()
def splice_solve(C: torch.Tensor, Z: torch.Tensor, l1_penalty: float,
                 max_iter: int = 2000, tol: float = 1e-5) -> torch.Tensor:
    """
    min_{W >= 0}  sum_b ||W_b C - Z_b||^2 + l1_penalty * ||W_b||_1       (same as official repo)
    C : (c, d) dictionary rows,  Z : (B, d) centered image embeddings  ->  W : (B, c)
    """
    B, c = Z.shape[0], C.shape[0]
    # Lipschitz constant of the gradient: 2 * sigma_max(C)^2 = 2 * lambda_max(C C^T)
    L = 2.0 * torch.linalg.eigvalsh(C @ C.T).max().item()
    step = 1.0 / L
    thresh = step * l1_penalty

    W = torch.zeros(B, c, device=C.device, dtype=C.dtype)
    Y = W.clone()
    t = 1.0
    for it in range(max_iter):
        grad = 2.0 * ((Y @ C - Z) @ C.T)
        W_new = torch.relu(Y - step * grad - thresh)          # prox of l1 + non-negativity
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        Y = W_new + ((t - 1.0) / t_new) * (W_new - W)
        delta = (W_new - W).abs().max().item() if it % 10 == 0 else None
        W, t = W_new, t_new
        if delta is not None and delta < tol:
            break
    return W


def recompose(W: torch.Tensor, C: torch.Tensor, image_mean: torch.Tensor) -> torch.Tensor:
    r = F.normalize(W @ C, dim=-1)
    return F.normalize(r + image_mean, dim=-1)


# ==========================================
# 5. Main
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="Training-free SpLiCE decomposition with GeoRSCLIP")
    ap.add_argument("--parquet", default="image.parquet")
    ap.add_argument("--image_col", default="image_bytes")
    ap.add_argument("--keep_cols", nargs="*", default=[], help="extra parquet columns to copy to the output (e.g. id)")
    ap.add_argument("--out_dir", default="splice_out")
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--arch", default=MODEL_ARCH)
    ap.add_argument("--batch_size", type=int, default=64, help="CLIP image batch size")
    ap.add_argument("--solve_batch", type=int, default=1024, help="images per sparse-solve batch")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only process first N rows (debug)")
    ap.add_argument("--l1_penalty", type=float, default=0.25,
                    help="higher = sparser. Paper/repo use 0.15-0.35 for ~5-30 concepts per image")
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--n_unigrams", type=int, default=10000)
    ap.add_argument("--n_bigrams", type=int, default=5000)
    ap.add_argument("--vocab_file", default=None, help="custom vocabulary, one concept per line (replaces LAION vocab)")
    ap.add_argument("--image_mean", default=None,
                    help="path to a saved image_mean.pt. Default: estimate from the input images")
    ap.add_argument("--topk", type=int, default=10, help="concepts to print / show in the histogram summary")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, image_transform, tokenizer = load_georsclip(args.ckpt, args.arch, device)

    # ---- concept dictionary ----
    vocab = load_vocabulary(args.vocab_file, args.n_unigrams, args.n_bigrams)
    C = build_dictionary(model, tokenizer, vocab, device, cache_key=f"{args.arch}|{args.ckpt}").to(device)
    print(f"Dictionary: {tuple(C.shape)}")

    # ---- image embeddings ----
    ds = ParquetImageDataset(args.parquet, image_transform, args.image_col, IMAGE_SIZE, args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    t0 = time.time()
    img_emb, ok = embed_images(model, loader, device)          # (N, d), L2-normalised, CPU
    print(f"Embedded {len(ds):,} images in {time.time() - t0:.1f}s "
          f"({int((~ok).sum())} failed to decode)")

    # ---- modality-gap centering ----
    if args.image_mean:
        image_mean = torch.load(args.image_mean, map_location="cpu").float()
        print(f"Loaded image mean from {args.image_mean}")
    else:
        image_mean = img_emb[torch.from_numpy(ok)].mean(dim=0)
        print("Estimated image mean from the input images "
              "(save/reuse image_mean.pt for consistent decompositions across runs)")
    torch.save(image_mean, out / "image_mean.pt")
    image_mean = image_mean.to(device)

    # ---- sparse decomposition ----
    t0 = time.time()
    all_W, all_cos = [], []
    for i in range(0, len(img_emb), args.solve_batch):
        z = img_emb[i:i + args.solve_batch].to(device)
        zc = F.normalize(z - image_mean, dim=-1)
        W = splice_solve(C, zc, args.l1_penalty, args.max_iter, args.tol)
        z_hat = recompose(W, C, image_mean)
        # empty decompositions (all-zero W) -> recompose gives the mean direction; cosine still defined
        all_cos.append((z_hat * z).sum(-1).cpu())
        all_W.append(W.cpu())
        print(f"  solved {min(i + args.solve_batch, len(img_emb)):,}/{len(img_emb):,}", end="\r", flush=True)
    W = torch.cat(all_W)
    cos = torch.cat(all_cos).numpy()
    print(f"\nSolved in {time.time() - t0:.1f}s")

    # ---- save ----
    import scipy.sparse as sp
    W_np = W.numpy()
    sp.save_npz(out / "weights_sparse.npz", sp.csr_matrix(W_np))
    (out / "concepts.txt").write_text("\n".join(vocab), encoding="utf-8")

    concepts_col, weights_col, l0 = [], [], []
    for row in W_np:
        nz = np.nonzero(row)[0]
        nz = nz[np.argsort(-row[nz])]
        concepts_col.append([vocab[j] for j in nz])
        weights_col.append(row[nz].astype(np.float32).tolist())
        l0.append(len(nz))
    res = pd.DataFrame({"row_idx": np.arange(len(W_np)), "decode_ok": ok,
                        "l0": l0, "recon_cosine": cos,
                        "concepts": concepts_col, "weights": weights_col})
    for col in args.keep_cols:
        res[col] = ds.df[col].values
    res.to_parquet(out / "decompositions.parquet", index=False)

    # dataset-level histogram
    tot = W_np.sum(0)
    cnt = (W_np > 0).sum(0)
    hist = pd.DataFrame({"concept": vocab, "total_weight": tot,
                         "mean_weight": tot / max(len(W_np), 1), "n_images": cnt})
    hist = hist[hist.n_images > 0].sort_values("total_weight", ascending=False)
    hist.to_csv(out / "concept_histogram.csv", index=False)

    # ---- report ----
    l0a = np.array(l0)
    print("\n" + "=" * 70)
    print(f"images: {len(W_np):,} | mean L0: {l0a.mean():.1f} (median {np.median(l0a):.0f}, "
          f"min {l0a.min()}, max {l0a.max()}) | zero-concept images: {(l0a == 0).sum()}")
    print(f"mean cosine(SpLiCE recon, original CLIP): {cos.mean():.3f}  "
          f"(low is expected -- see paper Sec 5.2; it only keeps semantics)")
    print(f"\nTop {args.topk} dataset-level concepts:")
    for _, r in hist.head(args.topk).iterrows():
        print(f"  {r.concept:<28s} total={r.total_weight:8.2f}  in {int(r.n_images):,} images")
    print("\nFirst few decompositions:")
    for i in range(min(5, len(res))):
        pairs = ", ".join(f"{c} ({w:.3f})" for c, w in zip(concepts_col[i][:args.topk], weights_col[i][:args.topk]))
        print(f"  [{i}] L0={l0[i]:>3d}  {pairs}")
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()