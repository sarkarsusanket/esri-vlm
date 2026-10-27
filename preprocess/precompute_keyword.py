#!/usr/bin/env python3
"""Pre-compute Qwen3-Embedding EOS states for every unique caption keyword.

Scans every ``caption-*.parquet`` under ``data_dir``, collects the individual strings in
``key_elements``, de-duplicates them, and writes one compact lookup artifact::

    keyword_eos/
        keywords.npy    (N, hidden_dim) fp16   row i = EOS state of keywords.json[i]
        keywords.json   [keyword, ...]         sorted; row order is stable
        metadata.json   model / dtype / counts / source shards

The single matrix + JSON index is intentionally chosen over one file per keyword: it is a few
hundred MB, memory-mappable, and a training-time lookup is just ``index[keyword] -> row``.

These are raw last-layer EOS hidden states (same convention as ``precompute_eos.py``), not
projected features. In training they go through ``model.encode_text_emb`` exactly like the
summary / dense caption states. ``src/data.py`` samples keyword combinations at runtime and
represents each one by the sum of its individual keyword rows, so no Qwen forward pass is needed.
Use ``--normalize`` only if downstream code expects unit vectors (it changes that sum).

Usage::

    python preprocess/precompute_keyword.py
    python preprocess/precompute_keyword.py /data/susanket/vlm/images3 --device cuda:2 --overwrite
    python preprocess/precompute_keyword.py --shards 00,01 --device cuda:2
    python preprocess/precompute_keyword.py --output_dir /data/susanket/vlm/images3/keyword_eos

During training::

    from preprocess.precompute_keyword import KeywordEmbeddings
    kw = KeywordEmbeddings("/data/susanket/vlm/images3/keyword_eos")
    emb = kw["trees"]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_DATA_DIR = "/data/susanket/vlm/images3"
OUTPUT_SUBDIR = "keyword_eos"


class KeywordEmbeddings:
    """Memory-mapped keyword -> EOS-state lookup.

    The index is loaded eagerly (a plain dict) and the matrix is memory-mapped, so a training
    process pays only for the keywords it actually touches.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        with open(self.root / "keywords.json", "r", encoding="utf-8") as f:
            self.keywords: List[str] = json.load(f)
        self.embeddings = np.load(self.root / "keywords.npy", mmap_mode="r")
        if len(self.keywords) != len(self.embeddings):
            raise ValueError(f"{self.root}: {len(self.keywords)} keywords but "
                             f"{len(self.embeddings)} embedding rows")
        self.index = {k: i for i, k in enumerate(self.keywords)}

    def __len__(self) -> int:
        return len(self.keywords)

    def __contains__(self, keyword: object) -> bool:
        return keyword in self.index

    def __getitem__(self, keyword: str) -> np.ndarray:
        return self.embeddings[self.index[keyword]]

    def get(self, keyword: str, default=None):
        i = self.index.get(keyword)
        return default if i is None else self.embeddings[i]

    def rows(self, keywords: Iterable[str]) -> np.ndarray:
        """Rows for the requested keywords, preserving order and dropping unknown ones."""
        idx = [self.index[k] for k in keywords if k in self.index]
        return np.asarray(self.embeddings[idx])


def _keywords_from_cell(cell) -> Iterable[str]:
    if cell is None:
        return ()
    if isinstance(cell, (list, tuple)):
        return cell
    if hasattr(cell, "tolist"):
        return cell.tolist()
    if isinstance(cell, str):
        return (cell,)
    return ()


def scan_file(path: Path, batch_size: int) -> set[str]:
    pf = pq.ParquetFile(path)
    if "key_elements" not in pf.schema_arrow.names:
        logging.warning("%s: no 'key_elements' column, skipping", path.name)
        return set()
    found: set[str] = set()
    with tqdm(total=pf.metadata.num_rows, desc=path.name, unit=" rows") as pbar:
        for batch in pf.iter_batches(batch_size=batch_size, columns=["key_elements"]):
            for cell in batch.column(0).to_pylist():
                for keyword in _keywords_from_cell(cell):
                    if not isinstance(keyword, str):
                        continue
                    keyword = keyword.strip()
                    if keyword:
                        found.add(keyword)
            pbar.update(batch.num_rows)
    return found


def collect_keywords(files: Sequence[Path], batch_size: int) -> List[str]:
    keywords: set[str] = set()
    for path in files:
        before = len(keywords)
        keywords.update(scan_file(path, batch_size))
        logging.info("%s: +%d new keywords (%d total)", path.name, len(keywords) - before,
                     len(keywords))
    return sorted(keywords)


def _atomic_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def choose_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free = []
        for i in range(torch.cuda.device_count()):
            try:
                free.append((torch.cuda.mem_get_info(i)[0], i))
            except Exception:
                pass
        if free:
            return f"cuda:{max(free)[1]}"
    except Exception:
        pass
    return "cpu"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR,
                    help=f"Directory containing caption-XX.parquet (default: {DEFAULT_DATA_DIR}).")
    ap.add_argument("--output_dir", default=None,
                    help=f"Artifact directory (default: <data_dir>/{OUTPUT_SUBDIR}).")
    ap.add_argument("--pattern", default="caption-*.parquet")
    ap.add_argument("--shards", default=None,
                    help="Optional comma list of shard ids, e.g. 00,01 (default: all matches).")
    ap.add_argument("--scan_batch_size", type=int, default=200_000,
                    help="Parquet rows read per batch while collecting keywords.")
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--model_dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 | eager")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="Storage dtype of keywords.npy.")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize the EOS states (default: raw hidden state).")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_tokens", type=int, default=65_536,
                    help="Padded-token budget per GPU batch.")
    ap.add_argument("--max_bs", type=int, default=512)
    ap.add_argument("--device", default=None,
                    help="Torch device, e.g. cuda:2 or cpu (default: CUDA with most free memory).")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args(argv)


def main() -> None:
    a = parse_args()
    data_dir = Path(a.data_dir)
    out_dir = Path(a.output_dir) if a.output_dir else data_dir / OUTPUT_SUBDIR

    files = sorted(p for p in data_dir.glob(a.pattern) if p.is_file() and not p.name.startswith("."))
    if a.shards:
        wanted = {s.strip() for s in a.shards.split(",") if s.strip()}
        files = [p for p in files if p.stem.split("-")[-1] in wanted]
    if not files:
        raise SystemExit(f"no files matching {a.pattern!r} in {data_dir}")

    matrix_path = out_dir / "keywords.npy"
    index_path = out_dir / "keywords.json"
    if (matrix_path.exists() or index_path.exists()) and not a.overwrite:
        raise SystemExit(f"{out_dir} already contains keyword embeddings; pass --overwrite to rebuild")

    logging.info("scanning %d caption file(s) in %s", len(files), data_dir)
    keywords = collect_keywords(files, a.scan_batch_size)
    if not keywords:
        raise SystemExit("no keywords found")
    logging.info("collected %d unique keywords", len(keywords))

    device = choose_device(a.device)
    logging.info("embedding on %s (%s, normalize=%s)", device, a.model_dtype, a.normalize)
    if device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(torch.cuda.device_count()))
    try:
        from .precompute_eos import Embedder
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from precompute_eos import Embedder

    emb = Embedder(a.model_name, device, a.model_dtype, a.attn,
                   a.max_length, a.max_tokens, a.max_bs, a.normalize)
    ids = emb.tokenize(keywords)
    batches = emb.plan_batches(ids)
    vec = emb.embed(ids, batches)
    storage_dtype = np.float16 if a.dtype == "float16" else np.float32

    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npy(matrix_path, vec.astype(storage_dtype))
    _atomic_json(index_path, keywords)
    _atomic_json(out_dir / "metadata.json", {
        "model_name": a.model_name,
        "hidden_dim": int(emb.dim),
        "storage_dtype": a.dtype,
        "normalize": bool(a.normalize),
        "count": len(keywords),
        "eos_token_id": int(emb.eos),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p) for p in files],
    })
    logging.info("wrote %d x %d embeddings -> %s", len(keywords), emb.dim, out_dir)


if __name__ == "__main__":
    main()
