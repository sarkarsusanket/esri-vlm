"""Translate the `summary` column of caption-xx.parquet files IN PLACE.

Fast path: NLLB-200 (distilled 600M) served by CTranslate2 (int8_float16, greedy, token-based
batching, ONE call per file for all languages via per-row target prefixes). Typically 5-10x
faster than HF `generate` and, unlike M2M100-418M, NLLB really has Traditional Chinese and
Cantonese (the old script silently emitted garbage for "yue" and plain Simplified for "zh").

One-time setup:
    pip install ctranslate2 transformers sentencepiece polars tqdm
    ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
        --output_dir /data/susanket/nllb600m-ct2 --quantization float16 --force

Run:
    python translate_captions.py --input /data/susanket/vlm/images3

In-place behaviour, per caption-*.parquet:
  * the file is read fully, the two columns `translated_col` / `translated_lang` are appended
    (null for rows that were not selected when --frac < 1), every other column and the row order
    are untouched, and the result replaces the SAME file;
  * the new data is first written to <name>.tmp, read back and checked (same row count, expected
    columns) and only then swapped in with an atomic os.replace, so a crash never leaves a
    half-written caption file;
  * files that already have `translated_col` are skipped (re-running resumes at file level and
    never re-translates); pass --force to redo them.
"""
from __future__ import annotations

import argparse
import os

# Only a default: a CUDA_VISIBLE_DEVICES set in the launch command wins.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

NEW_COLS = ("translated_col", "translated_lang")

# NLLB-200 (FLORES-200) codes
LANGUAGES = {
    "French": "fra_Latn",
    "Portuguese": "por_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Hindi": "hin_Deva",
    # "Bengali": "ben_Beng",
    # "Marathi": "mar_Deva",
    # "Tamil": "tam_Taml",
    # "Telugu": "tel_Telu",
    "Simplified Chinese": "zho_Hans",
    "Traditional Chinese": "zho_Hant",
    "Cantonese": "yue_Hant",
}


class CT2Translator:
    def __init__(self, model_dir: str, tok_name: str, device: str, compute_type: str,
                 max_batch_tokens: int, max_len: int):
        import ctranslate2
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tok_name, src_lang="eng_Latn", use_fast=True)
        self.tr = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type,
                                         inter_threads=1, intra_threads=os.cpu_count() // 2 or 1)
        self.max_batch_tokens, self.max_len = max_batch_tokens, max_len

    def __call__(self, texts: list[str], tgt_codes: list[str]) -> list[str]:
        # fast (rust, batched) tokenisation -> token strings for CT2
        ids = self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
        src = [self.tok.convert_ids_to_tokens(i) for i in ids]
        out = self.tr.translate_batch(
            src,
            target_prefix=[[c] for c in tgt_codes],   # per-row target language
            beam_size=1,                              # greedy: fastest
            max_batch_size=self.max_batch_tokens,
            batch_type="tokens",                      # CT2 length-sorts internally
            max_decoding_length=self.max_len,
            repetition_penalty=1.1,                   # greedy NLLB can loop on long inputs
            return_scores=False,
        )
        hyp_ids = [self.tok.convert_tokens_to_ids(r.hypotheses[0][1:]) for r in out]  # drop lang tag
        return self.tok.batch_decode(hyp_ids, skip_special_tokens=True)


def process_file(file: Path, col: str, frac: float, translate, rng_seed: int, force: bool = False) -> str:
    """Translate `file` in place. Returns 'done' | 'skipped' | 'no-col'."""
    schema = pl.read_parquet_schema(file)             # footer only, no data read
    if col not in schema:
        print(f"Skipping {file.name}: column '{col}' not found")
        return "no-col"
    if NEW_COLS[0] in schema and not force:
        return "skipped"                              # already translated (use --force to redo)

    df = pl.read_parquet(file)                        # fully in memory -> safe to replace the file
    texts = df[col].to_list()
    valid = np.array([isinstance(t, str) and t.strip() != "" for t in texts], dtype=bool)
    cand = np.flatnonzero(valid)
    rng = np.random.default_rng(rng_seed)
    n_sel = int(round(len(df) * frac))
    sel = np.sort(rng.choice(cand, size=min(n_sel, len(cand)), replace=False)) if len(cand) else cand

    names = list(LANGUAGES)
    lang_names = [names[i] for i in rng.integers(0, len(names), size=len(sel))]
    codes = [LANGUAGES[n] for n in lang_names]

    translated = translate([texts[i] for i in sel], codes) if len(sel) else []
    assert len(translated) == len(sel), f"translator returned {len(translated)} for {len(sel)} inputs"

    res = [None] * len(df)
    lng = [None] * len(df)
    for i, t, n in zip(sel, translated, lang_names):
        res[i], lng[i] = t, n

    out = df.drop([c for c in NEW_COLS if c in df.columns]).with_columns(
        pl.Series(NEW_COLS[0], res, dtype=pl.Utf8),
        pl.Series(NEW_COLS[1], lng, dtype=pl.Utf8),
    )

    # write -> verify -> atomic swap onto the original path
    tmp = file.with_name(file.name + ".tmp")          # never matches the caption-*.parquet glob
    out.write_parquet(tmp)
    chk = pl.read_parquet_schema(tmp)
    n_back = pl.scan_parquet(tmp).select(pl.len()).collect().item()
    if n_back != len(df) or any(c not in chk for c in (*df.columns, *NEW_COLS)):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"verification failed for {file.name}; original left untouched")
    os.replace(tmp, file)
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/susanket/vlm/sft/")
    ap.add_argument("--model-dir", default="/data/susanket/nllb600m-ct2", help="CTranslate2-converted NLLB dir")
    ap.add_argument("--tokenizer", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--col", default="summary")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--compute-type", default="int8_float16", help="float16 for slightly better quality")
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-translate files that already have translated_col")
    args = ap.parse_args()

    in_dir = Path(args.input)
    files = sorted(in_dir.glob("te*.parquet"))
    print(f"{len(files)} caption file(s) found in {in_dir}")

    translate = CT2Translator(args.model_dir, args.tokenizer, args.device, args.compute_type,
                              args.max_batch_tokens, args.max_len)
    counts: dict[str, int] = {}
    for k, f in enumerate(tqdm(files, desc="files")):
        r = process_file(f, args.col, args.frac, translate, rng_seed=args.seed * 100003 + k, force=args.force)
        counts[r] = counts.get(r, 0) + 1
    print("done", counts)


if __name__ == "__main__":
    main()