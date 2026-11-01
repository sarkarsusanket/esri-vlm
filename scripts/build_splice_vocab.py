"""
Build a SpLiCE concept vocabulary from your own captions, following Sec. 4.1 of the paper:

  1. count the most frequent 1-word and 2-word n-grams in the captions
  2. (optional) drop captions containing blocklisted (NSFW) words
  3. drop bigrams whose embedding is too similar (cos > 0.9) to the average of the embeddings
     of their two individual words (i.e. bigrams that add nothing over their words)
  4. prune so that no two concept embeddings have cosine similarity > 0.9
     (greedy, most frequent concept wins)
  5. keep the top N unigrams (default 10,000) and top M bigrams (default 5,000)

Embeddings come from the GeoRSCLIP text tower, so the pruning reflects *your* model's geometry.

    python scripts/build_splice_vocab.py --caption_dir /data/susanket/vlm/images3/ --out_dir vocab_out
    python scripts/build_splice_vocab.py --caption_dir /data/susanket/vlm/images3 --glob "caption-[0-9][0-9].parquet" --counts_only

Outputs (--out_dir):
    vocab.txt             unigrams + bigrams, ready for:  splice_georsclip.py --vocab_file vocab_out/vocab.txt
    vocab_unigrams.txt    least -> most frequent (same ordering convention as the official LAION files)
    vocab_bigrams.txt     same
    vocab_counts.csv      every candidate with count and why it was dropped (if it was)
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Compact English stopword list (no nltk dependency) + words that are noise in VLM-written captions.
STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our
out over own same she should so some such than that the their them then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your
""".split())
CAPTION_NOISE = set("""
image picture photo photograph shows show shown showing appears appear seen visible view depicts depicted
depicting featuring features feature contains contain contained located present area scene
""".split())

TOKEN_RE = re.compile(r"[a-z]{2,}")
SPLIT_RE = re.compile(r"[.,;:!?()\[\]\"\n\r/\\|]+")


# --------------------------------------------------------------------------
# 1. Counting
# --------------------------------------------------------------------------
def iter_captions(files: List[Path], col: str, max_captions: Optional[int]):
    import pyarrow.parquet as pq
    n = 0
    for f in files:
        tbl = pq.read_table(f, columns=[col])
        for cap in tbl.column(col).to_pylist():
            if isinstance(cap, str) and cap.strip():
                yield cap
                n += 1
                if max_captions and n >= max_captions:
                    return


def count_ngrams(captions, stop: Set[str], blocklist: Set[str], doc_freq: bool) -> Tuple[Counter, Counter, int, int]:
    uni, bi = Counter(), Counter()
    n_caps = n_blocked = 0
    for cap in captions:
        text = cap.lower()
        if blocklist and any(w in blocklist for w in TOKEN_RE.findall(text)):
            n_blocked += 1
            continue
        n_caps += 1
        u_seen, b_seen = set(), set()
        for seg in SPLIT_RE.split(text):          # n-grams never span punctuation
            toks = TOKEN_RE.findall(seg)
            for t in toks:
                if t not in stop:
                    (u_seen.add(t) if doc_freq else uni.update([t]))
            for a, b in zip(toks, toks[1:]):
                if a not in stop and b not in stop and a != b:
                    g = f"{a} {b}"
                    (b_seen.add(g) if doc_freq else bi.update([g]))
        if doc_freq:
            uni.update(u_seen)
            bi.update(b_seen)
    return uni, bi, n_caps, n_blocked


# --------------------------------------------------------------------------
# 2. Embedding + filtering (numpy; embeddings are small)
# --------------------------------------------------------------------------
def encode_texts(model, tokenizer, texts: List[str], device, batch_size: int = 1024) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tok = tokenizer(texts[i:i + batch_size]).to(device)
            out.append(F.normalize(model.encode_text(tok).float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 1), np.float32)


def greedy_prune(emb: np.ndarray, thresh: float, preselected: Optional[np.ndarray] = None,
                 chunk: int = 2048) -> np.ndarray:
    """
    emb rows are L2-normalised and ordered by priority (most frequent first).
    Keeps a row iff its cosine to every previously kept row (and to `preselected`) is <= thresh.
    Returns a boolean keep mask.
    """
    n, d = emb.shape
    cap = n + (0 if preselected is None else len(preselected))
    buf = np.empty((cap, d), np.float32)
    nk = 0
    if preselected is not None and len(preselected):
        buf[:len(preselected)] = preselected
        nk = len(preselected)
    keep = np.zeros(n, bool)
    for s in range(0, n, chunk):
        blk = emb[s:s + chunk]
        alive = np.ones(len(blk), bool)
        if nk:
            alive &= (blk @ buf[:nk].T).max(axis=1) <= thresh
        idx = np.nonzero(alive)[0]
        if len(idx):
            sub = blk[idx]
            ss = sub @ sub.T
            ok = np.ones(len(idx), bool)
            for a in range(len(idx)):             # greedy within the chunk, in priority order
                if ok[a]:
                    ok[a + 1:] &= ss[a, a + 1:] <= thresh
            sel = idx[ok]
            buf[nk:nk + len(sel)] = blk[sel]
            nk += len(sel)
            keep[s + sel] = True
    return keep


def bigram_redundant(bigrams: List[str], bi_emb: np.ndarray, word_emb: Dict[str, np.ndarray],
                     thresh: float) -> np.ndarray:
    """True where cos(bigram, normalize(mean(word1, word2))) > thresh."""
    avg = np.stack([word_emb[b.split()[0]] + word_emb[b.split()[1]] for b in bigrams])
    avg /= np.linalg.norm(avg, axis=1, keepdims=True) + 1e-12
    return (bi_emb * avg).sum(axis=1) > thresh


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build a SpLiCE vocabulary from dense captions (paper Sec. 4.1)")
    ap.add_argument("--caption_dir", required=True)
    ap.add_argument("--glob", default="*.parquet", help='e.g. "caption-*.parquet"')
    ap.add_argument("--caption_col", default="dense_caption")
    ap.add_argument("--out_dir", default="vocab_out")
    ap.add_argument("--n_unigrams", type=int, default=10000)
    ap.add_argument("--n_bigrams", type=int, default=5000)
    ap.add_argument("--oversample", type=float, default=3.0,
                    help="candidates = oversample x target before the similarity filters")
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--sim_thresh", type=float, default=0.9, help="paper: 0.9")
    ap.add_argument("--doc_freq", action="store_true", help="count #captions containing the n-gram instead of raw occurrences")
    ap.add_argument("--keep_stopwords", action="store_true", help="do not filter stopwords / caption boilerplate words")
    ap.add_argument("--extra_stopwords", default=None, help="file, one word per line, added to the stopword list")
    ap.add_argument("--blocklist", default=None, help="NSFW word list; captions containing any of these words are skipped")
    ap.add_argument("--max_captions", type=int, default=None)
    ap.add_argument("--counts_only", action="store_true", help="only count n-grams (no model needed)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()

    files = sorted(Path(args.caption_dir).glob(args.glob))
    if not files:
        raise SystemExit(f"No files matching {args.glob} in {args.caption_dir}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} parquet files")

    stop = set() if args.keep_stopwords else STOPWORDS | CAPTION_NOISE
    if args.extra_stopwords:
        stop |= {l.strip().lower() for l in open(args.extra_stopwords, encoding="utf-8") if l.strip()}
    blocklist = {l.strip().lower() for l in open(args.blocklist, encoding="utf-8") if l.strip()} if args.blocklist else set()

    # ---- 1. count ----
    uni_c, bi_c, n_caps, n_blocked = count_ngrams(
        iter_captions(files, args.caption_col, args.max_captions), stop, blocklist, args.doc_freq)
    print(f"Counted {n_caps:,} captions ({n_blocked:,} skipped by blocklist) | "
          f"{len(uni_c):,} unique unigrams, {len(bi_c):,} unique bigrams")

    n_uc, n_bc = int(args.n_unigrams * args.oversample), int(args.n_bigrams * args.oversample)
    uni_cand = [(w, c) for w, c in uni_c.most_common(n_uc) if c >= args.min_count]
    bi_cand = [(w, c) for w, c in bi_c.most_common(n_bc) if c >= args.min_count]
    print(f"Candidates (count >= {args.min_count}): {len(uni_cand):,} unigrams, {len(bi_cand):,} bigrams")

    rows = [{"concept": w, "type": "unigram", "count": c, "status": "candidate"} for w, c in uni_cand] + \
           [{"concept": w, "type": "bigram", "count": c, "status": "candidate"} for w, c in bi_cand]
    status = {(r["type"], r["concept"]): "kept" for r in rows}     # updated below

    if args.counts_only:
        pd.DataFrame(rows).to_csv(out / "vocab_counts.csv", index=False)
        (out / "vocab_unigrams.txt").write_text("\n".join(w for w, _ in uni_cand[:args.n_unigrams][::-1]), encoding="utf-8")
        (out / "vocab_bigrams.txt").write_text("\n".join(w for w, _ in bi_cand[:args.n_bigrams][::-1]), encoding="utf-8")
        print(f"--counts_only: wrote raw frequency lists to {out} (no similarity filtering)")
        return

    # ---- 2. embeddings with the GeoRSCLIP text tower ----
    import torch
    import splice_georsclip as sg          # reuses the model loader from the main script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, tokenizer = sg.load_georsclip(args.ckpt or sg.CKPT_PATH, args.arch or sg.MODEL_ARCH, device)

    uni_words = [w for w, _ in uni_cand]
    bi_words = [w for w, _ in bi_cand]
    need_words = sorted({t for b in bi_words for t in b.split()} - set(uni_words))   # words only appearing inside bigrams
    print(f"Encoding {len(uni_words):,} unigrams, {len(bi_words):,} bigrams, {len(need_words):,} extra component words ...")
    uni_emb = encode_texts(model, tokenizer, uni_words, device)
    bi_emb = encode_texts(model, tokenizer, bi_words, device)
    extra_emb = encode_texts(model, tokenizer, need_words, device) if need_words else np.zeros((0, uni_emb.shape[1]), np.float32)
    word_emb = {w: e for w, e in zip(uni_words, uni_emb)}
    word_emb.update({w: e for w, e in zip(need_words, extra_emb)})

    # ---- 3. drop bigrams that are just the average of their words ----
    if len(bi_words):
        red = bigram_redundant(bi_words, bi_emb, word_emb, args.sim_thresh)
        for b, r in zip(bi_words, red):
            if r:
                status[("bigram", b)] = "dropped: ~average of its words"
        print(f"Bigrams redundant w.r.t. their words (cos > {args.sim_thresh}): {int(red.sum()):,}")
        bi_words = [b for b, r in zip(bi_words, red) if not r]
        bi_emb = bi_emb[~red]

    # ---- 4. prune near-duplicates (most frequent wins): unigrams first, then bigrams vs. kept unigrams ----
    ku = greedy_prune(uni_emb, args.sim_thresh)
    for w, k in zip(uni_words, ku):
        if not k:
            status[("unigram", w)] = f"dropped: cos > {args.sim_thresh} to a more frequent concept"
    kb = greedy_prune(bi_emb, args.sim_thresh, preselected=uni_emb[ku]) if len(bi_words) else np.zeros(0, bool)
    for w, k in zip(bi_words, kb):
        if not k:
            status[("bigram", w)] = f"dropped: cos > {args.sim_thresh} to a more frequent concept"
    print(f"After pruning: {int(ku.sum()):,} unigrams, {int(kb.sum()):,} bigrams")

    # ---- 5. top-N / top-M ----
    final_uni = [w for w, k in zip(uni_words, ku) if k][:args.n_unigrams]        # already sorted by count desc
    final_bi = [w for w, k in zip(bi_words, kb) if k][:args.n_bigrams]
    keep_uni_set = set(final_uni)
    for w in uni_words:
        if status[("unigram", w)] == "kept" and w not in keep_uni_set:
            status[("unigram", w)] = "dropped: beyond top-N"
    keep_bi_set = set(final_bi)
    for w in bi_words:
        if status[("bigram", w)] == "kept" and w not in keep_bi_set:
            status[("bigram", w)] = "dropped: beyond top-M"
    if len(final_uni) < args.n_unigrams:
        print(f"[warn] only {len(final_uni):,} unigrams survived (target {args.n_unigrams:,}); "
              f"raise --oversample, lower --min_count, or add more captions")
    if len(final_bi) < args.n_bigrams:
        print(f"[warn] only {len(final_bi):,} bigrams survived (target {args.n_bigrams:,})")

    # ---- write (least -> most frequent, like the official LAION files) ----
    (out / "vocab_unigrams.txt").write_text("\n".join(final_uni[::-1]), encoding="utf-8")
    (out / "vocab_bigrams.txt").write_text("\n".join(final_bi[::-1]), encoding="utf-8")
    (out / "vocab.txt").write_text("\n".join(final_uni[::-1] + final_bi[::-1]), encoding="utf-8")
    for r in rows:
        r["status"] = status[(r["type"], r["concept"])]
    pd.DataFrame(rows).to_csv(out / "vocab_counts.csv", index=False)

    print(f"\nFinal vocabulary: {len(final_uni):,} unigrams + {len(final_bi):,} bigrams -> {out / 'vocab.txt'}")
    print("Top unigrams:", ", ".join(final_uni[:15]))
    print("Top bigrams: ", ", ".join(final_bi[:10]))
    print(f"\nNext: python splice_georsclip.py --parquet image.parquet --vocab_file {out / 'vocab.txt'}")


if __name__ == "__main__":
    main()