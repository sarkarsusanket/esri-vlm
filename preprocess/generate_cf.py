#!/usr/bin/env python3
"""Add a hard-negative column `cf_dense_caption` to every caption-XX.parquet in a directory (vLLM).

Every SENTENCE of each `dense_caption` gets its own edit (default: exactly one per sentence), so the
whole paragraph is hard-negated, sentence by sentence. The LLM does NOT rewrite the sentence. It returns
1..N tiny {"find": ..., "replace": ...} edits that flip a concrete, visually checkable fact
(color / count / direction / material / shape / object / presence / state / relation).
The script applies them programmatically, so:
  * output is ~30 tokens per sentence instead of a full rewrite -> much faster
  * every sentence is its own short request (more parallelism, a different random edit TYPE each)
  * the hard negative is IDENTICAL to the original except for the edited span(s)
  * edits are validated (find must be an exact substring, no synonyms/plural-only swaps,
    no added sentences, a/an repaired) and failures are retried hotter

Why it is fast (same tricks as caption_vllm.py, plus):
  * one shared system+few-shot prefix -> vLLM prefix cache; prefix tokenized ONCE
  * model loaded once for all files; continuous batching (512 seqs in flight)
  * per-file atomic write (tmp -> rename) and per-chunk progress JSONL (resumable)
  * files that already contain the column are skipped

Install:  pip install vllm pyarrow tqdm
Run:
  python cf_dense_caption_vllm.py --input_dir /data/captions
  python cf_dense_caption_vllm.py --input_dir /data/captions --dry_run 20     # inspect, writes nothing
  python cf_dense_caption_vllm.py --input_dir /data/captions --output_dir /data/captions_cf
"""
from __future__ import annotations

import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import argparse
import difflib
import json
import logging
import random
import re
import sys
import zlib
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PH = "@@USER@@"
_SENT_SPLIT = re.compile(r"((?<=[.!?])\s+)")
TYPE_WEIGHTS = {"color": 3.0, "count": 2.0, "direction": 2.0, "material": 1.5, "shape": 1.0,
                "object": 2.0, "presence": 1.5, "state": 1.0, "relation": 1.0}

SYSTEM = """You create HARD NEGATIVES for aerial / satellite image captions (contrastive training of vision-language models).
Given ONE SENTENCE from a caption, change EDITS concrete, visually checkable fact(s) in it so the sentence becomes WRONG for the image, while staying fluent and almost identical to the original.

Reply with ONLY a one-line JSON array of exactly EDITS objects: [{"find": "...", "replace": "..."}]
- "find": text copied EXACTLY from the SENTENCE (same case/punctuation), 1-8 words.
- "replace": what to put instead. Keep the sentence grammatical: include articles, plurals and verb agreement inside find/replace (e.g. "a red roof" -> "an orange roof").
- TYPE tells which kind of fact to flip. If the sentence has no such fact, use the closest other type; every sentence contains something that can be flipped.
  color (red->blue), count (two->five), direction (north->south, left->right, upper->lower), material (brick->wooden, asphalt->gravel),
  shape (rectangular->circular), object (swap for a different object of the same kind that cannot be both: pool->pond, truck->bus, tennis court->basketball court),
  presence (remove/deny an item: "a white building" -> "no building"), state (parked->moving, full->empty, open->closed), relation (adjacent to->far from, inside->outside).
- NEVER use synonyms or broader/narrower words (red->maroon, house->building, car->vehicle, gray->grey are all INVALID). No plural-only changes.
- Do not add sentences, hedging or commentary. Do not touch anything else in the sentence."""

SHOTS = [
    (1, "color", "Several blue shipping containers are stacked near the eastern fence of an industrial yard.",
     '[{"find": "blue shipping containers", "replace": "red shipping containers"}]'),
    (1, "count", "Two white cars are parked in a gray driveway beside the house.",
     '[{"find": "Two white cars", "replace": "Five white cars"}]'),
    (1, "direction", "Wooden docks extend into calm green water from the western shoreline.",
     '[{"find": "from the western shoreline", "replace": "from the eastern shoreline"}]'),
    (1, "presence", "A baseball field has a tan infield, with a white building behind the left field fence.",
     '[{"find": "with a white building behind the left field fence", "replace": "with no building behind the left field fence"}]'),
    (1, "material", "A residential complex with brick buildings surrounds a courtyard.",
     '[{"find": "brick buildings", "replace": "wooden buildings"}]'),
    (1, "shape", "Long rows of dark solar panels cover a field beside a rectangular water reservoir.",
     '[{"find": "a rectangular water reservoir", "replace": "a circular water reservoir"}]'),
    (1, "state", "The parking lot is full of vehicles.",
     '[{"find": "full of vehicles", "replace": "empty of vehicles"}]'),
    (1, "relation", "A small pond lies adjacent to the northern edge of the field.",
     '[{"find": "adjacent to the northern edge", "replace": "far from the northern edge"}]'),
    (2, "any", "Blue shipping containers are stacked near the eastern fence of a large yard.",
     '[{"find": "Blue shipping containers", "replace": "Orange shipping containers"}, {"find": "eastern fence", "replace": "western fence"}]'),
]


def user_msg(k: int, typ: str, sentence: str) -> str:
    return f"EDITS: {k}\nTYPE: {typ}\nSENTENCE: {sentence}"


# ----------------------------------------------------------------------------- edit logic
_WORD = re.compile(r"[a-z0-9']+")
_A_EXC = ("uni", "use", "usu", "uti", "eur", "one", "once", "ubiq", "ura")
_AN_EXC = ("hour", "honest", "heir", "honor", "honour")


def article_for(w: str) -> str:
    w = w.lower()
    if w.startswith(_A_EXC):
        return "a"
    if w.startswith(_AN_EXC):
        return "an"
    return "an" if w[:1] in "aeiou" else "a"


def norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower().replace("grey", "gray")))


def parse_edits(raw: str) -> list[tuple[str, str]]:
    raw = raw.strip()
    m = re.search(r"\[.*\]", raw, re.S) or re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no_json")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        raise ValueError("bad_json")
    if isinstance(obj, dict):
        obj = [obj]
    out = []
    for e in obj:
        if not isinstance(e, dict) or "find" not in e or "replace" not in e:
            raise ValueError("bad_schema")
        out.append((str(e["find"]).strip(), str(e["replace"]).strip()))
    return out


def locate(text: str, find: str) -> Optional[tuple[int, int]]:
    i = text.find(find)
    if i >= 0:
        return i, i + len(find)
    m = re.search(re.escape(find), text, re.I)
    return (m.start(), m.end()) if m else None


def trivial(find: str, rep: str) -> bool:
    """True if the change is a synonym / spelling / plural-only swap."""
    a, b = find.lower().split(), rep.lower().split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
            x = re.sub(r"[^a-z0-9]", "", a[i1]).replace("grey", "gray")
            y = re.sub(r"[^a-z0-9]", "", b[j1]).replace("grey", "gray")
            if x == y or x.rstrip("s") == y.rstrip("s"):
                return True
            # inflection of the same stem (red/reddish, large/larger) - NOT east/west, grass/glass, house/pond
            short, long_ = sorted((x, y), key=len)
            if len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 4:
                return True
    return False


def apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    res, pos = "", 0
    for s, e, r in sorted(spans):
        res += text[pos:s]
        m = re.match(r"(a|an)\s+([A-Za-z]+)", r, re.I)       # repair article inside replacement
        if m:
            r = r[:m.start(1)] + (article_for(m.group(2)) if m.group(1).islower() else article_for(m.group(2)).capitalize()) + r[m.end(1):]
        else:
            pm = re.search(r"\b(a|an)(\s+)$", res, re.I)      # repair article just before replacement
            w = re.match(r"[A-Za-z]+", r)
            if pm and w:
                want = article_for(w.group(0))
                if pm.group(1).lower() != want:
                    res = res[:pm.start(1)] + (want.capitalize() if pm.group(1)[0].isupper() else want) + pm.group(2)
        res += r
        pos = e
    return res + text[pos:]


def build_cf(caption: str, edits: list[tuple[str, str]], k: int) -> tuple[Optional[str], str]:
    if not 1 <= len(edits) <= k:
        return None, "n_edits"
    spans = []
    for f, r in edits:
        if not f or not r:
            return None, "empty"
        if len(f.split()) > 10 or len(r.split()) > 14:
            return None, "too_long"
        loc = locate(caption, f)
        if loc is None:
            return None, "not_found"
        if norm(f) == norm(r) or trivial(f, r):
            return None, "trivial"
        if len(re.findall(r"[.!?]", r)) > len(re.findall(r"[.!?]", f)):
            return None, "added_sentence"
        spans.append((loc[0], loc[1], r))
    spans.sort()
    if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
        return None, "overlap"
    cf = apply_spans(caption, spans)
    if cf == caption or norm(cf) == norm(caption):
        return None, "unchanged"
    return cf, ""


# ----------------------------------------------------------------------------- engine wrapper
class CFEditor:
    def __init__(self, llm: Any = None, tokenizer: Any = None, model: str = "Qwen/Qwen2.5-7B-Instruct",
                 types: Optional[dict] = None, n_edits: int = 1, temperature: float = 0.5, retries: int = 2,
                 seed: int = 0, fallback: bool = False, quantization: Optional[str] = None,
                 max_model_len: int = 4096, max_num_seqs: int = 512, max_num_batched_tokens: int = 16384,
                 gpu_mem_util: float = 0.90):
        if llm is None:
            from vllm import LLM
            kw = dict(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util,
                      max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                      enable_prefix_caching=True, trust_remote_code=True, seed=seed)
            if quantization:
                kw["quantization"] = quantization
            llm = LLM(**kw)
        self.llm, self.tok = llm, tokenizer or llm.get_tokenizer()
        try:
            from vllm import SamplingParams as SP
        except ImportError:  # mock-friendly
            class SP:  # type: ignore
                def __init__(self, **kw):
                    self.__dict__.update(kw)
        self.SP = SP
        self.types, self.k, self.temperature, self.retries, self.seed = types or TYPE_WEIGHTS, n_edits, temperature, retries, seed
        self.rule = None
        if fallback:
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from negative_captions import NegativeCaptionGenerator  # type: ignore
                self.rule = NegativeCaptionGenerator(seed=seed)
            except Exception as e:  # noqa: BLE001
                logging.warning("rule-based fallback unavailable: %s", e)
        self._build_prefix()

    def _build_prefix(self):
        msgs = [{"role": "system", "content": SYSTEM}]
        for k, typ, cap, ans in SHOTS:
            msgs += [{"role": "user", "content": user_msg(k, typ, cap)}, {"role": "assistant", "content": ans}]
        msgs.append({"role": "user", "content": PH})
        try:
            t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pre, self.post = t.split(PH)
        self.prefix_ids = self.tok(pre, add_special_tokens=False)["input_ids"]   # tokenized once

    def _type(self, uid: int) -> str:
        r = random.Random(self.seed * 1_000_003 + uid)
        return r.choices(list(self.types), list(self.types.values()))[0]

    def edit(self, caps: list[str], uids: list[int]) -> list[dict]:
        """caps = sentences. Returns one result dict per sentence."""
        n, k = len(caps), self.k
        types = [self._type(u) for u in uids]
        users = [user_msg(k, types[i], caps[i].replace(PH, "")) + self.post for i in range(n)]
        suf = self.tok(users, add_special_tokens=False)["input_ids"]
        prompts = [{"prompt_token_ids": self.prefix_ids + s} for s in suf]
        out: list[Optional[tuple[str, list]]] = [None] * n
        why = [""] * n
        pending = list(range(n))
        for attempt, temp in enumerate([self.temperature, 0.85, 1.0, 1.0][: self.retries + 1]):
            if not pending:
                break
            if attempt:   # tell the model what went wrong (few prompts -> cheap to re-tokenize)
                for i in pending:
                    note = f"\nNOTE: previous answer was invalid ({why[i]}). Copy FIND exactly from the sentence."
                    ids = self.tok(user_msg(k, types[i], caps[i]) + note + self.post, add_special_tokens=False)["input_ids"]
                    prompts[i] = {"prompt_token_ids": self.prefix_ids + ids}
            sp = self.SP(temperature=temp, top_p=0.9 if temp > 0 else 1.0, max_tokens=90 * k)
            outs = self.llm.generate([prompts[i] for i in pending], sp, use_tqdm=False)
            for i, o in zip(pending, outs):
                try:
                    edits = parse_edits(o.outputs[0].text)
                    cf, reason = build_cf(caps[i], edits, k)
                except ValueError as e:
                    cf, reason, edits = None, str(e), []
                if cf is not None:
                    out[i] = (cf, [{"find": f, "replace": r} for f, r in edits])
                else:
                    why[i] = reason
            pending = [i for i in pending if out[i] is None]

        res = []
        for i in range(n):
            if out[i]:
                res.append({"cf": out[i][0], "edits": out[i][1], "type": types[i], "status": "ok"})
                continue
            if self.rule is not None:
                r = self.rule.generate(caps[i], mut_count=1, num_sentences=1)
                if r["n_mutations"]:
                    res.append({"cf": r["negative"], "edits": [], "type": "rule", "status": "fallback_rule"})
                    continue
            res.append({"cf": None, "edits": [], "type": types[i], "status": "failed:" + why[i]})
        return res


# ----------------------------------------------------------------------------- file pipeline
def process_file(src: Path, dst: Path, ed: CFEditor, a) -> None:
    table = pq.read_table(src)
    for c in (a.out_col, "cf_edits", "cf_status"):
        if c in table.column_names:
            table = table.drop_columns([c])
    caps = table.column(a.in_col).to_pylist()
    n = len(caps)
    salt = zlib.crc32(src.stem.encode())

    # one unit per eligible sentence of every row (deterministic order -> resumable)
    rows = [i for i in range(n) if isinstance(caps[i], str) and caps[i].strip()]
    if a.dry_run:
        rows = rows[: a.dry_run]
    layout: dict[int, tuple[list[str], list[str], dict[int, int]]] = {}
    units: list[str] = []
    for i in rows:
        parts = _SENT_SPLIT.split(caps[i])
        sents, seps = parts[0::2], parts[1::2]
        mp = {}
        for j, st in enumerate(sents):
            if len(st.split()) >= a.min_words:
                mp[j] = len(units)
                units.append(st)
        layout[i] = (sents, seps, mp)

    progress = dst.with_name(dst.name + ".cf_progress.jsonl")
    done: dict[int, dict] = {}
    if progress.exists() and not a.dry_run:
        for line in progress.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["u"]] = r
        logging.info("%s: resuming, %d/%d sentences already done", src.name, len(done), len(units))

    todo = [u for u in range(len(units)) if u not in done]
    pbar = tqdm(total=len(units), initial=len(done), desc=src.name)
    pf = None if a.dry_run else open(progress, "a")
    for s0 in range(0, len(todo), a.chunk_size):
        ids = todo[s0:s0 + a.chunk_size]
        rs = ed.edit([units[u] for u in ids], [salt + u for u in ids])
        for u, r in zip(ids, rs):
            r["u"] = u
            done[u] = r
            if pf:
                pf.write(json.dumps(r, ensure_ascii=False) + "\n")
        if pf:
            pf.flush()
        pbar.update(len(ids))
    pbar.close()
    if pf:
        pf.close()

    # reassemble paragraphs: every eligible sentence replaced by its edited version
    cf: list[Optional[str]] = [None] * n
    edits_col: list[Optional[str]] = [None] * n
    status: list[str] = ["skipped"] * n
    n_sent = n_ok = 0
    for i, (sents, seps, mp) in layout.items():
        if not mp:
            continue
        out_s, all_edits, stats = list(sents), [], []
        for j, u in mp.items():
            r = done[u]
            stats.append(r["status"])
            if r["cf"]:
                out_s[j] = r["cf"]
                all_edits += [dict(e, sentence=j) for e in r["edits"]]
        good = [x for x in stats if x in ("ok", "fallback_rule")]
        n_sent += len(stats)
        n_ok += len(good)
        st = "ok" if len(good) == len(stats) else ("partial" if good else "failed:" + stats[0].split(":")[-1])
        if a.require_all and st != "ok":
            text = None
        else:
            text = out_s[0]
            for sep, sent in zip(seps, out_s[1:]):
                text += sep + sent
        cf[i] = None if st.startswith("failed") else text      # nothing edited -> no negative
        edits_col[i], status[i] = json.dumps(all_edits, ensure_ascii=False), st
        if a.dry_run:
            print(f"\n[{st}] ORIG: {caps[i]}\n      CF  : {cf[i]}")
            for j, u in mp.items():
                print(f"      s{j} {done[u]['type']:<9} {done[u]['status']:<14} {done[u]['edits']}")
    logging.info("%s: %d/%d sentences edited successfully (%.1f%%)", src.name, n_ok, n_sent, 100 * n_ok / max(1, n_sent))
    if a.dry_run:
        return

    table = table.append_column(a.out_col, pa.array(cf, pa.string()))
    if not a.no_extra:
        table = table.append_column("cf_edits", pa.array(edits_col, pa.string()))
        table = table.append_column("cf_status", pa.array(status, pa.string()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("." + dst.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, dst)                      # atomic
    if progress.exists():
        progress.unlink()
    logging.info("%s -> %s | %d rows", src.name, dst, n)


def parse_types(s: Optional[str]) -> dict:
    if not s:
        return dict(TYPE_WEIGHTS)
    bad = [t for t in s.split(",") if t not in TYPE_WEIGHTS]
    if bad:
        raise SystemExit(f"unknown types {bad}; choose from {list(TYPE_WEIGHTS)}")
    return {t: TYPE_WEIGHTS[t] for t in s.split(",")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory with caption-XX.parquet files.")
    ap.add_argument("--output_dir", default=None, help="Write copies here instead of rewriting files in place.")
    ap.add_argument("--pattern", default="caption*.parquet")
    ap.add_argument("--in_col", default="dense_caption")
    ap.add_argument("--out_col", default="cf_dense_caption")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (faster on H100)")
    ap.add_argument("--edits", type=int, default=1, help="Facts to flip PER SENTENCE (default 1).")
    ap.add_argument("--min_words", type=int, default=4, help="Sentences shorter than this are left unchanged.")
    ap.add_argument("--require_all", action="store_true", help="Set cf_dense_caption to null unless EVERY sentence was edited.")
    ap.add_argument("--types", default=None, help=f"Comma list from {list(TYPE_WEIGHTS)} (default: weighted mix).")
    ap.add_argument("--chunk_size", type=int, default=16384, help="Sentences per vLLM batch.")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--fallback", action="store_true", help="Use negative_captions.py rule-based generator if LLM fails.")
    ap.add_argument("--max_num_seqs", type=int, default=512)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_extra", action="store_true", help="Only add the cf_dense_caption column.")
    ap.add_argument("--overwrite", action="store_true", help="Recompute files that already have the column.")
    ap.add_argument("--dry_run", type=int, default=0, help="Show N paragraphs from the first pending file; write nothing.")
    a = ap.parse_args()

    root = Path(a.input_dir)
    files = sorted(p for p in root.glob("**/" + a.pattern) if not p.name.startswith("."))
    pending = []
    for p in files:
        dst = (Path(a.output_dir) / p.relative_to(root)) if a.output_dir else p
        have = dst.exists() and a.out_col in pq.ParquetFile(dst).schema_arrow.names
        if have and not a.overwrite:
            logging.info("Skipping finished file: %s", dst.name)
            continue
        if a.in_col not in pq.ParquetFile(p).schema_arrow.names:
            logging.warning("%s has no '%s' column; skipping", p.name, a.in_col)
            continue
        pending.append((p, dst))
    if not pending:
        logging.info("Nothing to do.")
        return
    logging.info("%d file(s) to process", len(pending))
    if a.dry_run:
        pending = pending[:1]

    ed = CFEditor(model=a.model_name, types=parse_types(a.types), n_edits=a.edits, temperature=a.temperature,
                  retries=a.retries, seed=a.seed, fallback=a.fallback, quantization=a.quantization,
                  max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, gpu_mem_util=a.gpu_mem_util)
    for src, dst in pending:
        process_file(src, dst, ed, a)


if __name__ == "__main__":
    main()