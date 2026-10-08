"""Torch-free text helpers used by the data pipeline."""
from __future__ import annotations

import ast
import re
from typing import Dict, Iterable, List, Sequence, Tuple

_QUOTED = re.compile(r"'(.*?)'")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def parse_key_elements(value) -> List[str]:
    """Parse a key_elements cell into a list of strings.

    The cell is normally a str such as "['red building', 'road']". We try
    ast.literal_eval first (handles apostrophes inside double-quoted items), then
    fall back to the `'(.*?)'` regex, then to a plain comma split. Order is preserved
    and nothing is de-duplicated: for the mask csv the position of each keyword
    is the index of its mask in the .npy.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        seq = value.tolist() if hasattr(value, "tolist") else value
        return [str(v).strip() for v in seq]
    if isinstance(value, float):  # NaN from pandas
        return []
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(v).strip() for v in parsed]
    except (ValueError, SyntaxError):
        pass
    found = _QUOTED.findall(s)
    if found:
        return [f.strip() for f in found]
    return [p.strip() for p in s.strip("[]").split(",") if p.strip()]


def split_sentences(text: str, min_chars: int = 3) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [p.strip() for p in _SENT_SPLIT.split(text.strip()) if len(p.strip()) >= min_chars]


def build_unique(groups: Sequence[Iterable[str]]) -> Tuple[List[str], List[List[int]]]:
    """De-duplicate texts across items.

    groups[i] are the texts that belong to item i. Returns (unique_texts, idx) where
    idx[i] are the indices into unique_texts for item i. Identical strings shared by
    several items become ONE text column, so they are positives for all of those items
    instead of false negatives for the others.
    """
    table: Dict[str, int] = {}
    uniq: List[str] = []
    idx: List[List[int]] = []
    for g in groups:
        cur: List[int] = []
        for t in g:
            j = table.get(t)
            if j is None:
                j = table[t] = len(uniq)
                uniq.append(t)
            if j not in cur:
                cur.append(j)
        idx.append(cur)
    return uniq, idx
