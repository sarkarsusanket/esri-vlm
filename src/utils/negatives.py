#!/usr/bin/env python3
"""Fast, algorithmic hard-negative caption generator for AERIAL image captions.

No LLM / no network / no heavy deps (pure stdlib; pandas only for file IO).
~tens of thousands of captions per second per core.

Each caption is split into sentences and `--mut-count` small mutations are applied
to EVERY sentence (as many as the sentence supports). Mutations are context-aware,
not blind string swaps:

  color      red -> blue, using palettes that are plausible for the noun it modifies
             (roofs / vehicles / vegetation / water / roads ...), perceptual (RGB)
             distance so near-synonyms (red -> maroon) are never produced, shade flips
             (dark <-> light), avoids duplicating colors already in the sentence.
  direction  north/south/east/west (+ compounds / -ern / -ward), left/right, top/bottom,
             top-left style corners (single-axis flips = hard negatives), center/edge,
             foreground/background, front/rear.
  relation   above/below, inside/outside, in front of/behind, adjacent/far from,
             along/across, toward/away from, clockwise/counterclockwise.
  material   metal/tiled/thatched/... roofs, asphalt/gravel/dirt... surfaces,
             brick/wood/stone... structures (set chosen from the noun's context).
  attribute  size, length, width, height, density, shape, roof shape, path, condition,
             age, moisture, occupancy, motion, alignment, arrangement, terrain ...
  degree     fully / mostly / partially ...
  quantity   numbers (two -> five, 12 -> 18), few <-> many (keeps plural agreement).
  object     co-hyponym swap with number kept: pool -> pond, cars -> trucks,
             tennis court -> basketball court (never hypernym/hyponym pairs).
  negation   with <-> without, "a pool" -> "no pool", there is a -> there is no,
             is -> is not, has -> lacks, visible -> hidden, can be seen -> cannot be seen
             (max ONE negation-type edit per sentence so edits never cancel out).
  wordnet    (optional, --wordnet) adjective antonyms for words not in the lexicon.

Grammar is repaired automatically (a/an, case, plural forms).
`--hardness` in [0,1]: 0 = prefer obvious/dissimilar swaps, 1 = prefer the most
similar (subtle) swaps that are still contradictory.

Usage
-----
  python negative_captions.py --caption "A red roof house has a swimming pool to the north." --mut-count 2
  python negative_captions.py --input in.parquet --output out.parquet --caption-col dense_caption \
         --mut-count 2 --workers 32 --num-variants 3
  cat captions.txt | python negative_captions.py --mut-count 1          # JSONL to stdout

Library
-------
  from negative_captions import NegativeCaptionGenerator
  g = NegativeCaptionGenerator(seed=0, hardness=0.5)
  r = g.generate("A red roof ...", mut_count=2)
  r["negative"], r["mutations"]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

# =============================================================================
# small text utilities
# =============================================================================

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s.lower().strip())


def match_case(src: str, dst: str) -> str:
    if not dst:
        return dst
    if len(src) > 1 and src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


_A_EXC = ("uni", "use", "usu", "uti", "eur", "one", "once", "ubiq", "ura")
_AN_EXC = ("hour", "honest", "heir", "honor", "honour")


def article_for(word: str) -> str:
    w = word.lower()
    if w.startswith(_A_EXC):
        return "a"
    if w.startswith(_AN_EXC):
        return "an"
    return "an" if w[:1] in "aeiou" else "a"


_IRREG = {"man": "men", "woman": "women", "child": "children", "person": "people"}


def pluralize(phrase: str) -> str:
    head, _, last = phrase.rpartition(" ")
    low = last.lower()
    if low in _IRREG:
        p = _IRREG[low]
    elif re.search(r"(s|x|z|ch|sh)$", low):
        p = last + "es"
    elif re.search(r"[^aeiou]y$", low):
        p = last[:-1] + "ies"
    else:
        p = last + "s"
    return (head + " " + p) if head else p


_TOK = re.compile(r"[a-z]+(?:-[a-z]+)*")

# =============================================================================
# color model
# =============================================================================

RGB = {
    "red": (200, 40, 40), "orange": (240, 140, 30), "yellow": (240, 220, 60),
    "green": (60, 160, 60), "blue": (40, 90, 200), "purple": (130, 60, 160),
    "pink": (240, 150, 180), "brown": (130, 85, 50), "tan": (200, 170, 120),
    "beige": (225, 205, 170), "gray": (128, 128, 128), "black": (25, 25, 25),
    "white": (240, 240, 240), "silver": (190, 195, 200), "gold": (212, 175, 55),
    "teal": (0, 128, 128), "maroon": (110, 20, 30), "cream": (250, 240, 210),
    "turquoise": (64, 224, 208), "olive": (110, 120, 40), "navy": (20, 30, 100),
}
ISH = {
    "red": "reddish", "orange": "orangish", "yellow": "yellowish", "green": "greenish",
    "blue": "bluish", "purple": "purplish", "pink": "pinkish", "brown": "brownish",
    "gray": "grayish", "white": "whitish", "black": "blackish",
}
COLOR_SURF = {}  # surface -> (base, is_ish)
for _b in RGB:
    COLOR_SURF[_b] = (_b, False)
for _b, _i in ISH.items():
    COLOR_SURF[_i] = (_b, True)
COLOR_SURF["grey"] = ("gray", False)
COLOR_SURF["greyish"] = ("gray", True)

SHADES = {"dark": "light", "light": "dark", "bright": "dull", "dull": "bright",
          "pale": "deep", "deep": "pale", "vivid": "muted", "muted": "vivid"}

COLOR_RX = re.compile(
    r"(?<![\w])(?:(?P<mod>" + "|".join(SHADES) + r")(?P<sep>[- ]))?(?P<col>"
    + "|".join(sorted(COLOR_SURF, key=len, reverse=True)) + r")(?![\w])", re.I)

PALETTES = {
    "roof": ["red", "brown", "gray", "black", "blue", "green", "white", "tan", "orange", "silver", "maroon"],
    "building": ["white", "gray", "beige", "tan", "brown", "red", "blue", "yellow", "cream", "black"],
    "vehicle": ["white", "black", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown"],
    "vegetation": ["green", "brown", "yellow", "tan", "olive", "gold", "orange"],
    "water": ["blue", "green", "turquoise", "brown", "teal", "gray"],
    "road": ["gray", "black", "tan", "brown", "white", "beige", "red"],
    "ground": ["brown", "tan", "beige", "gray", "red", "yellow", "white"],
}
DEFAULT_PALETTE = list(RGB)
MIN_COLOR_DIST = 85.0
_MAXD = math.dist((0, 0, 0), (255, 255, 255))

# noun -> context class (used for colors and materials)
_CLASS_LISTS = {
    "roof": "roof rooftop roofed rooftops",
    "building": "building house home warehouse barn shed garage church factory structure wall facade "
                "apartment complex tower silo bridge fence deck school stadium cabin hangar greenhouse "
                "mansion villa condo dwelling",
    "vehicle": "car truck van bus vehicle motorcycle tractor trailer boat ship sedan suv pickup",
    "vegetation": "tree grass field vegetation lawn forest crop shrub foliage farmland bush meadow "
                  "pasture garden canopy woodland orchard vineyard cropland grassland plants",
    "water": "water pool pond lake river canal stream reservoir sea ocean lagoon creek",
    "road": "road driveway pavement parking lot runway path sidewalk street highway lane track trail "
            "taxiway alley",
    "ground": "soil dirt sand ground desert terrain beach land patch clearing",
}
CLASS_OF: dict[str, str] = {}
for _c, _ws in _CLASS_LISTS.items():
    for _w in _ws.split():
        CLASS_OF.setdefault(_w, _c)


def _cls(tok: str) -> Optional[str]:
    if tok in CLASS_OF:
        return CLASS_OF[tok]
    if tok.endswith("es") and tok[:-2] in CLASS_OF:
        return CLASS_OF[tok[:-2]]
    if tok.endswith("s") and tok[:-1] in CLASS_OF:
        return CLASS_OF[tok[:-1]]
    return None


def ctx_class(text: str, s: int, e: int) -> Optional[str]:
    for t in _TOK.findall(text[e:e + 60].lower())[:4]:
        c = _cls(t)
        if c:
            return c
    for t in reversed(_TOK.findall(text[max(0, s - 80):s].lower())[-8:]):
        c = _cls(t)
        if c:
            return c
    return None


# =============================================================================
# materials
# =============================================================================

MATERIAL_GROUPS = {
    "metal": ["metal", "metallic", "steel", "aluminum"], "wood": ["wooden", "wood"],
    "tile": ["tiled"], "shingle": ["shingled"], "thatch": ["thatched"], "concrete": ["concrete"],
    "slate": ["slate"], "brick": ["brick"], "stone": ["stone"], "glass": ["glass"],
    "stucco": ["stucco"], "asphalt": ["asphalt", "tarmac", "blacktop"], "gravel": ["gravel"],
    "dirt": ["dirt"], "cobble": ["cobblestone"], "sandy": ["sandy"], "grassy": ["grassy"],
}
MAT_OF = {p: g for g, ps in MATERIAL_GROUPS.items() for p in ps}
MAT_CTX = {
    "roof": ["metal", "tile", "shingle", "thatch", "concrete", "wood", "slate"],
    "road": ["asphalt", "concrete", "gravel", "dirt", "cobble", "sandy"],
    "ground": ["asphalt", "concrete", "gravel", "dirt", "cobble", "sandy"],
    "building": ["brick", "wood", "concrete", "stone", "glass", "metal", "stucco"],
    None: ["metal", "wood", "concrete", "brick", "stone", "glass", "asphalt", "gravel", "dirt"],
}
MAT_SIM = [{"concrete", "asphalt"}, {"gravel", "dirt", "sandy"}, {"brick", "stone", "stucco"},
           {"metal", "glass"}, {"tile", "shingle", "slate"}]
MAT_RX = re.compile(r"(?<![\w-])(?:" + "|".join(sorted(map(re.escape, MAT_OF), key=len, reverse=True)) + r")(?![\w])", re.I)

# =============================================================================
# object (noun) sets: members of a set are mutually-exclusive co-hyponyms
# each member is a synonym group; first entry is canonical
# =============================================================================

NOUN_SETS = {
    "building": [["house"], ["warehouse"], ["barn"], ["shed"], ["church"], ["garage"], ["factory"],
                 ["apartment complex", "apartment block"], ["greenhouse"]],
    "vehicle": [["car"], ["truck", "lorry"], ["bus"], ["van"], ["motorcycle"], ["tractor"]],
    "aircraft": [["airplane", "plane", "aeroplane"], ["helicopter"]],
    "water": [["pond"], ["lake"], ["river"], ["canal"], ["swimming pool", "pool"], ["reservoir"], ["stream", "creek"]],
    "plant": [["tree"], ["shrub", "bush"]],
    "landcover": [["forest", "woodland"], ["farmland", "cropland"], ["grassland", "meadow"],
                  ["wetland", "marsh"], ["desert"], ["orchard"], ["vineyard"]],
    "road": [["road"], ["railway", "railroad", "rail line"], ["runway"], ["trail", "footpath"], ["sidewalk"]],
    "sport": [["tennis court"], ["basketball court"], ["baseball diamond", "baseball field"],
              ["soccer field"], ["running track"], ["golf course"], ["volleyball court"]],
    "structure": [["bridge"], ["dam"], ["pier", "jetty"], ["silo"], ["water tower"], ["smokestack"]],
    "site": [["parking lot", "car park"], ["cemetery"], ["playground"], ["construction site"],
             ["solar farm"], ["quarry"], ["marina"], ["stadium"]],
    "rooftop": [["solar panel"], ["chimney"], ["skylight"], ["satellite dish"], ["air conditioning unit", "ac unit"]],
    "fence": [["fence"], ["wall"], ["hedge"]],
    "boat": [["sailboat"], ["barge"], ["ferry"], ["yacht"], ["tanker"], ["kayak"]],
}
NOUN_SURF: dict[str, tuple[str, int, bool]] = {}
for _sn, _groups in NOUN_SETS.items():
    for _gi, _g in enumerate(_groups):
        for _m in _g:
            for _form, _pl in ((_m, False), (pluralize(_m), True)):
                if _form in NOUN_SURF and NOUN_SURF[_form][:2] != (_sn, _gi):
                    raise ValueError(f"duplicate noun surface {_form}")
                NOUN_SURF[_form] = (_sn, _gi, _pl)
NOUN_RX = re.compile(r"(?<![\w-])(?:" + "|".join(
    re.escape(p).replace(r"\ ", r"\s+") for p in sorted(NOUN_SURF, key=len, reverse=True)) + r")(?![\w])", re.I)

# =============================================================================
# generic axis lexicons (attribute / direction / relation / degree / quantity / negation)
# =============================================================================


def ax(groups, opp=None, skip=None, allow=None, deny=None):
    return {"groups": groups, "opp": opp or {}, "skip": skip or set(), "allow": allow, "deny": deny}


ATTRIBUTE_AXES = [
    ax([["large", "big", "huge", "massive", "sizable", "enormous"], ["small", "tiny", "compact", "miniature"]]),
    ax([["long", "elongated"], ["short", "stubby"]]),
    ax([["wide", "broad"], ["narrow", "thin", "slender"]]),
    ax([["tall", "high-rise"], ["low", "low-rise"]]),
    ax([["dense", "clustered", "closely spaced", "tightly packed", "congested"],
        ["sparse", "scattered", "isolated", "sporadic", "widely spaced", "spread out"]]),
    ax([["rectangular"], ["square"], ["circular", "round"], ["oval", "elliptical"], ["triangular"],
        ["irregular", "irregularly shaped"], ["l-shaped"]], skip={(0, 1), (1, 0)}),
    ax([["straight", "linear"], ["curved", "winding", "meandering", "serpentine", "curving"]]),
    ax([["empty", "vacant", "unoccupied", "deserted"], ["crowded", "occupied", "busy", "packed"]]),
    ax([["calm", "smooth"], ["rough", "choppy", "turbulent"]]),
    ax([["dry", "arid", "parched"], ["wet", "flooded", "muddy", "waterlogged", "damp"]]),
    ax([["lush", "verdant", "vegetated", "leafy", "forested", "wooded"], ["barren", "bare", "treeless", "unvegetated"]]),
    ax([["well-maintained", "manicured", "neat", "tidy", "pristine", "well-kept"],
        ["overgrown", "unkempt", "neglected", "dilapidated", "run-down", "derelict"]]),
    ax([["new", "newly built", "modern", "recently built"], ["old", "aged", "historic", "ancient"]]),
    ax([["fenced", "enclosed", "gated", "walled"], ["unfenced", "open", "unenclosed"]]),
    ax([["parked", "stationary", "stopped"], ["moving", "driving", "traveling"]]),
    ax([["sunlit", "illuminated"], ["shaded", "shadowed"]]),
    ax([["completed", "finished"], ["unfinished", "under construction", "incomplete"]]),
    ax([["horizontal"], ["vertical"], ["diagonal"]], opp={(0, 1): 3.0}),
    ax([["parallel", "aligned"], ["perpendicular", "staggered", "skewed"]]),
    ax([["orderly", "grid-like", "regularly spaced", "evenly spaced", "uniform"],
        ["haphazard", "random", "irregularly spaced", "disordered"]]),
    ax([["paved"], ["unpaved"]]),
]
STORY_AXES = [ax([["multi-story", "multistory", "multi-storey"],
                  ["single-story", "one-story", "single-storey"]], allow={"building", "roof"})]
ROOF_AXES = [ax([["flat", "flat-topped"], ["gabled"], ["hipped"], ["domed"]], allow={"roof"})]
TERRAIN_AXES = [ax([["flat", "level"], ["hilly", "undulating", "rugged", "mountainous"]],
                   deny={"roof", "vehicle", "building"})]
DIRECTION_AXES = [
    ax([["left", "left-hand"], ["right", "right-hand"]]),
    ax([["top", "upper", "uppermost"], ["bottom", "lower"]]),
    ax([["foreground"], ["background"]]),
    ax([["center", "centre", "middle"], ["corner", "periphery"]]),
    ax([["central"], ["peripheral"]]),
    ax([["front"], ["rear"]]),
]
RELATION_AXES = [
    ax([["above", "on top of"], ["below", "beneath", "underneath"]]),
    ax([["in front of", "ahead of"], ["behind", "beyond"]]),
    ax([["inside", "within"], ["outside", "outside of"]]),
    ax([["adjacent to", "next to", "beside", "close to", "alongside", "adjoining", "near"],
        ["far from", "distant from", "separated from"]]),
    ax([["along"], ["across", "perpendicular to"]]),
    ax([["toward", "towards"], ["away from"]]),
    ax([["clockwise"], ["counterclockwise", "counter-clockwise", "anticlockwise"]]),
]
DEGREE_AXES = [
    ax([["fully", "completely", "entirely", "totally"], ["mostly", "largely", "predominantly", "primarily", "mainly"],
        ["partially", "partly", "slightly", "somewhat", "barely"]], opp={(0, 2): 2.0}),
]
QUANT_AXES = [
    ax([["few", "a few", "a couple of", "a handful of", "only a few"],
        ["many", "numerous", "countless", "dozens of", "a large number of", "plentiful"]]),
]
NEGATION_AXES = [
    ax([["visible", "present", "apparent", "evident"], ["hidden", "absent", "obscured", "invisible"]]),
]
NEG_BEFORE_RX = re.compile(r"\b(?:no|not|without|never|n't)\b[^.,;]{0,30}$", re.I)
GUARDS = {"right": re.compile(r"\s+angle", re.I)}


class AxisLex:
    def __init__(self, category: str, axes: list[dict], tag: Optional[str] = None, neg_guard: bool = False):
        self.category, self.axes, self.tag, self.neg_guard = category, axes, tag, neg_guard
        self.phr: dict[str, tuple[int, int]] = {}
        for ai, a in enumerate(axes):
            for gi, g in enumerate(a["groups"]):
                for p in g:
                    if p in self.phr:
                        raise ValueError(f"duplicate phrase '{p}' in {category}")
                    self.phr[p] = (ai, gi)
        pats = sorted(self.phr, key=len, reverse=True)
        self.rx = re.compile(r"(?<![\w-])(?:" + "|".join(
            re.escape(p).replace(r"\ ", r"\s+") for p in pats) + r")(?![\w])", re.I)

    def find(self, text: str):
        for m in self.rx.finditer(text):
            raw = m.group(0)
            key = norm(raw)
            if key not in self.phr:
                continue
            if raw[0].isupper() and m.start() > 0 and raw[1:].islower():
                continue  # proper noun
            g = GUARDS.get(key)
            if g and g.match(text, m.end()):
                continue
            if self.neg_guard and NEG_BEFORE_RX.search(text[:m.start()]):
                continue
            ai, gi = self.phr[key]
            a = self.axes[ai]
            if a["allow"] or a["deny"]:
                c = ctx_class(text, m.start(), m.end())
                if a["allow"] and c not in a["allow"]:
                    continue
                if a["deny"] and c in a["deny"]:
                    continue
            opts = []
            for gj, grp in enumerate(a["groups"]):
                if gj == gi or (gi, gj) in a["skip"]:
                    continue
                w = a["opp"].get((gi, gj), a["opp"].get((gj, gi), 1.0))
                for k, member in enumerate(grp):
                    opts.append((match_case(raw, member), w * (1.0 if k == 0 else 0.6)))
            if opts:
                yield Cand(m.start(), m.end(), self.category, opts, self.tag)


# =============================================================================
# candidates
# =============================================================================


@dataclass
class Cand:
    start: int
    end: int
    cat: str
    options: list
    tag: Optional[str] = None
    block_end: Optional[int] = None  # extra span (e.g. "a red pool" -> "no ...") nothing else may touch


ATTR_LEXES = [AxisLex("attribute", ATTRIBUTE_AXES), AxisLex("attribute", ROOF_AXES), AxisLex("attribute", STORY_AXES),
              AxisLex("attribute", TERRAIN_AXES)]
DIR_LEX = AxisLex("direction", DIRECTION_AXES)
REL_LEX = AxisLex("relation", RELATION_AXES)
DEG_LEX = AxisLex("degree", DEGREE_AXES)
QUANT_LEX = AxisLex("quantity", QUANT_AXES)
NEG_LEX = AxisLex("negation", NEGATION_AXES, tag="neg", neg_guard=True)

OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
CARD_RX = re.compile(
    r"(?<![\w-])(?:(?P<a>north|south)(?P<sep>[- ]?)(?P<b>east|west)|(?P<single>north|south|east|west))"
    r"(?P<suf>ern|wards?)?(?![\w])", re.I)
CORNER_RX = re.compile(r"(?<![\w-])(?P<v>top|upper|bottom|lower)(?P<sep>[- ])(?P<h>left|right)(?![\w])", re.I)
VFLIP = {"top": "bottom", "bottom": "top", "upper": "lower", "lower": "upper"}
HFLIP = {"left": "right", "right": "left"}

NUMW = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
NUM_VAL = {w: i + 2 for i, w in enumerate(NUMW)}
NUM_RX = re.compile(r"(?<![\w.,/-])(?:" + "|".join(NUMW) + r"|\d{1,4})(?![\w%/]|[.,]\d)", re.I)

THERE_DET_RX = re.compile(
    r"(?<![\w-])there\s+(?P<c>is|are)\s+(?:a|an|one|some|several|many|numerous|multiple|two|three|four|five|six|\d+)(?![\w])", re.I)
THERE_NO_RX = re.compile(r"(?<![\w-])there\s+(?P<c>is|are)\s+no(?![\w])", re.I)
WITH_RX = re.compile(r"(?<![\w-])with(?![\w-])", re.I)
WITHOUT_RX = re.compile(r"(?<![\w-])without(?![\w-])", re.I)
NO_RX = re.compile(r"(?<![\w-])no(?=\s+[a-z])", re.I)
COP_RX = re.compile(r"(?<![\w-])(?<!there )(?P<c>is|are|was|were)(?![\w-])(?!\s+(?:not|no|being)\b)(?=\s+(?P<n>[a-z-]+))", re.I)
COPNOT_RX = re.compile(r"(?<![\w-])(?P<c>is|are|was|were)\s+not(?![\w-])|(?<![\w-])(?P<c2>is|are|was|were)n't", re.I)
CANNOT_RX = re.compile(r"(?<![\w-])(?:cannot|can\s+not|can't)(?![\w-])", re.I)
CANBE_RX = re.compile(r"(?<![\w-])can\s+be(?=\s+(?:seen|observed|found|identified|noticed|detected))", re.I)
VERB_FLIP = {"has": "lacks", "have": "lack", "contains": "lacks", "contain": "lack",
             "features": "lacks", "includes": "lacks", "lacks": "has", "lack": "have"}
VERB_RX = re.compile(r"(?<![\w-])(?:" + "|".join(VERB_FLIP) + r")(?![\w-])(?!\s+(?:been|also|to)\b)", re.I)
WITH_PREV_STOP = {"along", "together", "aligned", "parallel", "coupled", "combined", "mixed", "filled",
                  "covered", "lined", "dotted", "surrounded", "bordered", "paved", "planted", "topped",
                  "scattered", "interspersed", "adorned", "decorated", "marked", "equipped", "connected"}
NP_STOP = {"and", "or", "of", "with", "near", "by", "to", "in", "on", "at", "from", "the", "that", "which",
           "is", "are", "has", "have", "for", "over", "under"}
PRE_DET_RX = re.compile(
    r"(?:^|\s)(?P<d>a|an|several|many|numerous|multiple|some|two|three|four|five|six|seven|eight|nine|\d+)\s+"
    r"(?P<mid>(?:[\w-]+\s+){0,2})$", re.I)

# =============================================================================
# WordNet fallback (optional)
# =============================================================================


@lru_cache(maxsize=100000)
def _wn_antonyms(word: str) -> tuple:
    from nltk.corpus import wordnet as wn  # type: ignore
    adj = wn.synsets(word, pos="a") + wn.synsets(word, pos="s")
    if not adj:
        return ()
    other = len(wn.synsets(word, pos="n")) + len(wn.synsets(word, pos="v"))
    if len(adj) <= other:
        return ()
    ants = set()
    for syn in adj:
        for lem in syn.lemmas():
            if lem.name().lower() == word:
                for a in lem.antonyms():
                    ants.add(a.name().replace("_", " "))
    return tuple(sorted(a for a in ants if a.lower() != word))


# =============================================================================
# generator
# =============================================================================

CATEGORIES = ("color", "direction", "relation", "material", "attribute", "degree",
              "quantity", "object", "negation", "wordnet")
DEFAULT_WEIGHTS = {"color": 3.0, "direction": 2.5, "relation": 1.5, "material": 2.0, "attribute": 2.0,
                   "degree": 0.8, "quantity": 1.5, "object": 2.0, "negation": 1.5, "wordnet": 0.4}
TAG_LIMITS = {"neg": 1}
_SENT_SPLIT = re.compile(r"((?<=[.!?])\s+)")


class NegativeCaptionGenerator:
    def __init__(self, seed: Optional[int] = None, hardness: float = 0.5,
                 weights: Optional[dict[str, float]] = None, categories: Optional[list[str]] = None,
                 wordnet: bool = False, decay: float = 0.35):
        self.seed = seed
        self.hardness = min(1.0, max(0.0, hardness))
        self.decay = decay
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        allowed = set(categories) if categories else set(CATEGORIES)
        if not wordnet:
            allowed.discard("wordnet")
        for c in CATEGORIES:
            if c not in allowed:
                self.weights[c] = 0.0
        self.wordnet = wordnet and self.weights.get("wordnet", 0) > 0
        if self.wordnet:
            try:
                _wn_antonyms("large")
            except Exception as e:  # noqa: BLE001
                print(f"[warn] WordNet unavailable ({e}); disabling --wordnet", file=sys.stderr)
                self.wordnet = False
        self._rng = random.Random(seed)
        self._find_c = lru_cache(maxsize=8192)(self._find)  # variants/duplicates reuse candidates

    # ---------------------------------------------------------------- finders
    def _color(self, t: str):
        ms = list(COLOR_RX.finditer(t))
        if not ms:
            return
        present = {COLOR_SURF[m.group("col").lower()][0] for m in ms}
        h = self.hardness
        for m in ms:
            if m.group("col")[0].isupper() and m.start() > 0 and m.group("col")[1:].islower() and not m.group("mod"):
                # capitalized mid-sentence ("Red Cross") -> skip
                continue
            col = m.group("col")
            base, is_ish = COLOR_SURF[col.lower()]
            mod, sep = m.group("mod"), m.group("sep") or ""
            palette = PALETTES.get(ctx_class(t, m.start(), m.end()), DEFAULT_PALETTE)
            cols = [c for c in palette if c not in present and math.dist(RGB[base], RGB[c]) >= MIN_COLOR_DIST]
            if not cols:
                cols = [c for c in RGB if c not in present and math.dist(RGB[base], RGB[c]) >= MIN_COLOR_DIST]
            opts = []
            for c in cols:
                sim = 1.0 - math.dist(RGB[base], RGB[c]) / _MAXD
                w = math.exp(4.0 * h * sim)
                new = ISH[c] if (is_ish and c in ISH) else c
                new = match_case(col, new) if not mod else new
                opts.append(((mod + sep if mod else "") + new, w))
            if mod:
                opts.append((match_case(mod, SHADES[mod.lower()]) + sep + col, 0.8 + 2.0 * h))
            if opts:
                yield Cand(m.start(), m.end(), "color", opts)

    def _material(self, t: str):
        for m in MAT_RX.finditer(t):
            raw = m.group(0)
            if raw[0].isupper() and m.start() > 0:
                continue
            g = MAT_OF[raw.lower()]
            c = ctx_class(t, m.start(), m.end())
            if c in ("vehicle", "vegetation", "water"):
                continue
            pool = MAT_CTX.get(c, MAT_CTX[None])
            opts = []
            for g2 in pool:
                if g2 == g:
                    continue
                sim = any(g in s and g2 in s for s in MAT_SIM)
                opts.append((match_case(raw, MATERIAL_GROUPS[g2][0]), 1.0 + (2.0 * self.hardness if sim else 0.0)))
            if opts:
                yield Cand(m.start(), m.end(), "material", opts)

    def _cardinal(self, t: str):
        h = self.hardness
        for m in CARD_RX.finditer(t):
            raw = m.group(0)
            if raw[0].isupper() and m.start() > 0 and raw[1:].islower():
                continue
            suf = m.group("suf") or ""
            if m.group("single"):
                w0 = m.group("single").lower()
                opts = []
                for o in OPP:
                    if o != w0:
                        opts.append((match_case(raw, o + suf), 3.0 if OPP[w0] == o else 1.0))
                yield Cand(m.start(), m.end(), "direction", opts)
            else:
                a, sep, b = m.group("a"), m.group("sep"), m.group("b")
                a_, b_ = a.lower(), b.lower()
                def mk(x, y):
                    return match_case(raw, x + sep + y + suf)
                opts = [(mk(OPP[a_], b_), 1.0 + 2.0 * h), (mk(a_, OPP[b_]), 1.0 + 2.0 * h), (mk(OPP[a_], OPP[b_]), 1.0)]
                yield Cand(m.start(), m.end(), "direction", opts)

    def _corner(self, t: str):
        h = self.hardness
        for m in CORNER_RX.finditer(t):
            v, sep, hh = m.group("v"), m.group("sep"), m.group("h")
            raw = m.group(0)
            vf, hf = VFLIP[v.lower()], HFLIP[hh.lower()]
            def mk(a, b):
                return match_case(raw, a + sep + b)
            opts = [(mk(vf, hh.lower()), 1.0 + 2.0 * h), (mk(v.lower(), hf), 1.0 + 2.0 * h), (mk(vf, hf), 1.0)]
            yield Cand(m.start(), m.end(), "direction", opts)

    def _number(self, t: str):
        h = self.hardness
        for m in NUM_RX.finditer(t):
            raw = m.group(0)
            low = raw.lower()
            v = NUM_VAL.get(low) or (int(low) if low.isdigit() else 0)
            if v < 2:
                continue
            step = max(1, v // 6)
            cands = {v + d * step for d in (-3, -2, -1, 1, 2, 3)} | {v * 2, max(2, v // 2)}
            cands = {c for c in cands if c >= 2 and c != v}
            opts = []
            for c in cands:
                rel = abs(c - v) / v
                w = math.exp(-3.0 * h * rel)
                s = NUMW[c - 2] if (low in NUM_VAL and c <= 12) else str(c)
                opts.append((match_case(raw, s), w))
            if opts:
                yield Cand(m.start(), m.end(), "quantity", opts)

    def _objects(self, t: str):
        seen = set()
        for m in NOUN_RX.finditer(t):
            raw = m.group(0)
            if raw[0].isupper() and m.start() > 0 and raw[1:].islower():
                continue
            sn, gi, pl = NOUN_SURF[norm(raw)]
            opts = []
            for gj, g in enumerate(NOUN_SETS[sn]):
                if gj == gi:
                    continue
                for k, member in enumerate(g):
                    form = pluralize(member) if pl else member
                    opts.append((match_case(raw, form), 1.0 if k == 0 else 0.4))
            yield Cand(m.start(), m.end(), "object", opts)
            # object removal:  "a pool" / "several cars"  ->  "no pool" / "no cars"
            base = max(0, m.start() - 40)
            pm = PRE_DET_RX.search(t[base:m.start()])
            if pm:
                mids = pm.group("mid").split()
                if all(w.lower() not in NP_STOP for w in mids):
                    s, e = base + pm.start("d"), base + pm.end("d")
                    if (s, e) not in seen:
                        seen.add((s, e))
                        yield Cand(s, e, "negation", [(match_case(pm.group("d"), "no"), 1.0)], "neg", block_end=m.end())

    def _negation(self, t: str):
        for m in THERE_DET_RX.finditer(t):
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), f"there {m.group('c').lower()} no"), 1.0)], "neg")
        for m in THERE_NO_RX.finditer(t):
            c = m.group("c").lower()
            alts = ["a"] if c == "is" else ["several", "multiple"]
            yield Cand(m.start(), m.end(), "negation",
                       [(match_case(m.group(0), f"there {c} {a}"), 1.0) for a in alts], "neg")
        for m in WITH_RX.finditer(t):
            prev = _TOK.findall(t[max(0, m.start() - 30):m.start()].lower())
            p = prev[-1] if prev else ""
            if p in WITH_PREV_STOP or p.endswith(("ed", "ing", "en")):
                continue
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), "without"), 1.0)], "neg")
        for m in WITHOUT_RX.finditer(t):
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), "with"), 1.0)], "neg")
        for m in NO_RX.finditer(t):
            if re.search(r"there\s+(is|are)\s+$", t[max(0, m.start() - 12):m.start()], re.I):
                continue
            nxt = _TOK.findall(t[m.end():m.end() + 50].lower())[:3]
            plural = any(w.endswith("s") and not w.endswith(("ss", "us", "is")) for w in nxt)
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), "some" if plural else "a"), 1.0)], "neg")
        for m in COP_RX.finditer(t):
            nxt = m.group("n").lower()
            if nxt.endswith("ly") or nxt in {"also", "only", "still", "just", "then", "now", "often", "usually"}:
                continue
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group("c"), m.group("c").lower() + " not"), 1.0)], "neg")
        for m in COPNOT_RX.finditer(t):
            c = (m.group("c") or m.group("c2")).lower()
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), c), 1.0)], "neg")
        for m in CANNOT_RX.finditer(t):
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), "can"), 1.0)], "neg")
        for m in CANBE_RX.finditer(t):
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), "cannot be"), 1.0)], "neg")
        for m in VERB_RX.finditer(t):
            yield Cand(m.start(), m.end(), "negation", [(match_case(m.group(0), VERB_FLIP[m.group(0).lower()]), 1.0)], "neg")

    def _wordnet(self, t: str, taken: list[tuple[int, int]]):
        for m in re.finditer(r"[A-Za-z]{4,}", t):
            if any(not (m.end() <= s or m.start() >= e) for s, e in taken):
                continue
            w = m.group(0).lower()
            if m.group(0)[0].isupper() and m.start() > 0:
                continue
            try:
                ants = _wn_antonyms(w)
            except Exception:  # noqa: BLE001
                return
            if ants:
                yield Cand(m.start(), m.end(), "wordnet", [(match_case(m.group(0), a), 1.0) for a in ants])

    def _find(self, t: str) -> list[Cand]:
        w = self.weights
        out: list[Cand] = []
        if w["color"]:
            out += list(self._color(t))
        if w["material"]:
            out += list(self._material(t))
        if w["direction"]:
            out += list(self._cardinal(t)) + list(self._corner(t)) + list(DIR_LEX.find(t))
        if w["relation"]:
            out += list(REL_LEX.find(t))
        if w["attribute"]:
            for lex in ATTR_LEXES:
                out += list(lex.find(t))
        if w["degree"]:
            out += list(DEG_LEX.find(t))
        if w["quantity"]:
            out += list(self._number(t)) + list(QUANT_LEX.find(t))
        if w["object"]:
            out += [c for c in self._objects(t) if c.cat == "object" or w["negation"]]
        if w["negation"]:
            out += list(self._negation(t)) + list(NEG_LEX.find(t))
        if self.wordnet:
            out += list(self._wordnet(t, [(c.start, c.end) for c in out]))
        return [c for c in out if w.get(c.cat, 0) > 0]

    # ---------------------------------------------------------------- selection / apply
    @staticmethod
    def _wpick(opts, rng):
        return rng.choices([o for o, _ in opts], [w for _, w in opts])[0]

    def _select(self, t: str, cands: list[Cand], k: int, rng: random.Random):
        chosen: list[tuple[Cand, str]] = []
        cat_use: Counter = Counter()
        tag_use: Counter = Counter()
        for _ in range(k):
            avail = [c for c in cands
                     if all(max(c.end, c.block_end or 0) <= u.start or c.start >= max(u.end, u.block_end or 0)
                            for u, _ in chosen)
                     and not (c.tag and tag_use[c.tag] >= TAG_LIMITS.get(c.tag, 99))]
            if not avail:
                break
            by: dict[str, list[Cand]] = defaultdict(list)
            for c in avail:
                by[c.cat].append(c)
            cats = list(by)
            cat = rng.choices(cats, [self.weights[c] * (self.decay ** cat_use[c]) for c in cats])[0]
            c = rng.choice(by[cat])
            opts = [(o, w) for o, w in c.options if o.lower() != t[c.start:c.end].lower()]
            if not opts:
                cands = [x for x in cands if x is not c]
                continue
            chosen.append((c, self._wpick(opts, rng)))
            cat_use[cat] += 1
            if c.tag:
                tag_use[c.tag] += 1
        return chosen

    @staticmethod
    def _apply(t: str, chosen: list[tuple[Cand, str]]) -> str:
        res, pos = "", 0
        for c, rep in sorted(chosen, key=lambda x: x[0].start):
            res += t[pos:c.start]
            first = re.match(r"[A-Za-z]+", rep)
            m = re.search(r"\b(a|an)(\s+)$", res, re.I)
            if m and first and c.cat in ("color", "material", "attribute", "object", "degree", "wordnet"):
                want = article_for(first.group(0))
                if m.group(1).lower() != want:
                    res = res[:m.start(1)] + match_case(m.group(1), want) + m.group(2)
            if re.search(r"\b(a|an)$", rep, re.I):
                nx = re.match(r"\s*([A-Za-z-]+)", t[c.end:])
                if nx:
                    rep = rep[:-len(re.search(r"(a|an)$", rep, re.I).group(0))] + article_for(nx.group(1))
            res += rep
            pos = c.end
        return res + t[pos:]

    # ---------------------------------------------------------------- public API
    def generate(self, caption: str, mut_count: int = 1, num_sentences: int = 0,
                 rng: Optional[random.Random] = None) -> dict[str, Any]:
        rng = rng or self._rng
        if not isinstance(caption, str) or not caption.strip():
            return {"original": caption, "negative": caption, "mutations": [], "n_mutations": 0}
        parts = _SENT_SPLIT.split(caption)
        sents, seps = parts[0::2], parts[1::2]
        cand_lists = [self._find_c(s) for s in sents]
        active = [i for i, c in enumerate(cand_lists) if c]
        if num_sentences and num_sentences < len(active):
            active = sorted(rng.sample(active, num_sentences))
        muts: list[dict] = []
        new_sents = list(sents)
        for i in active:
            chosen = self._select(sents[i], cand_lists[i], mut_count, rng)
            if not chosen:
                continue
            new_sents[i] = self._apply(sents[i], chosen)
            for c, rep in sorted(chosen, key=lambda x: x[0].start):
                muts.append({"sentence": i, "category": c.cat, "original": sents[i][c.start:c.end], "replacement": rep})
        out = new_sents[0]
        for sep, s in zip(seps, new_sents[1:]):
            out += sep + s
        return {"original": caption, "negative": out, "mutations": muts, "n_mutations": len(muts)}

    def generate_variants(self, caption: str, n: int = 1, **kw) -> list[dict[str, Any]]:
        res, seen = [], set()
        for _ in range(max(n * 6, 1)):
            r = self.generate(caption, **kw)
            if r["negative"] in seen and r["n_mutations"]:
                continue
            seen.add(r["negative"])
            res.append(r)
            if len(res) >= n:
                break
        return res


# =============================================================================
# CLI
# =============================================================================

_G: Optional[NegativeCaptionGenerator] = None
_CFG: dict = {}


def _init(cfg: dict):
    global _G, _CFG
    _CFG = cfg
    _G = NegativeCaptionGenerator(hardness=cfg["hardness"], weights=cfg["weights"],
                                  categories=cfg["categories"], wordnet=cfg["wordnet"])


def _work(task):
    idx, col, caption = task
    rng = random.Random((_CFG["seed"] * 1_000_003 + idx) * 31 + hash(col) % 1000)
    kw = dict(mut_count=_CFG["mut_count"], num_sentences=_CFG["num_sentences"], rng=rng)
    if _CFG["num_variants"] > 1:
        g = _G
        res, seen = [], set()
        for _ in range(_CFG["num_variants"] * 6):
            r = g.generate(caption, **kw)
            if r["negative"] in seen and r["n_mutations"]:
                continue
            seen.add(r["negative"])
            res.append(r)
            if len(res) >= _CFG["num_variants"]:
                break
        return idx, col, res
    return idx, col, [_G.generate(caption, **kw)]


def parse_weights(s: Optional[str]) -> dict:
    if not s:
        return {}
    return {k.strip(): float(v) for k, v in (p.split("=") for p in s.split(","))}


def generate_hard_negatives(
    caption: Optional[str] = None,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    caption_col: str = "dense_caption",
    mut_count: int = 1,
    num_sentences: int = 0,
    num_variants: int = 1,
    hardness: float = 0.5,
    weights: Optional[Union[str, dict]] = None,
    categories: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    wordnet: bool = False,
    seed: int = 0,
    workers: int = 1,
    no_mutation_log: bool = False,
) -> None:
    """
    Algorithmic hard-negative caption generator (aerial).
    Can process either a single caption string, stdin, or an input file (parquet/jsonl/csv).
    """
    # Parse categories list
    if isinstance(categories, str):
        cats = [c.strip() for c in categories.split(",") if c.strip()]
    elif isinstance(categories, list):
        cats = categories
    else:
        cats = list(CATEGORIES)

    # Parse exclusions
    if exclude:
        ex = set(exclude.split(",")) if isinstance(exclude, str) else set(exclude)
        cats = [c for c in cats if c not in ex]

    # Parse weights dictionary if passed as string
    parsed_weights = parse_weights(weights) if isinstance(weights, str) else weights

    cfg = dict(
        hardness=hardness,
        weights=parsed_weights,
        categories=cats,
        wordnet=wordnet,
        mut_count=mut_count,
        num_sentences=num_sentences,
        num_variants=num_variants,
        seed=seed,
    )

    # ---- Single caption / stdin mode ----
    if caption is not None or not input_path:
        _init(cfg)
        lines = [caption] if caption is not None else [l.rstrip("\n") for l in sys.stdin if l.strip()]
        for i, line in enumerate(lines):
            _, _, res = _work((i, "c", line))
            for r in res:
                if caption is not None:
                    print("ORIGINAL:", r["original"])
                    print("NEGATIVE:", r["negative"])
                    for m in r["mutations"]:
                        print(f"  [{m['category']:<9}] s{m['sentence']}: {m['original']!r} -> {m['replacement']!r}")
                    print()
                else:
                    print(json.dumps(r, ensure_ascii=False))
        return r

    # ---- File processing mode ----
    import pandas as pd

    p = input_path
    df = (
        pd.read_parquet(p)
        if p.endswith(".parquet")
        else (pd.read_json(p, lines=True) if p.endswith((".jsonl", ".json")) else pd.read_csv(p))
    )

    cols = [c.strip() for c in caption_col.split(",")]
    tasks = [(i, c, str(v) if v is not None else "") for c in cols for i, v in enumerate(df[c].tolist())]
    results: dict[tuple[int, str], list] = {}

    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
            for idx, col, res in pool.imap_unordered(_work, tasks, chunksize=512):
                results[(idx, col)] = res
    else:
        _init(cfg)
        for t in tasks:
            idx, col, res = _work(t)
            results[(idx, col)] = res

    for c in cols:
        base = c
        neg = [results[(i, c)][0]["negative"] for i in range(len(df))]
        df[f"negative_{base}"] = neg
        df[f"n_mutations_{base}"] = [results[(i, c)][0]["n_mutations"] for i in range(len(df))]
        
        if not no_mutation_log:
            df[f"mutations_{base}"] = [
                json.dumps(results[(i, c)][0]["mutations"], ensure_ascii=False) for i in range(len(df))
            ]
            
        if num_variants > 1:
            df[f"negatives_{base}"] = [
                json.dumps([r["negative"] for r in results[(i, c)]], ensure_ascii=False)
                for i in range(len(df))
            ]

    out = output_path or (p.rsplit(".", 1)[0] + "_neg.parquet")
    if out.endswith(".parquet"):
        df.to_parquet(out, index=False)
    elif out.endswith((".jsonl", ".json")):
        df.to_json(out, orient="records", lines=True)
    else:
        df.to_csv(out, index=False)

    ok = sum(1 for v in results.values() if v[0]["n_mutations"])
    print(f"Done. {len(results)} captions, {ok} mutated -> {out}", file=sys.stderr)