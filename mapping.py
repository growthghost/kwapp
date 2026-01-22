### START mapping.py — PART 1 / 4

import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

STOPWORDS = {
    "the","and","for","to","a","an","of","with","in","on","at","by","from",
    "is","are","be","can","should","how","what","who","where","why","when",
    "me","you","us","our","your","we","it","they","them"
}

# -------------------------------
# Keyword tokenization
# -------------------------------

def _tokenize(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    return {
        t.lower()
        for t in TOKEN_RE.findall(text.lower())
        if t.lower() not in STOPWORDS
    }

# -------------------------------
# Intent slot detection
# -------------------------------

# FIX: \b word-boundary was incorrectly written as \\b (literal backslash+b).
AEO_PAT = re.compile(
    r"^(who|what|when|where|why|how|can|should|is|are|do|does|did|will)\b",
    re.I,
)

AIO_PAT = re.compile(
    r"(what is|definition|meaning|guide|overview|explain|examples|types of)",
    re.I,
)

def keyword_slots(keyword: str) -> List[str]:
    """
    Return ALL slots this keyword can qualify for, ordered by intent strength.
    (We choose the final slot per-URL later based on which caps are still open.)
    """
    if not isinstance(keyword, str):
        return ["SEO"]

    text = keyword.strip().lower()
    slots: List[str] = []

    if AEO_PAT.search(text):
        slots.append("AEO")
    if AIO_PAT.search(text):
        slots.append("AIO")

    slots.append("SEO")  # always allow SEO fallback
    # de-dupe while preserving order
    out: List[str] = []
    for s in slots:
        if s not in out:
            out.append(s)
    return out

### END mapping.py — PART 1 / 4

### START mapping.py — PART 2 / 4

# -------------------------------
# Page token construction
# -------------------------------

def build_page_tokens(page_signals_by_url: Dict[str, Dict]) -> Tuple[
    List[str],                 # page_urls
    List[Set[str]],             # page_token_sets
    Dict[str, bool],            # veo_ready flags
]:
    """
    Build clean, structural-only token sets per page.
    """
    page_urls: List[str] = []
    page_token_sets: List[Set[str]] = []
    veo_ready: Dict[str, bool] = {}

    for url, sig in page_signals_by_url.items():
        tokens = set()

        # STRICTLY structural signals from crawler
        tokens |= set(sig.get("slug_tokens", []))
        tokens |= set(sig.get("title_tokens", []))
        tokens |= set(sig.get("h1_tokens", []))
        tokens |= set(sig.get("h2h3_tokens", []))
        tokens |= set(sig.get("meta_tokens", []))

        if not tokens:
            continue

        page_urls.append(url)
        page_token_sets.append(tokens)
        veo_ready[url] = bool(sig.get("veo_ready", False))

    return page_urls, page_token_sets, veo_ready


# -------------------------------
# Inverted index
# -------------------------------

def build_inverted_index(page_token_sets: List[Set[str]]) -> Dict[str, Set[int]]:
    """
    token -> set(page indices)
    """
    index: Dict[str, Set[int]] = defaultdict(set)
    for i, toks in enumerate(page_token_sets):
        for t in toks:
            index[t].add(i)
    return index

### END mapping.py — PART 2 / 4

### START mapping.py — PART 3 / 4

# -------------------------------
# Candidate scoring
# -------------------------------

MIN_OVERLAP = 2  # NEW: require at least 2 shared tokens to allow mapping

def score_keyword_against_pages(
    kw_tokens: Set[str],
    page_token_sets: List[Set[str]],
    inv_index: Dict[str, Set[int]],
) -> List[Tuple[int, float, int]]:
    """
    Returns list of:
      (page_index, coverage, overlap)
    """
    if not kw_tokens:
        return []

    # Candidate pages via inverted index
    candidate_pages: Set[int] = set()
    for t in kw_tokens:
        candidate_pages |= inv_index.get(t, set())

    scored: List[Tuple[int, float, int]] = []

    for pi in candidate_pages:
        page_tokens = page_token_sets[pi]
        overlap_tokens = kw_tokens & page_tokens
        overlap = len(overlap_tokens)

        # NEW: block weak 1-token matches
        if overlap < MIN_OVERLAP:
            continue

        coverage = overlap / max(1, len(kw_tokens))
        scored.append((pi, coverage, overlap))

    return scored


def rank_candidates(
    candidates: List[Tuple[int, float, int]],
    page_urls: List[str],
) -> List[int]:
    """
    Deterministic ranking:
    1. coverage DESC
    2. overlap DESC
    3. URL depth ASC
    4. URL string ASC
    """
    def url_depth(u: str) -> int:
        return len([p for p in u.split("/") if p])

    ranked = sorted(
        candidates,
        key=lambda x: (
            -x[1],                    # coverage
            -x[2],                    # overlap
            url_depth(page_urls[x[0]]),
            page_urls[x[0]],
        )
    )

    return [pi for pi, _, _ in ranked]

### END mapping.py — PART 3 / 4

### START mapping.py — PART 4 / 4

# -------------------------------
# Slot enforcement + assignment
# -------------------------------

SLOT_LIMITS = {
    "SEO": 2,
    "AIO": 1,
    "AEO": 1,
}

def run_mapping(
    df: pd.DataFrame,
    page_signals_by_url: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Deterministic keyword → URL mapping.
    - Uses crawler signals ONLY
    - Enforces slot caps per URL (2 SEO, 1 AIO, 1 AEO)
    - Chooses the best slot PER URL when a keyword qualifies for multiple
    """

    # Resolve required columns
    def _resolve(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        raise ValueError(f"Missing required column: one of {colnames}")

    KW_COL = _resolve(["Keyword", "keyword", "query", "term"])
    ELIGIBLE_COL = _resolve(["Eligible", "eligible"])

    # Output column (align with app export)
    OUT_COL = "Map URL" if "Map URL" in df.columns else "Mapped URL"
    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    # Build page token structures
    page_urls, page_token_sets, veo_ready = build_page_tokens(page_signals_by_url)
    if not page_urls:
        return df

    inv_index = build_inverted_index(page_token_sets)

    # Per-URL slot usage
    slot_usage: Dict[str, Dict[str, int]] = {
        u: {"SEO": 0, "AIO": 0, "AEO": 0}
        for u in page_urls
    }

    def _remaining(u: str, slot: str) -> int:
        return max(0, SLOT_LIMITS[slot] - slot_usage[u][slot])

    def _choose_slot_for_url(u: str, possible_slots: List[str]) -> str:
        """
        If keyword qualifies for multiple slots, use the slot this URL still needs most.
        """
        # Only consider slots that still have room
        candidates = [s for s in possible_slots if _remaining(u, s) > 0]
        if not candidates:
            return ""  # none available

        # Prefer the slot with the highest remaining capacity on that URL.
        # Tie-break keeps original order from possible_slots (AEO, AIO, SEO).
        best = max(candidates, key=lambda s: (_remaining(u, s), -possible_slots.index(s)))
        return best

    # Iterate keywords in row order (CSV already sorted upstream)
    for idx, row in df.iterrows():
        if row.get(ELIGIBLE_COL) != "Yes":
            continue

        keyword = str(row.get(KW_COL, "")).strip()
        if not keyword:
            continue

        kw_tokens = _tokenize(keyword)
        if not kw_tokens:
            continue

        possible_slots = keyword_slots(keyword)  # NEW: can be ["AEO","AIO","SEO"] etc.

        # Score candidates
        scored = score_keyword_against_pages(
            kw_tokens=kw_tokens,
            page_token_sets=page_token_sets,
            inv_index=inv_index,
        )
        if not scored:
            continue

        ranked_pages = rank_candidates(scored, page_urls)

        # Try to place on best page, picking best slot for that page
        for pi in ranked_pages:
            url = page_urls[pi]
            chosen_slot = _choose_slot_for_url(url, possible_slots)
            if not chosen_slot:
                continue

            df.at[idx, OUT_COL] = url
            slot_usage[url][chosen_slot] += 1
            break

    return df

### END mapping.py — PART 4 / 4

