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

AEO_PAT = re.compile(
    r"^(who|what|when|where|why|how|can|should|is|are|do|does|did|will)\\b",
    re.I,
)

AIO_PAT = re.compile(
    r"(what is|definition|meaning|guide|overview|explain|examples|types of)",
    re.I,
)

def keyword_slot(keyword: str) -> str:
    if not isinstance(keyword, str):
        return "SEO"
    text = keyword.strip().lower()
    if AEO_PAT.search(text):
        return "AEO"
    if AIO_PAT.search(text):
        return "AIO"
    return "SEO"

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
    Deterministic keyword → URL mapping (URL-FIRST quota fill).
    - Uses crawler signals ONLY
    - Enforces per-URL quotas: 2 SEO, 1 AIO, 1 AEO
    - Enforces global keyword uniqueness
    """

    # Resolve required columns
    def _resolve(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        raise ValueError(f"Missing required column: one of {colnames}")

    KW_COL = _resolve(["Keyword", "keyword", "query", "term"])
    ELIGIBLE_COL = _resolve(["Eligible", "eligible"])

    # Prepare output column (single authority)
    if "Map URL" not in df.columns:
        df["Map URL"] = ""
    else:
        # Clear stale mappings before recompute
        df["Map URL"] = ""

    # Build page token structures
    page_urls, page_token_sets, _veo_ready = build_page_tokens(page_signals_by_url)
    if not page_urls:
        return df

    inv_index = build_inverted_index(page_token_sets)

    # Deterministic row position (reflects upstream sorting)
    row_pos: Dict[int, int] = {idx: pos for pos, idx in enumerate(df.index)}

    # Per-page candidate buckets: url -> slot -> list[(coverage, overlap, row_pos, idx)]
    per_page: Dict[str, Dict[str, List[Tuple[float, int, int, int]]]] = {
        u: {"SEO": [], "AIO": [], "AEO": []} for u in page_urls
    }

    # Build candidates (keyword-first gathering, URL-first assigning)
    for idx, row in df.iterrows():
        if row.get(ELIGIBLE_COL) != "Yes":
            continue

        keyword = str(row.get(KW_COL, "")).strip()
        if not keyword:
            continue

        slot = keyword_slot(keyword)
        kw_tokens = _tokenize(keyword)
        if not kw_tokens:
            continue

        # Candidate pages via inverted index (fast)
        candidate_pages: Set[int] = set()
        for t in kw_tokens:
            candidate_pages |= inv_index.get(t, set())
        if not candidate_pages:
            continue

        for pi in candidate_pages:
            overlap_tokens = kw_tokens & page_token_sets[pi]
            overlap = len(overlap_tokens)

            # Respect MIN_OVERLAP rule from PART 3
            if overlap < MIN_OVERLAP:
                continue

            coverage = overlap / max(1, len(kw_tokens))
            url = page_urls[pi]
            per_page[url][slot].append((coverage, overlap, row_pos.get(idx, 10**9), idx))

    # Sort candidate lists (best first, deterministic)
    for url in page_urls:
        for slot in ("AEO", "AIO", "SEO"):
            per_page[url][slot].sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))

    # URL-FIRST quota fill
    used_keywords: Set[int] = set()

    for url in page_urls:
        # Fill AEO then AIO then SEO (so each page gets its answer-style slots first)
        for slot in ("AEO", "AIO", "SEO"):
            need = SLOT_LIMITS[slot]
            filled = 0

            for (_cov, _ov, _pos, idx) in per_page[url][slot]:
                if idx in used_keywords:
                    continue
                df.at[idx, "Map URL"] = url
                used_keywords.add(idx)
                filled += 1
                if filled >= need:
                    break

    return df

### END mapping.py — PART 4 / 4

