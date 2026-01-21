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
    Deterministic keyword → URL mapping.
    - Uses crawler signals ONLY
    - Enforces slot caps per URL
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

    # Output column (prefer Map URL if app already created it)
    OUT_COL = "Map URL" if "Map URL" in df.columns else "Mapped URL"
    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    # Build page token structures (all structural tokens)
    page_urls, page_token_sets, veo_ready = build_page_tokens(page_signals_by_url)
    if not page_urls:
        return df

    inv_index = build_inverted_index(page_token_sets)

    # Build STRONG token sets per page (slug/title/h1 only) for fallback
    strong_token_sets: List[Set[str]] = []
    for u in page_urls:
        sig = page_signals_by_url.get(u, {}) or {}
        strong = set()
        strong |= set(sig.get("slug_tokens", []))
        strong |= set(sig.get("title_tokens", []))
        strong |= set(sig.get("h1_tokens", []))
        strong_token_sets.append(strong)

    inv_index_strong = build_inverted_index(strong_token_sets)

    # Per-URL slot usage
    slot_usage: Dict[str, Dict[str, int]] = {
        u: {"SEO": 0, "AIO": 0, "AEO": 0}
        for u in page_urls
    }

    # Global keyword usage
    used_keywords: Set[int] = set()

    # Iterate keywords in row order (CSV already sorted upstream)
    for idx, row in df.iterrows():
        if row.get(ELIGIBLE_COL) != "Yes":
            continue
        if idx in used_keywords:
            continue

        keyword = str(row.get(KW_COL, "")).strip()
        if not keyword:
            continue

        slot = keyword_slot(keyword)
        kw_tokens = _tokenize(keyword)

        if not kw_tokens:
            continue

        # ---------- Pass 1: strict overlap against ALL structural tokens ----------
        scored = score_keyword_against_pages(
            kw_tokens=kw_tokens,
            page_token_sets=page_token_sets,
            inv_index=inv_index,
        )

        ranked_pages: List[int] = []
        if scored:
            ranked_pages = rank_candidates(scored, page_urls)

        placed = False

        for pi in ranked_pages:
            url = page_urls[pi]

            # Slot capacity check
            if slot_usage[url][slot] >= SLOT_LIMITS[slot]:
                continue

            # Assign
            df.at[idx, OUT_COL] = url
            slot_usage[url][slot] += 1
            used_keywords.add(idx)
            placed = True
            break

        # ---------- Pass 2 (FALLBACK): allow 1-token match ONLY in slug/title/h1 ----------
        if placed:
            continue

        # Build candidates using strong inverted index
        candidate_pages: Set[int] = set()
        for t in kw_tokens:
            candidate_pages |= inv_index_strong.get(t, set())

        if not candidate_pages:
            continue

        fallback_scored: List[Tuple[int, float, int]] = []
        for pi in candidate_pages:
            overlap_tokens = kw_tokens & strong_token_sets[pi]
            if not overlap_tokens:
                continue

            overlap = len(overlap_tokens)          # will be >= 1
            coverage = overlap / max(1, len(kw_tokens))
            fallback_scored.append((pi, coverage, overlap))

        if not fallback_scored:
            continue

        ranked_pages_fb = rank_candidates(fallback_scored, page_urls)

        for pi in ranked_pages_fb:
            url = page_urls[pi]
            if slot_usage[url][slot] >= SLOT_LIMITS[slot]:
                continue

            df.at[idx, OUT_COL] = url
            slot_usage[url][slot] += SLOT_LIMITS.get(slot, 1) and 1
            used_keywords.add(idx)
            break

    return df

### END mapping.py — PART 4 / 4

