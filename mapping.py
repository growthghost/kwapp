# mapping.py
# Hybrid Option D: Weighted keyword-to-URL mapping with density, stopword learning, and phrase/multi-token checks

import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Set

# ---------- Weights ----------
WEIGHTS = {
    "slug": 5,
    "title": 4,
    "h1": 3,
    "meta": 2,
    "body": 1
}

# ---------- Per-page caps ----------
PAGE_CAPS = {
    "SEO": 2,
    "AIO": 1,
    "VEO": 1
}

# ---------- Hybrid thresholds ----------
MIN_SCORE = 5           # must reach this weighted score
DENSITY_THRESHOLD = 0.3 # % of keyword tokens required to overlap
STOPWORD_FREQ = 0.5     # token appears in >50% of pages → boilerplate
MIN_TOKEN_OVERLAP = 2   # require at least 2 meaningful token overlaps

TOKENIZER = re.compile(r"[a-zA-Z0-9]+")


# ---------- Token utilities ----------
def tokenize(text: str) -> List[str]:
    return TOKENIZER.findall(text.lower()) if text else []


def build_stopwords(page_signals_by_url: Dict) -> Set[str]:
    """
    Learn boilerplate tokens that appear on too many pages.
    """
    doc_freq = Counter()
    total_docs = len(page_signals_by_url)

    for signals in page_signals_by_url.values():
        seen = set()
        for field in ["slug", "title", "h1", "meta", "body"]:
            seen.update(tokenize(signals.get(field, "")))
        for token in seen:
            doc_freq[token] += 1

    return {tok for tok, freq in doc_freq.items() if freq / total_docs >= STOPWORD_FREQ}


def url_depth(url: str) -> int:
    """
    Calculate the depth of a URL (number of path segments).
    Example: https://site.com/a/b/c -> depth = 3
    """
    parts = url.strip("/").split("/")
    return len(parts) - 1 if parts[0].startswith("http") else len(parts)


# ---------- Core mapping logic (Hybrid D) ----------
def weighted_map_keywords(df: pd.DataFrame, page_signals_by_url: Dict) -> List[Dict]:
    """
    Maps each keyword in df to the best URL using Hybrid D rules.

    Inputs:
        df: pandas DataFrame with keywords, category, etc.
        page_signals_by_url: dict {url: {slug, title, h1, meta, body}}

    Output:
        mapping_results: list of dicts with:
            keyword, category, chosen_url, weighted_score, reasons
    """
    stopwords = build_stopwords(page_signals_by_url)
    assigned_counts = {url: {"SEO": 0, "AIO": 0, "VEO": 0} for url in page_signals_by_url}
    results = []

    for _, row in df.iterrows():
        kw = row.get("Keyword", "")
        category = row.get("Category", "SEO")
        kw_tokens = tokenize(kw)
        meaningful_kw_tokens = [t for t in kw_tokens if t not in stopwords]

        candidate_scores = []
        for url, signals in page_signals_by_url.items():
            score = 0
            reasons = []
            all_overlaps = set()

            # collect signal text
            combined_fields = []
            for field, text in signals.items():
                overlaps = set(tokenize(kw)) & set(tokenize(text))
                if overlaps:
                    weight = WEIGHTS.get(field, 0)
                    score += len(overlaps) * weight
                    reasons.append(f"{field} match: {', '.join(overlaps)} (x{len(overlaps)} • w{weight})")
                    all_overlaps |= overlaps
                combined_fields.append(text.lower())
            page_text = " ".join(combined_fields)

            # --- Hybrid Rule 1: must hit weighted threshold ---
            if score < MIN_SCORE:
                continue

            # --- Hybrid Rule 2: density check ---
            overlap_meaningful = [t for t in meaningful_kw_tokens if t in page_text]
            coverage = len(overlap_meaningful) / len(meaningful_kw_tokens) if meaningful_kw_tokens else 0
            if coverage < DENSITY_THRESHOLD:
                continue

            # --- Hybrid Rule 3: phrase or multi-token ---
            phrase_match = any(" ".join(meaningful_kw_tokens) in f.lower() for f in combined_fields)
            multi_token_match = len(overlap_meaningful) >= MIN_TOKEN_OVERLAP
            if not (phrase_match or multi_token_match):
                continue

            candidate_scores.append((url, score, reasons))

        if candidate_scores:
            # Sort by score (desc), then depth (shallowest wins), then URL length (shorter wins)
            candidate_scores.sort(
                key=lambda x: (x[1], -url_depth(x[0]), -len(x[0])),
                reverse=True
            )
            for url, score, reasons in candidate_scores:
                if assigned_counts[url][category] < PAGE_CAPS.get(category, 99):
                    assigned_counts[url][category] += 1
                    results.append({
                        "keyword": kw,
                        "category": category,
                        "chosen_url": url,
                        "weighted_score": score,
                        "reasons": "; ".join(reasons)
                    })
                    break
        else:
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": None,
                "weighted_score": 0,
                "reasons": "No match (failed Hybrid D rules)"
            })

    return results
