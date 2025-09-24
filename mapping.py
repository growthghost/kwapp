# mapping.py
# This file contains the weighted keyword-to-URL mapping logic.

import pandas as pd
from typing import Dict, List

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

# ---------- Thresholds ----------
MIN_SCORE = 5  # must reach this weighted score to be mapped
REQUIRE_STRONG_FIELD = True  # must match slug OR title OR h1

def overlap_count(keyword: str, text: str) -> int:
    """
    Count the number of overlapping tokens between a keyword and some text.
    """
    if not text:
        return 0
    kw_tokens = set(keyword.lower().split())
    txt_tokens = set(text.lower().split())
    return len(kw_tokens & txt_tokens)

def url_depth(url: str) -> int:
    """
    Calculate the depth of a URL (number of path segments).
    Example: https://site.com/a/b/c -> depth = 3
    """
    parts = url.strip("/").split("/")
    return len(parts) - 1 if parts[0].startswith("http") else len(parts)

def weighted_map_keywords(df: pd.DataFrame, page_signals_by_url: Dict) -> List[Dict]:
    """
    Maps each keyword in df to the best URL using weighted signals.

    Inputs:
        df: pandas DataFrame with keywords, category, etc.
        page_signals_by_url: dict {url: {slug, title, h1, meta, body}}

    Output:
        mapping_results: list of dicts with:
            keyword, category, chosen_url, weighted_score, reasons
    """

    # Track assigned counts per page/category
    assigned_counts = {url: {"SEO": 0, "AIO": 0, "VEO": 0} for url in page_signals_by_url}
    results = []

    for _, row in df.iterrows():
        kw = row.get("Keyword", "")
        category = row.get("Category", "SEO")

        candidate_scores = []
        for url, signals in page_signals_by_url.items():
            score = 0
            reasons = []
            strong_field_match = False  # track slug/title/h1 matches

            for field, text in signals.items():
                matches = overlap_count(kw, text)
                if matches > 0:
                    weight = WEIGHTS.get(field, 0)
                    score += matches * weight
                    reasons.append(f"{field} match x{matches} (weight {weight})")
                    if field in ["slug", "title", "h1"]:
                        strong_field_match = True

            # Apply stricter rules
            if score >= MIN_SCORE and (not REQUIRE_STRONG_FIELD or strong_field_match):
                candidate_scores.append((url, score, reasons))

        if candidate_scores:
            # Sort by score (desc), then depth (shallowest wins), then URL length (shorter wins)
            candidate_scores.sort(
                key=lambda x: (x[1], -url_depth(x[0]), -len(x[0])),
                reverse=True
            )
            for url, score, reasons in candidate_scores:
                # Enforce per-page caps
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
            # No matches → unmapped
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": None,
                "weighted_score": 0,
                "reasons": "No strong matches (failed thresholds)"
            })

    return results
