# mapping.py
# Weighted keyword-to-URL mapping logic with soft restriction + 2-token fallback

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


def overlap_tokens(keyword: str, text: str) -> set:
    """
    Return the set of overlapping tokens between a keyword and some text.
    """
    if not text:
        return set()
    kw_tokens = set(keyword.lower().split())
    txt_tokens = set(text.lower().split())
    return kw_tokens & txt_tokens


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
            strong_field_match = False
            all_overlaps = set()

            for field, text in signals.items():
                overlaps = overlap_tokens(kw, text)
                if overlaps:
                    weight = WEIGHTS.get(field, 0)
                    score += len(overlaps) * weight
                    reasons.append(f"{field} match: {', '.join(overlaps)} (x{len(overlaps)} • w{weight})")
                    all_overlaps |= overlaps
                    if field in ["slug", "title", "h1"]:
                        strong_field_match = True

            # Rule 1: must hit threshold
            # Rule 2: if slug/title/h1 exist, require a match in at least one of them
            # Rule 3: if no strong field match, require >=2 overlaps anywhere
            if score >= MIN_SCORE and (strong_field_match or len(all_overlaps) >= 2):
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
