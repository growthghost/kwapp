# mapping.py
# Weighted keyword-to-URL mapping logic (cleaned + hybrid rule)

import pandas as pd
import re
from urllib.parse import urlparse
from typing import Dict, List

# ---------- Weights ----------
WEIGHTS = {
    "slug": 5,
    "title": 4,
    "h1": 3,
    "meta": 2,
    "body": 1,
}

# ---------- Per-page caps ----------
PAGE_CAPS = {
    "SEO": 2,
    "AIO": 1,
    "VEO": 1,
}

# ---------- Tokenizer ----------
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Split text into lowercase alphanumeric tokens."""
    if not text:
        return []
    return [t.lower() for t in WORD_RE.findall(text)]


def overlap_count(keyword: str, text: str) -> int:
    """Count overlapping tokens between keyword and text."""
    return len(set(tokenize(keyword)) & set(tokenize(text)))


def url_depth(url: str) -> int:
    """Calculate depth of a URL based on path segments."""
    path = urlparse(url).path.strip("/")
    if not path:
        return 0
    return len(path.split("/"))


def weighted_map_keywords(df: pd.DataFrame, page_signals_by_url: Dict) -> List[Dict]:
    """
    Maps each keyword in df to the best URL using weighted signals.

    Inputs:
        df: pandas DataFrame with keywords and categories
        page_signals_by_url: dict {url: {slug, title, h1, meta, body}}

    Output:
        mapping_results: list of dicts with keyword, category, chosen_url, score, reasons
    """
    # Track how many keywords have been assigned per page/category
    assigned_counts = {url: {"SEO": 0, "AIO": 0, "VEO": 0} for url in page_signals_by_url}
    results = []

    for _, row in df.iterrows():
        kw = str(row.get("Keyword", "")).strip()
        category = str(row.get("Category", "SEO")).strip() or "SEO"

        candidate_scores = []
        for url, signals in page_signals_by_url.items():
            score = 0
            reasons = []
            strong_field_match = False

            # Score overlaps for each field
            for field, text in signals.items():
                matches = overlap_count(kw, text)
                if matches > 0:
                    weight = WEIGHTS.get(field, 0)
                    score += matches * weight
                    reasons.append(f"{field} match x{matches} (weight {weight})")
                    if field in ["slug", "title", "h1"]:
                        strong_field_match = True

            # Hybrid rule:
            # Require 1 strong-field match AND (≥2 tokens or score ≥ 6)
            kw_tokens = set(tokenize(kw))
            page_tokens = set(tokenize(" ".join(signals.values())))
            token_overlap = len(kw_tokens & page_tokens)

            if strong_field_match and (token_overlap >= 2 or score >= 6):
                candidate_scores.append((url, score, reasons))

        if candidate_scores:
            # Sort by score desc, then URL depth asc, then URL length asc
            candidate_scores.sort(
                key=lambda x: (x[1], -url_depth(x[0]), -len(x[0])),
                reverse=True,
            )
            for url, score, reasons in candidate_scores:
                # Enforce per-page caps
                if assigned_counts[url][category] < PAGE_CAPS.get(category, 99):
                    assigned_counts[url][category] += 1
                    results.append(
                        {
                            "keyword": kw,
                            "category": category,
                            "chosen_url": url,
                            "weighted_score": score,
                            "reasons": "; ".join(reasons),
                        }
                    )
                    break
        else:
            # No strong matches
            results.append(
                {
                    "keyword": kw,
                    "category": category,
                    "chosen_url": None,
                    "weighted_score": 0,
                    "reasons": "No strong matches (failed thresholds)",
                }
            )

    return results
