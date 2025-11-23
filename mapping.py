# mapping.py
# Structural scoring model – slug/title/H1/H2/H3 only (no generic penalty)

import pandas as pd
import re
from typing import Dict, List, Set, Tuple

# ---------- Structural Weights ----------
WEIGHTS = {
    "slug": 6,
    "title": 5,
    "h1": 5,
    "h2h3": 5
}

# ---------- Threshold and Bonuses ----------
THRESHOLD = 3  # must reach or exceed this to map
BONUS_URL_TITLE = 5
BONUS_TITLE_H1 = 10
BONUS_H2H3_ANY = 15

TOKENIZER = re.compile(r"[a-zA-Z0-9]+")


# ---------- Tokenization ----------
def tokenize(text: str) -> List[str]:
    return TOKENIZER.findall(text.lower()) if text else []


def _token_set(text: str) -> Set[str]:
    return set(tokenize(text))


# ---------- URL depth helper ----------
def url_depth(url: str) -> int:
    parts = [p for p in url.strip("/").split("/") if p]
    return len(parts)


# ---------- Structural Scoring ----------
def structural_score(keyword: str, signals: Dict[str, str]) -> Tuple[int, List[str]]:
    kw_tokens = _token_set(keyword)
    score = 0
    reasons = []
    matched_fields = set()

    slug = signals.get("slug", "") or ""
    title = signals.get("title", "") or ""
    h1 = signals.get("h1", "") or ""
    h2 = signals.get("h2", "") or ""
    h3 = signals.get("h3", "") or ""

    # 1️⃣ URL / Slug
    slug_tokens = _token_set(slug)
    slug_overlap = kw_tokens & slug_tokens
    if slug_overlap:
        base = WEIGHTS["slug"]
        score += base * len(slug_overlap)
        reasons.append(f"Slug match: {', '.join(slug_overlap)} (+{base * len(slug_overlap)})")
        matched_fields.add("slug")

    # 2️⃣ Title
    title_tokens = _token_set(title)
    title_overlap = kw_tokens & title_tokens
    if title_overlap:
        base = WEIGHTS["title"]
        score += base * len(title_overlap)
        reasons.append(f"Title match: {', '.join(title_overlap)} (+{base * len(title_overlap)})")
        matched_fields.add("title")

    # 3️⃣ H1
    h1_tokens = _token_set(h1)
    h1_overlap = kw_tokens & h1_tokens
    if h1_overlap:
        base = WEIGHTS["h1"]
        score += base * len(h1_overlap)
        reasons.append(f"H1 match: {', '.join(h1_overlap)} (+{base * len(h1_overlap)})")
        matched_fields.add("h1")

    # 4️⃣ H2 / H3
    h2h3_tokens = _token_set(h2) | _token_set(h3)
    h2h3_overlap = kw_tokens & h2h3_tokens
    if h2h3_overlap:
        base = WEIGHTS["h2h3"]
        score += base * len(h2h3_overlap)
        reasons.append(f"H2/H3 match: {', '.join(h2h3_overlap)} (+{base * len(h2h3_overlap)})")
        matched_fields.add("h2h3")

    # 5️⃣ Synergy Bonuses
    if "slug" in matched_fields and "title" in matched_fields:
        score += BONUS_URL_TITLE
        reasons.append(f"Bonus: URL+Title synergy (+{BONUS_URL_TITLE})")
    if "title" in matched_fields and "h1" in matched_fields:
        score += BONUS_TITLE_H1
        reasons.append(f"Bonus: Title+H1 synergy (+{BONUS_TITLE_H1})")
    if "h2h3" in matched_fields and ({"slug", "title", "h1"} & matched_fields):
        score += BONUS_H2H3_ANY
        reasons.append(f"Bonus: H2/H3 alignment (+{BONUS_H2H3_ANY})")

    return int(round(score)), reasons


# ---------- Mapping Function ----------
def weighted_map_keywords(df: pd.DataFrame, page_signals_by_url: Dict) -> List[Dict]:
    results = []
    for _, row in df.iterrows():
        kw = str(row.get("Keyword", "")).strip()
        category = str(row.get("Category", "SEO")).strip()
        if not kw:
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": None,
                "weighted_score": 0,
                "reasons": "Empty keyword"
            })
            continue

        candidate_scores = []
        for url, signals in page_signals_by_url.items():
            score, reasons = structural_score(kw, signals)
            if score >= THRESHOLD:
                candidate_scores.append((url, score, reasons))

        if candidate_scores:
            # Highest score wins, tie-break = shallower URL depth
            candidate_scores.sort(key=lambda x: (x[1], -url_depth(x[0]), -len(x[0])), reverse=True)
            best_url, best_score, best_reasons = candidate_scores[0]
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": best_url,
                "weighted_score": best_score,
                "reasons": "; ".join(best_reasons)
            })
        else:
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": None,
                "weighted_score": 0,
                "reasons": "No match (below threshold or no structural alignment)"
            })

    return results
