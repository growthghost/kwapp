# mapping.py
# Structural scoring model – slug/title/H1/H2/H3/meta with phrase & coverage awareness

import pandas as pd
import re
from typing import Dict, List, Set, Tuple

# ---------- Structural Weights ----------
WEIGHTS = {
    "slug": 7,   # URL slug is strong intent signal
    "title": 6,  # Page title
    "h1": 6,     # Primary heading
    "h2h3": 5,   # Supporting headings
    "meta": 3    # Meta description (weaker, but still useful)
}

# ---------- Threshold and Bonuses ----------
THRESHOLD = 3  # must reach or exceed this to map

BONUS_URL_TITLE = 5
BONUS_TITLE_H1 = 10
BONUS_H2H3_ANY = 15

# Phrase / coverage bonuses
PHRASE_BONUS_STRONG = 40   # full keyword phrase appears in a field
COVERAGE_SCALE       = 12  # boosts pages that cover a higher % of keyword tokens

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
    """
    Score a (keyword, page) pair using:
    - token overlap per field (slug/title/H1/H2/H3/meta)
    - coverage of keyword tokens
    - exact-phrase detection inside each field
    - synergy bonuses between fields
    """
    kw_tokens = _token_set(keyword)
    if not kw_tokens:
        return 0, ["No keyword tokens"]

    total_kw_tokens = len(kw_tokens)
    kw_phrase = " ".join(tokenize(keyword))  # normalized phrase form

    score = 0
    reasons: List[str] = []
    matched_fields = set()

    # Pull fields, normalizing slug to words
    slug_raw = signals.get("slug", "") or ""
    slug = slug_raw.replace("-", " ").replace("_", " ")
    title = signals.get("title", "") or ""
    h1 = signals.get("h1", "") or ""
    h2 = signals.get("h2", "") or ""
    h3 = signals.get("h3", "") or ""
    meta = signals.get("meta", "") or ""

    def _score_field(field_name: str, text: str):
        nonlocal score, reasons, matched_fields

        if not text:
            return

        field_tokens = _token_set(text)
        overlap = kw_tokens & field_tokens
        if not overlap:
            return

        base = WEIGHTS.get(field_name, 0)
        overlap_count = len(overlap)
        coverage = overlap_count / max(1, total_kw_tokens)

        # Base score from overlap count
        field_score = base * overlap_count

        # Coverage bonus – pages matching more of the keyword get extra credit
        field_score += int(round(COVERAGE_SCALE * coverage))

        # Phrase bonus – normalized keyword phrase appears inside the field
        text_norm = " ".join(tokenize(text))
        if kw_phrase and kw_phrase in text_norm:
            field_score += PHRASE_BONUS_STRONG
            phrase_note = " + phrase match"
        else:
            phrase_note = ""

        score += field_score
        matched_fields.add(field_name)

        overlap_list = ", ".join(sorted(overlap))
        reasons.append(
            f"{field_name.upper()} match: {overlap_list} "
            f"(overlap={overlap_count}, coverage={coverage:.2f}{phrase_note}, +{field_score})"
        )

    # Score individual fields
    _score_field("slug", slug)
    _score_field("title", title)
    _score_field("h1", h1)
    _score_field("h2h3", f"{h2} {h3}")
    _score_field("meta", meta)

    # 5️⃣ Synergy Bonuses (unchanged logic, now on top of richer base scoring)
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
    """
    For each keyword row in df, choose the best URL from page_signals_by_url
    using structural_score(). Returns a list of dicts:

    {
        "keyword": str,
        "category": str,
        "chosen_url": Optional[str],
        "weighted_score": int,
        "reasons": str
    }
    """
    results: List[Dict] = []

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

        candidate_scores: List[Tuple[str, int, List[str]]] = []

        for url, signals in page_signals_by_url.items():
            score, reasons = structural_score(kw, signals)
            if score >= THRESHOLD:
                candidate_scores.append((url, score, reasons))

        if candidate_scores:
            # Highest score wins
            # Tie-break:
            #   1) shallower URL depth
            #   2) shorter URL
            candidate_scores.sort(
                key=lambda x: (x[1], -url_depth(x[0]), -len(x[0])),
                reverse=True
            )
            best_url, best_score, best_reasons = candidate_scores[0]
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": best_url,
                "weighted_score": int(best_score),
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
