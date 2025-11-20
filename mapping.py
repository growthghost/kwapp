# mapping.py
# Elite Structural SEO Scoring Model — No body text. Only Slug/Title/H1/H2/H3.
# Exact token match only. Synergy bonuses. Threshold >=5.

import re
from urllib.parse import urlparse
from typing import Dict, List, Any, Tuple
import pandas as pd

# Informational intent markers in slugs (e.g. "how-to", "best-", "guide")
INTENT_MARKERS = [
    "how-to", "best-", "top-", "guide", "tutorial", "ultimate", "complete",
    "step-by-step", "vs-", "review", "comparison", "what-is", "why-", "how-"
]

# Common short verbs — helps classify tokens
COMMON_VERBS = {
    "be", "have", "do", "go", "get", "make", "know", "think", "take",
    "see", "come", "want", "use", "find", "give", "tell", "work", "call",
    "buy", "sell", "need", "ask", "try", "help", "learn", "start", "stop"
}

def classify_token(token: str) -> str:
    """Simple, fast POS-like classification for keywords"""
    token = token.lower().strip()
    if len(token) <= 2 or token in COMMON_VERBS:
        return "verb"
    if len(token) == 3:
        return "adj"   # most 3-letter keyword words are adjectives
    return "noun"      # length >=4 → treat as noun (most accurate for SEO keywords)


def has_intent_marker(slug: str) -> bool:
    """Detect informational/commercial intent in slug"""
    slug_clean = slug.lower().replace("_", "-")
    return any(marker in slug_clean for marker in INTENT_MARKERS)


def calculate_structural_score(keyword: str, signals: Dict[str, str]) -> Tuple[float, str]:
    """Core scoring logic — exactly as you designed"""
    if not keyword.strip():
        return 0.0, "empty keyword"

    tokens = [t.strip() for t in keyword.lower().split() if t.strip()]
    if not tokens:
        return 0.0, "no tokens"

    score = 0.0
    details = []

    slug = signals.get("slug", "")
    title = signals.get("title", "")
    h1 = signals.get("h1", "")
    h2 = signals.get("h2", "")
    h3 = signals.get("h3", "")
    h2_h3 = " ".join([h2, h3]).lower()

    # Track which major sections matched
    slug_hit = False
    title_hit = False
    h1_hit = False
    h2h3_hit = False

    # === 1. SLUG / URL ===
    slug_lower = slug.lower().replace("_", " ")
    for token in tokens:
        if token in slug_lower and classify_token(token) == "noun":
            score += 6
            details.append("slug(noun)")
            slug_hit = True
    if has_intent_marker(slug):
        score += 2
        details.append("slug(intent)")

    # === 2. TITLE ===
    title_lower = title.lower()
    title_nouns = sum(1 for t in tokens if classify_token(t) == "noun" and t in title_lower)
    title_adjs  = sum(1 for t in tokens if classify_token(t) == "adj"  and t in title_lower)
    title_verbs = sum(1 for t in tokens if classify_token(t) == "verb" and t in title_lower)

    if title_nouns:  score += title_nouns * 5;  details.append(f"title(noun×{title_nouns})"); title_hit = True
    if title_adjs:   score += title_adjs  * 4;  details.append(f"title(adj×{title_adjs})");  title_hit = True
    if title_verbs:  score += title_verbs * 3;  details.append(f"title(verb×{title_verbs})"); title_hit = True

    # Cap title contribution at +12
    title_contribution = sum([title_nouns*5, title_adjs*4, title_verbs*3])
    if title_contribution > 12:
        score -= (title_contribution - 12)

    # === 3. H1 ===
    h1_lower = h1.lower()
    h1_nouns = sum(1 for t in tokens if classify_token(t) == "noun" and t in h1_lower)
    h1_adjs  = sum(1 for t in tokens if classify_token(t) == "adj"  and t in h1_lower)
    h1_verbs = sum(1 for t in tokens if classify_token(t) == "verb" and t in h1_lower)

    if h1_nouns:  score += h1_nouns * 5;  details.append(f"h1(noun×{h1_nouns})"); h1_hit = True
    if h1_adjs:   score += h1_adjs  * 4;  details.append(f"h1(adj×{h1_adjs})");  h1_hit = True
    if h1_verbs:  score += h1_verbs * 3;  details.append(f"h1(verb×{h1_verbs})"); h1_hit = True

    h1_contribution = sum([h1_nouns*5, h1_adjs*4, h1_verbs*3])
    if h1_contribution > 12:
        score -= (h1_contribution - 12)

    # === 4. H2 + H3 ===
    h2h3_nouns = sum(1 for t in tokens if classify_token(t) == "noun" and t in h2_h3)
    h2h3_adjs  = sum(1 for t in tokens if classify_token(t) == "adj"  and t in h2_h3)

    if h2h3_nouns: score += h2h3_nouns * 5; details.append(f"h2h3(noun×{h2h3_nouns})"); h2h3_hit = True
    if h2h3_adjs:  score += h2h3_adjs  * 3; details.append(f"h2h3(adj×{h2h3_adjs})");  h2h3_hit = True

    h2h3_contribution = h2h3_nouns*5 + h2h3_adjs*3
    if h2h3_contribution > 8:
        score -= (h2h3_contribution - 8)

    # === 5. SYNERGY BONUSES ===
    if slug_hit and title_hit:
        score += 5
        details.append("synergy:slug+title")
    if title_hit and h1_hit:
        score += 10
        details.append("synergy:title+h1")
    if h2h3_hit and (slug_hit or title_hit or h1_hit):
        score += 15
        details.append("synergy:h2h3+top")

    note = ", ".join(details) if details else "no signal"
    return score, note


def url_depth(url: str) -> int:
    path = urlparse(url).path
    return len([p for p in path.strip("/").split("/") if p])


def has_exact_phrase(keyword: str, text: str) -> bool:
    if not text or not keyword:
        return False
    pattern = re.escape(keyword.strip())
    return bool(re.search(r"\b" + pattern + r"\b", text, re.IGNORECASE))


def weighted_map_keywords(df: pd.DataFrame, page_signals: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """Main mapping function — returns list of results"""
    if not page_signals:
        return [{"keyword": row.get("Keyword",""), "chosen_url": None, "note": "no pages crawled"} 
                for _, row in df.iterrows()]

    # Track how many keywords assigned per URL per category
    assigned = {url: {"SEO": 0, "AIO": 0, "VEO": 0} for url in page_signals}
    results = []

    for _, row in df.iterrows():
        kw = str(row.get("Keyword", "")).strip()
        category = str(row.get("Category", "SEO")).split(",")[0].strip() or "SEO"

        candidates = []
        for url, signals in page_signals.items():
            raw_score, note = calculate_structural_score(kw, signals)
            if raw_score < 5:  # Strict threshold
                continue

            # Tiebreakers
            title_h1_text = signals.get("title", "") + " " + signals.get("h1", "")
            exact_in_top = has_exact_phrase(kw, title_h1_text)
            depth = url_depth(url)

            candidates.append((
                raw_score,
                exact_in_top,      # exact phrase in title/h1 wins ties
                -depth,            # shallower depth wins
                -len(url),         # shorter URL wins
                url,
                note
            ))

        if not candidates:
            results.append({"keyword": kw, "chosen_url": None, "note": "score<5"})
            continue

        # Sort: highest score first
        candidates.sort(reverse=True)

        placed = False
        for _, _, _, _, url, note in candidates:
            cap = 2 if category == "SEO" else 1
            if assigned[url].get(category, 0) < cap:
                assigned[url][category] = assigned[url].get(category, 0) + 1
                results.append({
                    "keyword": kw,
                    "chosen_url": url,
                    "note": note
                })
                placed = Tru
                break

        if not placed:
            results.append({"keyword": kw, "chosen_url": None, "note": "caps exceeded"})

    return results