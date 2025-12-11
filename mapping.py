# mapping.py
# Structural scoring model with simple intent + URL-type containers
# - Still uses slug/title/H1/H2/H3 for scoring
# - Adds:
#     * Keyword intent detection (NAV/INFO/TRANS/COMMERCIAL)
#     * URL type detection (HOME/HUB/CONTENT/ACTION/OTHER)
#     * Filtering: only URLs with allowed types for that intent are scored

import pandas as pd
import re
from typing import Dict, List, Set, Tuple, Optional

# ---------- Structural Weights ----------
WEIGHTS = {
    "slug": 6,
    "title": 5,
    "h1": 5,
    "h2h3": 5,
}

# ---------- Threshold and Bonuses ----------
THRESHOLD = 3  # must reach or exceed this to map
BONUS_URL_TITLE = 5
BONUS_TITLE_H1 = 10
BONUS_H2H3_ANY = 15

TOKENIZER = re.compile(r"[a-zA-Z0-9]+")

# ---------- Intent & URL-Type Constants ----------

INTENT_NAVIGATIONAL = "NAVIGATIONAL"
INTENT_INFORMATIONAL = "INFORMATIONAL"
INTENT_TRANSACTIONAL = "TRANSACTIONAL"
INTENT_COMMERCIAL = "COMMERCIAL"

URL_TYPE_HOME = "HOME"
URL_TYPE_HUB = "HUB"
URL_TYPE_CONTENT = "CONTENT"
URL_TYPE_ACTION = "ACTION"
URL_TYPE_OTHER = "OTHER"

# For each intent, which URL types are allowed (used for filtering)
INTENT_ALLOWED_URL_TYPES: Dict[str, List[str]] = {
    INTENT_NAVIGATIONAL:  [URL_TYPE_HOME, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_ACTION, URL_TYPE_OTHER],
    INTENT_INFORMATIONAL: [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_ACTION, URL_TYPE_OTHER],
    INTENT_TRANSACTIONAL: [URL_TYPE_ACTION, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_OTHER],
    INTENT_COMMERCIAL:    [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_ACTION, URL_TYPE_OTHER],
}

# ---------- Tokenization ----------
def tokenize(text: str) -> List[str]:
    return TOKENIZER.findall(text.lower()) if text else []

def _token_set(text: str) -> Set[str]:
    return set(tokenize(text))

# ---------- URL depth helper ----------
def url_depth(url: str) -> int:
    parts = [p for p in url.strip("/").split("/") if p]
    return len(parts)

# ---------- Keyword Intent & Brand Detection ----------

def detect_brand_flag(keyword: str, brand_terms: Optional[List[str]]) -> bool:
    """
    Detect if the keyword is brand-related.
    This is an overlay flag – intent is still NAV/INFO/TRANS/COMMERCIAL.
    """
    if not brand_terms:
        return False
    kw = keyword.lower()
    for term in brand_terms:
        if not term:
            continue
        t = term.lower().strip()
        if t and t in kw:
            return True
    return False


def detect_intent(keyword: str) -> str:
    """
    Detect high-level search intent from the keyword text.
    Order of checks:
      1) Navigational
      2) Transactional
      3) Commercial
      4) Default to Informational
    """
    kw = (keyword or "").strip().lower()

    # Navigational patterns (site/section/brand-like)
    nav_terms = [
        " login", " log in", "homepage", "home page", "official site",
        " website", ".com", ".org", ".net", " contact", " about us",
        " about ", " careers", " jobs at ",
    ]
    if any(term in kw for term in nav_terms):
        return INTENT_NAVIGATIONAL

    # Transactional patterns (do it now)
    trans_terms = [
        "buy ", " buy", "purchase", "order ", " order", "sign up", "signup", "register",
        "apply", "enroll", "book ", " book", "schedule", "donate", "give now",
        "get quote", "request quote", "estimate", "get started",
    ]
    if any(term in kw for term in trans_terms):
        return INTENT_TRANSACTIONAL

    # Commercial investigation patterns (compare/evaluate)
    comm_terms = [
        "best ", " top ", " vs ", " versus ", "review", "reviews",
        "compare ", " comparison", "pricing", "price of", "cost of",
        "cheap ", "affordable ",
    ]
    if any(term in kw for term in comm_terms):
        return INTENT_COMMERCIAL

    # Default: Informational
    return INTENT_INFORMATIONAL

# ---------- URL Type Detection Helpers ----------

def _normalize_path(url: str) -> str:
    """
    Strip scheme + domain, return just the path starting with '/'.
    Works for both full URLs and path-only strings.
    """
    raw = (url or "").strip()
    # Remove scheme + domain if present
    path = re.sub(r"^https?://[^/]+", "", raw)
    if not path:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    return path


def classify_url_type(url: str, signals: Dict[str, str]) -> str:
    """
    Classify a URL into one of: HOME, ACTION, HUB, CONTENT, OTHER.
    Rule priority is important: HOME > ACTION > HUB > CONTENT > OTHER.
    """
    path = _normalize_path(url)
    path_lower = path.lower()

    title = (signals.get("title") or "").lower()
    h1 = (signals.get("h1") or "").lower()

    # 1) HOME
    if path == "/" or path == "":
        return URL_TYPE_HOME

    # 2) ACTION pages
    action_terms = [
        "apply", "donate", "contact", "login", "log-in", "sign-up", "signup",
        "checkout", "book", "schedule", "register", "quote", "get-quote",
        "get_quote", "get started",
    ]
    if any(t in path_lower for t in action_terms) or any(t in title for t in action_terms) or any(t in h1 for t in action_terms):
        return URL_TYPE_ACTION

    # Split path into segments
    segments = [p for p in path.strip("/").split("/") if p]

    # 3) HUB pages – shallow, category/section-like
    #    e.g., /services/, /blog/, /products/, /resources/
    hub_like_terms = [
        "services", "service", "products", "product", "solutions", "programs",
        "resources", "blog", "news", "locations", "benefits",
    ]
    if len(segments) <= 2:
        last_seg = segments[-1] if segments else ""
        # Hub if last segment is a known category-ish term
        if last_seg.lower() in hub_like_terms:
            return URL_TYPE_HUB
        # If last segment has few hyphens, treat as a hub/section rather than a long article slug
        if last_seg and last_seg.count("-") <= 2:
            return URL_TYPE_HUB

    # 4) CONTENT pages – deeper or clearly article/guide-like
    content_terms = [
        "blog", "article", "guide", "how-to", "how_to", "case-study",
        "case_study", "faq", "faqs", "resources", "insights",
    ]
    if any(t in path_lower for t in content_terms) or any(t in title for t in content_terms) or any(t in h1 for t in content_terms):
        return URL_TYPE_CONTENT
    if len(segments) >= 2:
        # Deeper paths usually content pages
        return URL_TYPE_CONTENT

    # 5) Fallback: OTHER
    return URL_TYPE_OTHER

# ---------- Structural Scoring ----------

def structural_score(keyword: str, signals: Dict[str, str]) -> Tuple[int, List[str]]:
    kw_tokens = _token_set(keyword)
    score = 0
    reasons: List[str] = []
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
        reasons.append(f"Slug match: {', '.join(sorted(slug_overlap))} (+{base * len(slug_overlap)})")
        matched_fields.add("slug")

    # 2️⃣ Title
    title_tokens = _token_set(title)
    title_overlap = kw_tokens & title_tokens
    if title_overlap:
        base = WEIGHTS["title"]
        score += base * len(title_overlap)
        reasons.append(f"Title match: {', '.join(sorted(title_overlap))} (+{base * len(title_overlap)})")
        matched_fields.add("title")

    # 3️⃣ H1
    h1_tokens = _token_set(h1)
    h1_overlap = kw_tokens & h1_tokens
    if h1_overlap:
        base = WEIGHTS["h1"]
        score += base * len(h1_overlap)
        reasons.append(f"H1 match: {', '.join(sorted(h1_overlap))} (+{base * len(h1_overlap)})")
        matched_fields.add("h1")

    # 4️⃣ H2 / H3
    h2h3_tokens = _token_set(h2) | _token_set(h3)
    h2h3_overlap = kw_tokens & h2h3_tokens
    if h2h3_overlap:
        base = WEIGHTS["h2h3"]
        score += base * len(h2h3_overlap)
        reasons.append(f"H2/H3 match: {', '.join(sorted(h2h3_overlap))} (+{base * len(h2h3_overlap)})")
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

def weighted_map_keywords(
    df: pd.DataFrame,
    page_signals_by_url: Dict[str, Dict[str, str]],
    brand_terms: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Map keywords to URLs using:
      - Intent containers (NAV/INFO/TRANS/COMMERCIAL)
      - URL type containers (HOME/HUB/CONTENT/ACTION/OTHER)
      - Structural scoring (slug/title/H1/H2/H3) for the final match

    Parameters
    ----------
    df : DataFrame
        Must contain "Keyword" and "Category".
    page_signals_by_url : dict
        {url: {"slug": ..., "title": ..., "h1": ..., "h2": ..., "h3": ...}}
    brand_terms : list[str], optional
        Brand/domain names for brand flag detection.
    """
    results: List[Dict] = []

    # Precompute URL types once
    url_type_by_url: Dict[str, str] = {}
    for url, signals in page_signals_by_url.items():
        url_type_by_url[url] = classify_url_type(url, signals)

    for _, row in df.iterrows():
        kw = str(row.get("Keyword", "")).strip()
        category = str(row.get("Category", "SEO")).strip()

        if not kw:
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": None,
                "weighted_score": 0,
                "reasons": "Empty keyword",
            })
            continue

        # Detect intent & brand flag
        intent = detect_intent(kw)
        is_brand = detect_brand_flag(kw, brand_terms)

        # Decide which URL types are allowed for this keyword
        allowed_types = INTENT_ALLOWED_URL_TYPES.get(intent, INTENT_ALLOWED_URL_TYPES[INTENT_INFORMATIONAL])

        # Slight tweak for brand navigational queries:
        # if it's clearly brand + nav, HOME/HUB get more consideration via filtering/score.
        # (We still rely on structural_score to pick the final winner.)
        candidate_scores = []
        for url, signals in page_signals_by_url.items():
            url_type = url_type_by_url.get(url, URL_TYPE_OTHER)

            # Filter by intent ↔ URL-type containers
            if url_type not in allowed_types:
                continue

            score, reasons = structural_score(kw, signals)
            if score >= THRESHOLD:
                # Optionally, we could add small type/brand bonuses here later.
                candidate_scores.append((url, score, reasons))

        if candidate_scores:
            # Highest score wins, tie-break = shallower URL depth, then shorter URL
            candidate_scores.sort(
                key=lambda x: (x[1], -url_depth(x[0]), -len(x[0])),
                reverse=True,
            )
            best_url, best_score, best_reasons = candidate_scores[0]
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": best_url,
                "weighted_score": best_score,
                "reasons": "; ".join(best_reasons),
            })
        else:
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": None,
                "weighted_score": 0,
                "reasons": "No match (below threshold or no structural alignment within allowed URL types)",
            })

    return results
