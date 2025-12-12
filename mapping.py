# mapping.py
# Structural scoring model with intent + URL-type containers (staged selection)
# - Still uses slug/title/H1/H2/H3 for scoring
# - Adds:
#     * Keyword intent detection (NAV/INFO/TRANS/COMMERCIAL)
#     * URL type detection (HOME/HUB/CONTENT/ACTION/OTHER)
#     * STAGED filtering: try preferred URL types first (avoid HUB stealing from CONTENT)
#     * Distinctive-token gate: prevents mappings driven only by generic tokens like "service/services"

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


# ---------- Intent -> URL Type Priority (STAGED) ----------
# We try URL types in order; once we find any candidates in a preferred bucket,
# we pick the best from that bucket and stop (prevents HUB from stealing INFO/COMM).
INTENT_URLTYPE_PRIORITY: Dict[str, List[str]] = {
    INTENT_NAVIGATIONAL:  [URL_TYPE_HOME, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_ACTION, URL_TYPE_OTHER],
    INTENT_INFORMATIONAL: [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_OTHER, URL_TYPE_ACTION],  # ACTION last for INFO
    INTENT_TRANSACTIONAL: [URL_TYPE_ACTION, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_OTHER],
    INTENT_COMMERCIAL:    [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_OTHER, URL_TYPE_ACTION],
}

# ---------- Generic/Stop Tokens (Distinctive Token Gate) ----------
# Goal: avoid mapping keywords based only on generic words like "service/services",
# which otherwise match lots of unrelated pages.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "did", "for",
    "from", "get", "how", "i", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "the", "their", "then", "there", "they", "this", "to", "up", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
    # common question fragments
    "need", "needed", "needed", "make", "creating", "create",
}

GENERIC_TOKENS = {
    # site/generic nav-ish words
    "home", "homepage", "official", "site", "website",
    # broad organization words
    "community", "resources", "resource", "services", "service", "program", "programs",
    "support", "help", "information", "about", "contact",
    "who", "what", "where", "how", "why",
    "give", "giving", "donate", "donation", "donations", "ways",
    "advocacy",  # often appears broadly on advocacy hubs; keep as generic to avoid over-mapping
    "benefits", "benefit",
}

# If a keyword has fewer than this many "distinctive" tokens, we won't enforce the gate strictly.
MIN_DISTINCTIVE_TOKENS_TO_GATE = 1


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
    NOTE: brand flag is intentionally not used for scoring yet unless you wire it in later.
    """
    if not brand_terms:
        return False
    kw = (keyword or "").lower()
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
    path = re.sub(r"^https?://[^/]+", "", raw)  # remove scheme+domain if present
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
        "get_quote",
    ]
    if (
        any(t in path_lower for t in action_terms)
        or any(t in title for t in action_terms)
        or any(t in h1 for t in action_terms)
    ):
        return URL_TYPE_ACTION

    # Split path into segments
    segments = [p for p in path.strip("/").split("/") if p]
    last_seg = segments[-1].lower() if segments else ""

    # 3) HUB pages – stricter classification
    # Known hub/section terms (site-agnostic and common)
    hub_like_terms = {
        "services", "service", "products", "product", "solutions", "programs",
        "resources", "blog", "news", "locations", "benefits",
        "about", "who-we-are", "what-we-do", "ways-to-give",
        "contact", "advocacy",
    }
    # Treat very shallow pages as HUB only if they look like a true section.
    if len(segments) <= 2:
        if last_seg in hub_like_terms:
            return URL_TYPE_HUB
        # If title/h1 looks like a section landing page
        sectionish = ("services" in title or "services" in h1 or "about" in title or "about" in h1
                      or "resources" in title or "resources" in h1 or "programs" in title or "programs" in h1
                      or "advocacy" in title or "advocacy" in h1)
        if sectionish and last_seg and last_seg.count("-") <= 3:
            return URL_TYPE_HUB

    # 4) CONTENT pages – deeper or clearly article/guide-like
    content_terms = [
        "blog", "article", "articles", "guide", "guides", "how-to", "how_to",
        "case-study", "case_study", "faq", "faqs", "insights",
    ]
    if (
        any(t in path_lower for t in content_terms)
        or any(t in title for t in content_terms)
        or any(t in h1 for t in content_terms)
    ):
        return URL_TYPE_CONTENT

    # If deeper than 2 segments, it's almost always content-like
    if len(segments) >= 2:
        return URL_TYPE_CONTENT

    # Shallow but not clearly a hub → treat as CONTENT (often "landing" pages are content-like)
    if len(segments) == 1 and last_seg:
        return URL_TYPE_CONTENT

    # 5) Fallback
    return URL_TYPE_OTHER


# ---------- Distinctive Token Gate ----------
def _distinctive_tokens(keyword: str) -> Set[str]:
    kw_tokens = _token_set(keyword)
    distinct = {t for t in kw_tokens if t not in STOPWORDS and t not in GENERIC_TOKENS and not t.isdigit()}
    return distinct


def passes_distinctive_gate(keyword: str, signals: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    """
    Returns (pass, reason_if_fail).
    Gate condition: at least one "distinctive" keyword token must appear in the page signals.
    Distinctive tokens exclude stopwords and a small list of generic tokens like "service/services".
    """
    distinct = _distinctive_tokens(keyword)

    # If there are no distinctive tokens, don't block mapping (nothing to gate on).
    if len(distinct) < MIN_DISTINCTIVE_TOKENS_TO_GATE:
        return True, None

    page_tokens = (
        _token_set(signals.get("slug", "") or "")
        | _token_set(signals.get("title", "") or "")
        | _token_set(signals.get("h1", "") or "")
        | _token_set(signals.get("h2", "") or "")
        | _token_set(signals.get("h3", "") or "")
    )

    if distinct & page_tokens:
        return True, None

    return False, f"Distinctive token gate failed (needed one of: {', '.join(sorted(distinct))})"


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
      - STAGED URL-type selection (try preferred types first)
      - Distinctive-token gate to prevent generic-token-only mappings
      - Structural scoring (slug/title/H1/H2/H3) for the final match

    Parameters
    ----------
    df : DataFrame
        Must contain "Keyword" and "Category".
    page_signals_by_url : dict
        {url: {"slug": ..., "title": ..., "h1": ..., "h2": ..., "h3": ...}}
    brand_terms : list[str], optional
        Brand/domain names for brand flag detection (kept optional and dormant unless used later).
    """
    results: List[Dict] = []

    # Precompute URL types once, and bucket URLs by type
    url_type_by_url: Dict[str, str] = {}
    urls_by_type: Dict[str, List[str]] = {
        URL_TYPE_HOME: [],
        URL_TYPE_ACTION: [],
        URL_TYPE_HUB: [],
        URL_TYPE_CONTENT: [],
        URL_TYPE_OTHER: [],
    }

    for url, signals in page_signals_by_url.items():
        t = classify_url_type(url, signals)
        url_type_by_url[url] = t
        urls_by_type.setdefault(t, []).append(url)

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

        # Detect intent & brand flag (brand is kept but not used for scoring yet)
        intent = detect_intent(kw)
        _ = detect_brand_flag(kw, brand_terms)  # intentionally unused for now

        type_priority = INTENT_URLTYPE_PRIORITY.get(intent, INTENT_URLTYPE_PRIORITY[INTENT_INFORMATIONAL])

        chosen = None  # (url, score, reasons)
        fail_gate_reason: Optional[str] = None

        # STAGED selection: try URL types in priority order
        for url_type in type_priority:
            candidate_scores = []

            for url in urls_by_type.get(url_type, []):
                signals = page_signals_by_url.get(url, {})

                # Distinctive token gate first (prevents "service" matching everything)
                gate_ok, gate_reason = passes_distinctive_gate(kw, signals)
                if not gate_ok:
                    fail_gate_reason = gate_reason  # keep latest; used only if everything fails
                    continue

                score, reasons = structural_score(kw, signals)
                if score >= THRESHOLD:
                    candidate_scores.append((url, score, reasons))

            if candidate_scores:
                candidate_scores.sort(
                    key=lambda x: (x[1], -url_depth(x[0]), -len(x[0])),
                    reverse=True,
                )
                chosen = candidate_scores[0]
                break  # stop at first bucket that yields any candidates

        if chosen:
            best_url, best_score, best_reasons = chosen
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": best_url,
                "weighted_score": best_score,
                "reasons": "; ".join(best_reasons),
            })
        else:
            # If we failed only because of the gate, include that as a clue
            reasons = "No match (below threshold or no structural alignment in preferred URL types)"
            if fail_gate_reason:
                reasons = f"No match ({fail_gate_reason})"
            results.append({
                "keyword": kw,
                "category": category,
                "chosen_url": None,
                "weighted_score": 0,
                "reasons": reasons,
            })

    return results
