# mapping.py
# Structural scoring model with intent + URL-type containers (staged selection)
# - Uses slug/title/H1/H2/H3 for scoring
# - Adds:
#     * Keyword intent detection (NAV/INFO/TRANS/COMMERCIAL)
#     * URL type detection (HOME/HUB/CONTENT/ACTION/OTHER)
#     * Intent → URL-type preference bonus (soft staging)
#     * Distinctive-token gate: prevents mappings driven only by generic tokens
#
# STEP B1–B3 (Mapping Improvements; NO UI changes):
#   ✅ Global keyword uniqueness (keyword used only once across all URLs)
#   ✅ Per-URL caps: AEO=1, AIO=1, SEO=2
#   ✅ Fill order per URL: AEO → AIO → SEO → SEO
#
# NOTE: This function writes Map URL directly onto df and returns df.

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


# ---------- Intent -> URL Type Priority (soft staged preference) ----------
INTENT_URLTYPE_PRIORITY: Dict[str, List[str]] = {
    INTENT_NAVIGATIONAL:  [URL_TYPE_HOME, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_ACTION, URL_TYPE_OTHER],
    INTENT_INFORMATIONAL: [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_OTHER, URL_TYPE_ACTION],  # ACTION last for INFO
    INTENT_TRANSACTIONAL: [URL_TYPE_ACTION, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_OTHER],
    INTENT_COMMERCIAL:    [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_OTHER, URL_TYPE_ACTION],
}

# ---------- Generic/Stop Tokens (Distinctive Token Gate) ----------
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "did", "for",
    "from", "get", "how", "i", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "the", "their", "then", "there", "they", "this", "to", "up", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "need", "needed", "make", "creating", "create",
}

GENERIC_TOKENS = {
    "home", "homepage", "official", "site", "website",
    "community", "resources", "resource", "services", "service", "program", "programs",
    "support", "help", "information", "about", "contact",
    "who", "what", "where", "how", "why",
    "give", "giving", "donate", "donation", "donations", "ways",
    "advocacy",
    "benefits", "benefit",
}

MIN_DISTINCTIVE_TOKENS_TO_GATE = 1


# ---------- Tokenization ----------
def tokenize(text: str) -> List[str]:
    return TOKENIZER.findall(text.lower()) if text else []

def _token_set(text: str) -> Set[str]:
    return set(tokenize(text))

# ---------- Site Vocabulary Gate ----------

def build_site_vocabulary(page_signals_by_url: Dict[str, Dict[str, str]]) -> Set[str]:
    """
    Build a site-wide vocabulary from all crawled page signals.
    """
    vocab: Set[str] = set()

    for signals in page_signals_by_url.values():
        for field in ("slug", "title", "h1", "h2", "h3"):
            text = signals.get(field, "")
            if not text:
                continue
            vocab.update(_token_set(text))

    return vocab


def keyword_in_site_vocab(keyword: str, site_vocab: Set[str]) -> bool:
    """
    Keyword passes if it shares at least one distinctive, non-generic token
    with the site vocabulary.
    """
    distinct = {
        t for t in _token_set(keyword)
        if t not in STOPWORDS
        and t not in GENERIC_TOKENS
        and not t.isdigit()
    }

    # If keyword has no distinctive tokens, don't block it here
    if not distinct:
        return True

    return bool(distinct & site_vocab)

# ---------- URL depth helper ----------
def url_depth(url: str) -> int:
    parts = [p for p in url.strip("/").split("/") if p]
    return len(parts)


# ---------- Keyword Intent ----------
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

    nav_terms = [
        " login", " log in", "homepage", "home page", "official site",
        " website", ".com", ".org", ".net", " contact", " about us",
        " about ", " careers", " jobs at ",
    ]
    if any(term in kw for term in nav_terms):
        return INTENT_NAVIGATIONAL

    trans_terms = [
        "buy ", " buy", "purchase", "order ", " order", "sign up", "signup", "register",
        "apply", "enroll", "book ", " book", "schedule", "donate", "give now",
        "get quote", "request quote", "estimate", "get started",
    ]
    if any(term in kw for term in trans_terms):
        return INTENT_TRANSACTIONAL

    comm_terms = [
        "best ", " top ", " vs ", " versus ", "review", "reviews",
        "compare ", " comparison", "pricing", "price of", "cost of",
        "cheap ", "affordable ",
    ]
    if any(term in kw for term in comm_terms):
        return INTENT_COMMERCIAL

    return INTENT_INFORMATIONAL


# ---------- URL Type Detection ----------
def _normalize_path(url: str) -> str:
    raw = (url or "").strip()
    path = re.sub(r"^https?://[^/]+", "", raw)
    if not path:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    return path


def classify_url_type(url: str, signals: Dict[str, str]) -> str:
    """
    Classify a URL into one of: HOME, ACTION, HUB, CONTENT, OTHER.
    Rule priority: HOME > ACTION > HUB > CONTENT > OTHER.
    """
    path = _normalize_path(url)
    path_lower = path.lower()

    title = (signals.get("title") or "").lower()
    h1 = (signals.get("h1") or "").lower()

    # HOME
    if path == "/" or path == "":
        return URL_TYPE_HOME

    # ACTION pages
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

    segments = [p for p in path.strip("/").split("/") if p]
    last_seg = segments[-1].lower() if segments else ""

    hub_like_terms = {
        "services", "service", "products", "product", "solutions", "programs",
        "resources", "blog", "news", "locations", "benefits",
        "about", "who-we-are", "what-we-do", "ways-to-give",
        "contact", "advocacy",
    }
    if len(segments) <= 2:
        if last_seg in hub_like_terms:
            return URL_TYPE_HUB
        sectionish = (
            ("services" in title or "services" in h1 or "about" in title or "about" in h1
             or "resources" in title or "resources" in h1 or "programs" in title or "programs" in h1
             or "advocacy" in title or "advocacy" in h1)
        )
        if sectionish and last_seg and last_seg.count("-") <= 3:
            return URL_TYPE_HUB

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

    if len(segments) >= 2:
        return URL_TYPE_CONTENT
    if len(segments) == 1 and last_seg:
        return URL_TYPE_CONTENT

    return URL_TYPE_OTHER


# ---------- Distinctive Token Gate ----------
def _distinctive_tokens(keyword: str) -> Set[str]:
    kw_tokens = _token_set(keyword)
    return {t for t in kw_tokens if t not in STOPWORDS and t not in GENERIC_TOKENS and not t.isdigit()}


def passes_distinctive_gate(keyword: str, signals: Dict[str, str]) -> bool:
    distinct = _distinctive_tokens(keyword)
    if not distinct:
        return True


    page_tokens = (
        _token_set(signals.get("slug", "") or "")
        | _token_set(signals.get("title", "") or "")
        | _token_set(signals.get("h1", "") or "")
        | _token_set(signals.get("h2", "") or "")
        | _token_set(signals.get("h3", "") or "")
    )
    return bool(distinct & page_tokens)


# ---------- Structural Scoring ----------
def structural_score(keyword: str, signals: Dict[str, str]) -> int:
    kw_tokens = _token_set(keyword)
    score = 0
    matched_fields = set()

    slug = signals.get("slug", "") or ""
    title = signals.get("title", "") or ""
    h1 = signals.get("h1", "") or ""
    h2 = signals.get("h2", "") or ""
    h3 = signals.get("h3", "") or ""

    slug_tokens = _token_set(slug)
    slug_overlap = kw_tokens & slug_tokens
    if slug_overlap:
        score += WEIGHTS["slug"] * len(slug_overlap)
        matched_fields.add("slug")

    title_tokens = _token_set(title)
    title_overlap = kw_tokens & title_tokens
    if title_overlap:
        score += WEIGHTS["title"] * len(title_overlap)
        matched_fields.add("title")

    h1_tokens = _token_set(h1)
    h1_overlap = kw_tokens & h1_tokens
    if h1_overlap:
        score += WEIGHTS["h1"] * len(h1_overlap)
        matched_fields.add("h1")

    h2h3_tokens = _token_set(h2) | _token_set(h3)
    h2h3_overlap = kw_tokens & h2h3_tokens
    if h2h3_overlap:
        score += WEIGHTS["h2h3"] * len(h2h3_overlap)
        matched_fields.add("h2h3")

    if "slug" in matched_fields and "title" in matched_fields:
        score += BONUS_URL_TITLE
    if "title" in matched_fields and "h1" in matched_fields:
        score += BONUS_TITLE_H1
    if "h2h3" in matched_fields and ({"slug", "title", "h1"} & matched_fields):
        score += BONUS_H2H3_ANY

    return int(round(score))


# ---------- Slot selection helpers ----------
def _is_yes(val: object) -> bool:
    return str(val or "").strip().lower() == "yes"


def _primary_slot_from_category(category: str) -> str:
    """
    Each keyword must serve one slot type for mapping.
    Priority: AEO > AIO > SEO.
    Category can be like: "SEO, GEO, AEO" or "SEO" etc.
    """
    c = (category or "").upper()
    if "AEO" in c:
        return "AEO"
    if "AIO" in c:
        return "AIO"
    return "SEO"


def _intent_urltype_bonus(intent: str, url_type: str) -> int:
    """
    Soft preference bonus based on intent URL-type priority rank.
    Earlier types get a small bonus. Does NOT override structural mismatch.
    """
    priority = INTENT_URLTYPE_PRIORITY.get(intent, INTENT_URLTYPE_PRIORITY[INTENT_INFORMATIONAL])
    if url_type not in priority:
        return 0
    rank = priority.index(url_type)  # 0 is best
    # Simple decreasing bonus: 8,6,4,2,0 ...
    return max(0, 8 - (rank * 2))


# ==========================================================
# 🔑 SINGLE ENTRY POINT USED BY APP.PY
# ==========================================================
def run_mapping(
    df: pd.DataFrame,
    page_signals_by_url: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """
    Mapping engine (no UI changes):
      - Uses Eligible == Yes as the candidate pool
      - Assigns up to 4 keywords per URL: AEO=1, AIO=1, SEO=2
      - Enforces global uniqueness: a keyword row can only be assigned once total
      - Fill order per URL: AEO → AIO → SEO → SEO

    Writes "Map URL" directly onto df and returns df.

    Required columns in df:
      - Keyword
      - Category
      - Eligible (Yes/No)
    """

    df = df.copy()

    # Ensure Map URL exists and starts blank for this run
    df["Map URL"] = ""

    if not page_signals_by_url:
        return df
    site_vocab = build_site_vocabulary(page_signals_by_url)


    # Precompute URL types once
    url_type_by_url: Dict[str, str] = {
        url: classify_url_type(url, signals or {})
        for url, signals in page_signals_by_url.items()
    }

    # Candidate pool: only Eligible == Yes
    eligible_idx = [i for i, v in df["Eligible"].items()] if "Eligible" in df.columns else list(df.index)
    if "Eligible" in df.columns:
        eligible_idx = [i for i in df.index if _is_yes(df.at[i, "Eligible"])]

    # Precompute per-row primary slot + intent + keyword text
    row_kw: Dict[int, str] = {}
    row_slot: Dict[int, str] = {}
    row_intent: Dict[int, str] = {}

    for i in eligible_idx:
        kw = str(df.at[i, "Keyword"] if "Keyword" in df.columns else "").strip()
        if not kw:
            continue
        cat = str(df.at[i, "Category"] if "Category" in df.columns else "SEO").strip()
        row_kw[i] = kw
        row_slot[i] = _primary_slot_from_category(cat)
        row_intent[i] = detect_intent(kw)

    # Build candidate lists per URL per slot
    # candidates[url][slot] -> list of (score, idx) sorted desc
    candidates: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    for url, signals in page_signals_by_url.items():
        candidates[url] = {"AEO": [], "AIO": [], "SEO": []}

        url_type = url_type_by_url.get(url, URL_TYPE_OTHER)

        for i, kw in row_kw.items():
            slot = row_slot.get(i, "SEO")
            intent = row_intent.get(i, INTENT_INFORMATIONAL)

            # Gate first to avoid generic-only matches
            if not passes_distinctive_gate(kw, signals):
                continue

            # 🔒 Site vocabulary gate (topical boundary)
            if not keyword_in_site_vocab(kw, site_vocab):
                continue

            base = structural_score(kw, signals)
            if base < THRESHOLD:
                continue

            score = base + _intent_urltype_bonus(intent, url_type)
            candidates[url][slot].append((score, i))

        # Sort each slot list once
        for slot in ("AEO", "AIO", "SEO"):
            candidates[url][slot].sort(key=lambda x: x[0], reverse=True)

    # Assignment (global uniqueness + per-URL caps)
    used_rows: Set[int] = set()

    caps = {"AEO": 1, "AIO": 1, "SEO": 2}
    fill_sequence = ["AEO", "AIO", "SEO", "SEO"]

    # Preserve URL order as provided by caller dict
    for url in page_signals_by_url.keys():
        filled = {"AEO": 0, "AIO": 0, "SEO": 0}

        for slot in fill_sequence:
            if filled[slot] >= caps[slot]:
                continue

            # Find best available unused candidate for this url+slot
            picked_idx = None
            for score, idx in candidates[url][slot]:
                if idx in used_rows:
                    continue
                picked_idx = idx
                break

            if picked_idx is None:
                continue

            df.at[picked_idx, "Map URL"] = url
            used_rows.add(picked_idx)
            filled[slot] += 1

    return df
