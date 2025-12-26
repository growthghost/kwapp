# mapping.py
# Structural scoring model with intent + URL-type containers (staged selection)
# STEP A: Consolidation only — no behavior changes

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
THRESHOLD = 3
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


INTENT_URLTYPE_PRIORITY: Dict[str, List[str]] = {
    INTENT_NAVIGATIONAL:  [URL_TYPE_HOME, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_ACTION, URL_TYPE_OTHER],
    INTENT_INFORMATIONAL: [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_OTHER, URL_TYPE_ACTION],
    INTENT_TRANSACTIONAL: [URL_TYPE_ACTION, URL_TYPE_HUB, URL_TYPE_CONTENT, URL_TYPE_OTHER],
    INTENT_COMMERCIAL:    [URL_TYPE_CONTENT, URL_TYPE_HUB, URL_TYPE_OTHER, URL_TYPE_ACTION],
}


STOPWORDS = {
    "a","an","and","are","as","at","be","by","can","could","do","does","did","for",
    "from","get","how","i","in","into","is","it","its","me","my","of","on","or",
    "our","the","their","then","there","they","this","to","up","was","we","what",
    "when","where","which","who","why","will","with","you","your",
    "need","needed","make","creating","create",
}

GENERIC_TOKENS = {
    "home","homepage","official","site","website",
    "community","resources","resource","services","service","program","programs",
    "support","help","information","about","contact",
    "who","what","where","how","why",
    "give","giving","donate","donation","donations","ways",
    "advocacy","benefits","benefit",
}

MIN_DISTINCTIVE_TOKENS_TO_GATE = 1


# ---------- Tokenization ----------
def tokenize(text: str) -> List[str]:
    return TOKENIZER.findall(text.lower()) if text else []

def _token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def url_depth(url: str) -> int:
    parts = [p for p in url.strip("/").split("/") if p]
    return len(parts)


# ---------- Intent Detection ----------
def detect_intent(keyword: str) -> str:
    kw = (keyword or "").lower()

    if any(t in kw for t in ["homepage","official site"," about "," careers"," jobs "]):
        return INTENT_NAVIGATIONAL

    if any(t in kw for t in ["buy","order","apply","enroll","donate","sign up","register","get quote"]):
        return INTENT_TRANSACTIONAL

    if any(t in kw for t in ["best","top","vs","review","compare","pricing","cost"]):
        return INTENT_COMMERCIAL

    return INTENT_INFORMATIONAL


# ---------- URL Type Classification ----------
def _normalize_path(url: str) -> str:
    raw = (url or "").strip()
    path = re.sub(r"^https?://[^/]+", "", raw)
    if not path:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    return path


def classify_url_type(url: str, signals: Dict[str, str]) -> str:
    path = _normalize_path(url)
    path_lower = path.lower()

    title = (signals.get("title") or "").lower()
    h1 = (signals.get("h1") or "").lower()

    if path == "/" or path == "":
        return URL_TYPE_HOME

    if any(t in path_lower for t in ["donate","apply","contact","login","signup","register"]):
        return URL_TYPE_ACTION

    segments = [p for p in path.strip("/").split("/") if p]
    last = segments[-1] if segments else ""

    if last in {"services","about","who-we-are","what-we-do","ways-to-give","advocacy"}:
        return URL_TYPE_HUB

    if len(segments) >= 2:
        return URL_TYPE_CONTENT

    return URL_TYPE_OTHER


# ---------- Distinctive Token Gate ----------
def _distinctive_tokens(keyword: str) -> Set[str]:
    return {
        t for t in _token_set(keyword)
        if t not in STOPWORDS and t not in GENERIC_TOKENS and not t.isdigit()
    }


def passes_distinctive_gate(keyword: str, signals: Dict[str, str]) -> bool:
    distinct = _distinctive_tokens(keyword)
    if len(distinct) < MIN_DISTINCTIVE_TOKENS_TO_GATE:
        return True

    page_tokens = (
        _token_set(signals.get("slug",""))
        | _token_set(signals.get("title",""))
        | _token_set(signals.get("h1",""))
        | _token_set(signals.get("h2",""))
        | _token_set(signals.get("h3",""))
    )

    return bool(distinct & page_tokens)


# ---------- Structural Scoring ----------
def structural_score(keyword: str, signals: Dict[str, str]) -> int:
    kw_tokens = _token_set(keyword)
    score = 0
    matched = set()

    for field, weight in WEIGHTS.items():
        tokens = _token_set(signals.get(field,""))
        overlap = kw_tokens & tokens
        if overlap:
            score += weight * len(overlap)
            matched.add(field)

    if "slug" in matched and "title" in matched:
        score += BONUS_URL_TITLE
    if "title" in matched and "h1" in matched:
        score += BONUS_TITLE_H1
    if "h2h3" in matched and matched & {"slug","title","h1"}:
        score += BONUS_H2H3_ANY

    return int(score)


# ==========================================================
# 🔑 SINGLE ENTRY POINT FOR APP.PY
# ==========================================================
def run_mapping(
    df: pd.DataFrame,
    page_signals_by_url: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """
    STEP A: Consolidated mapping entry point.
    Assumes df already contains:
      - Keyword
      - Category
      - Eligible
      - Tier
      - Strategy

    Writes Map URL directly onto df and returns df.
    """

    df = df.copy()
    df["Map URL"] = ""

    # Precompute URL types
    url_types = {u: classify_url_type(u, s) for u, s in page_signals_by_url.items()}

    for idx, row in df.iterrows():
        if str(row.get("Eligible","")).lower() != "yes":
            continue

        kw = str(row.get("Keyword","")).strip()
        if not kw:
            continue

        intent = detect_intent(kw)
        priorities = INTENT_URLTYPE_PRIORITY[intent]

        best_url = None
        best_score = 0

        for url_type in priorities:
            candidates = [
                u for u, t in url_types.items() if t == url_type
            ]

            scored = []
            for url in candidates:
                signals = page_signals_by_url[url]

                if not passes_distinctive_gate(kw, signals):
                    continue

                score = structural_score(kw, signals)
                if score >= THRESHOLD:
                    scored.append((url, score))

            if scored:
                scored.sort(key=lambda x: (x[1], -url_depth(x[0])), reverse=True)
                best_url, best_score = scored[0]
                break

        if best_url:
            df.at[idx, "Map URL"] = best_url

    return df
