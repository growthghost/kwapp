# ------------------- Imports & Setup -------------------
import io
import re
import math
import asyncio
import contextlib
import gzip
import hashlib
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict
from urllib.parse import urlparse, urljoin
from datetime import datetime

import pandas as pd
import streamlit as st

# Optional dependencies
try:
    import aiohttp
    HAVE_AIOHTTP = True
except Exception:
    HAVE_AIOHTTP = False

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

# ------------------- Brand / Theme -------------------
BRAND_BG = "#FFFFFF"
BRAND_ACCENT = "#00308F"
BRAND_GOOD = "#199965"
BRAND_WARN = "#FFAA00"
BRAND_BAD = "#D0021B"
FONT_FAMILY = "sans-serif"

# Strategy scoring rules
STRATEGY_RULES = {
    "Low Hanging Fruit": {
        "label": "LHF",
        "vol_min": 10,
        "vol_max": 1499,
        "kd_max": 29,
    },
    "In The Game": {
        "label": "ITG",
        "vol_min": 1500,
        "vol_max": 2999,
        "kd_max": 59,
    },
    "Competitive": {
        "label": "COMP",
        "vol_min": 3000,
        "vol_max": math.inf,
        "kd_max": 100,
    }
}

CATEGORY_LIMITS = {"SEO": 2, "AIO": 1, "VEO": 1}

# ------------------- Guardrail Matching Logic -------------------
def _ntokens(s: str) -> List[str]:
    return re.findall(r"\b[a-z]{2,}\b", s.lower())

def _extract_core_tokens(kw: str) -> Set[str]:
    stopwords = {
        "how", "what", "where", "when", "why", "who", "which", "get", "for", "to", "a", "an", "the",
        "is", "are", "was", "were", "can", "i", "you", "in", "on", "of", "and", "do", "does", "with",
        "my", "your", "me", "we", "be", "that", "this", "from", "it", "as", "at", "by", "if"
    }
    return set(t for t in _ntokens(kw) if t not in stopwords)

def _extract_bigrams(kw: str) -> List[Tuple[str, str]]:
    tokens = _ntokens(kw)
    return list(zip(tokens, tokens[1:]))

def _passes_core_concept_guardrail(kw: str, page_tokens: set, page_text: str) -> bool:
    kw_lower = kw.strip().lower()
    if kw_lower in page_text:
        return True
    core_tokens = _extract_core_tokens(kw)
    bigrams = _extract_bigrams(kw)
    if any(f"{a} {b}" in page_text for (a, b) in bigrams):
        return True
    if sum(1 for t in core_tokens if t in page_tokens) >= 2:
        return True
    return False
# ------------------- Streamlit UI Setup -------------------
st.set_page_config("OutrankIQ", layout="wide")
st.title("OutrankIQ – Keyword Mapper")

uploaded = st.file_uploader("Upload keyword CSV", type="csv")
strategy = st.selectbox("Choose Scoring Strategy", list(STRATEGY_RULES.keys()))
run_button = st.button("Map Keywords to URLs")

# ------------------- Simulated Crawl Content (Sample Data) -------------------
page_profiles = {
    "https://fello.org/service-dog-training/": {
        "tokens": {"service", "dog", "training", "support"},
        "text": "Get certified support or service dog training help"
    },
    "https://fello.org/admissions/": {
        "tokens": {"apply", "admissions", "cost", "tuition"},
        "text": "Admissions and tuition info"
    },
    "https://fello.org/contact/": {
        "tokens": {"contact", "email", "phone", "location"},
        "text": "How to contact us"
    },
    "https://fello.org/careers/": {
        "tokens": {"careers", "jobs", "employment", "paraprofessional"},
        "text": "Join our team in special education, human services, and more"
    },
    "https://fello.org/about/": {
        "tokens": {"mission", "values", "organization", "team"},
        "text": "Learn about our mission, values, and dedicated team"
    }
}
# ------------------- Keyword Scoring Utility -------------------
def _is_eligible(row: pd.Series, strat: str) -> bool:
    rules = STRATEGY_RULES[strat]
    vol = row["Search Volume"]
    kd = row["Keyword Difficulty"]
    return rules["vol_min"] <= vol <= rules["vol_max"] and kd <= rules["kd_max"]

def _get_cat_tags(raw: str) -> List[str]:
    return [tag.strip() for tag in str(raw).split(",") if tag.strip() in CATEGORY_LIMITS]

def _normalize_keyword(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

# ------------------- Download Helper -------------------
def _download_csv(df: pd.DataFrame, label: str = "Download CSV"):
    st.download_button(label, df.to_csv(index=False), file_name="mapped_keywords.csv", mime="text/csv")
# ------------------- Main Mapping Logic -------------------
if run_button and uploaded:
    df = pd.read_csv(uploaded)

    # Filter by scoring strategy eligibility
    df = df[df.apply(lambda r: _is_eligible(r, strategy), axis=1)].copy()
    df.reset_index(drop=True, inplace=True)

    # Track how many keywords are already mapped per category per URL
    used_slots: Dict[str, Dict[str, int]] = {
        url: {cat: 0 for cat in CATEGORY_LIMITS} for url in page_profiles
    }

    mapped_urls = []

    for _, row in df.iterrows():
        kw = row["Keyword"]
        cats = _get_cat_tags(row["Category Tags"])
        phrase = kw.lower()
        core = _extract_core_tokens(kw)
        bigrams = _extract_bigrams(kw)

        candidates = []
        for url, profile in page_profiles.items():
            if any(used_slots[url][c] >= CATEGORY_LIMITS[c] for c in cats):
                continue

            tokens = profile["tokens"]
            text = profile["text"].lower()

            # Global Core Concept Guardrail
            if _passes_core_concept_guardrail(kw, tokens, text):
                if phrase in text:
                    candidates.append((url, 3))  # Exact phrase match
                elif any(f"{a} {b}" in text for (a, b) in bigrams):
                    candidates.append((url, 2))  # Bigram match
                elif sum(1 for t in core if t in tokens) >= 2:
                    candidates.append((url, 1))  # Token overlap match

        if candidates:
            # Sort by rank descending, then alphabetically
            candidates.sort(key=lambda x: (-x[1], x[0]))
            best_url = candidates[0][0]

            # Reserve category slots
            for cat in cats:
                used_slots[best_url][cat] += 1

            mapped_urls.append(best_url)
        else:
            mapped_urls.append("")

    df["Mapped URL"] = mapped_urls
    st.success("✅ Mapping complete!")
    _download_csv(df, label="Download Mapped Keywords")
