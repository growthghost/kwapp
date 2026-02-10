### START crawler.py — PART 1 / 4

import re
import asyncio
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin

import aiohttp
from bs4 import BeautifulSoup

# -------------------------------
# Normalization helpers
# -------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

STOPWORDS = {
    "the","and","for","to","a","an","of","with","in","on","at","by","from",
    "is","are","be","can","should","how","what","who","where","why","when",
    "me","you","us","our","your","we","it","they","them"
}

def _normalize_url(u: str) -> str:
    u = u.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/') or '/'}"

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [
        t.lower()
        for t in TOKEN_RE.findall(text)
        if t.lower() not in STOPWORDS
    ]

def _slug_tokens(url: str) -> List[str]:
    path = urlparse(url).path.lower()
    parts = re.split(r"[-_/]+", path)
    return [p for p in parts if p and p not in STOPWORDS]

# -------------------------------
# VEO detection (secondary flag)
# -------------------------------

PHONE_PAT = re.compile(r"(\+?\d{1,2}[\s.-]?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
ADDR_PAT = re.compile(
    r"\b(street|st\.|road|rd\.|avenue|ave\.|blvd|lane|ln\.|drive|dr\.)\b",
    re.I,
)

# -------------------------------
# Core fetch function (single URL)
# -------------------------------

async def _fetch_html(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    try:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status >= 400:
                return None
            ctype = resp.headers.get("Content-Type", "").lower()
            if "html" not in ctype and "text" not in ctype:
                return None
            raw = await resp.text(errors="ignore")
            return raw
    except Exception:
        return None

### END crawler.py — PART 1 / 4

### START crawler.py — PART 2 / 4

# -------------------------------
# HTML parsing & signal extraction
# -------------------------------

def _extract_signals(html: str, final_url: str, requested_url: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content sections FIRST
    for tag in soup(["nav", "footer", "header", "aside", "script", "style", "noscript"]):
        tag.decompose()

    # ---- Title ----
    title_text = ""
    if soup.title and soup.title.get_text(strip=True):
        title_text = soup.title.get_text(" ", strip=True)

    # ---- Headings ----
    h1_texts = [h.get_text(" ", strip=True) for h in soup.find_all("h1")]
    h2h3_texts = [h.get_text(" ", strip=True) for h in soup.find_all(["h2", "h3"])]

    # ---- Meta description ----
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"]

    # ---- Canonical ----
    canonical = final_url
    link = soup.find("link", rel=lambda v: v and "canonical" in v)
    if link and link.get("href"):
        canonical = urljoin(final_url, link["href"])

    # ---- Limited body text (first ~150 words ONLY) ----
    body_text = soup.get_text(" ", strip=True)
    words = body_text.split()
    lead_text = " ".join(words[:150]).lower()

    # ---- VEO readiness (secondary flag only) ----
    veo_ready = False
    if PHONE_PAT.search(lead_text) or ADDR_PAT.search(lead_text):
        veo_ready = True
    if soup.find(attrs={"itemtype": re.compile(r"LocalBusiness|Organization", re.I)}):
        veo_ready = True

    return {
        "url": canonical,
        "requested_url": requested_url,
        "slug_tokens": set(_slug_tokens(final_url)),
        "title_tokens": set(_tokenize(title_text)),
        "h1_tokens": set(t for txt in h1_texts for t in _tokenize(txt)),
        "h2h3_tokens": set(t for txt in h2h3_texts for t in _tokenize(txt)),
        "meta_tokens": set(_tokenize(meta_desc)),
        "lead_text": lead_text,     # exact-phrase use only
        "veo_ready": veo_ready,     # secondary signal
    }

### END crawler.py — PART 2 / 4

### START crawler.py — PART 3 / 4

# -------------------------------
# Public entry: fetch_profiles
# -------------------------------

async def _fetch_profiles_async(urls: List[str]) -> List[Dict]:
    results: List[Dict] = []

    timeout = aiohttp.ClientTimeout(total=45)

    # Use browser-like headers (many career sites block obvious bot UAs)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        sem = asyncio.Semaphore(8)

        async def _fetch_one(u: str):
            async with sem:
                final = _normalize_url(u)
                html = await _fetch_html(session, final)
                if not html:
                    return None
                return _extract_signals(
                    html=html,
                    final_url=final,
                    requested_url=u
                )

        tasks = [_fetch_one(u) for u in urls]
        pages = await asyncio.gather(*tasks, return_exceptions=True)

        for p in pages:
            if isinstance(p, dict):
                results.append(p)

    return results


def fetch_profiles(urls: List[str]) -> List[Dict]:
    """
    Fetch and profile ONLY the user-supplied URLs.
    No crawling, no discovery, no sitemap expansion.
    """
    if not urls:
        return []

    # Normalize + dedupe (order preserved)
    seen = set()
    clean_urls: List[str] = []
    for u in urls:
        nu = _normalize_url(u)
        if nu not in seen:
            seen.add(nu)
            clean_urls.append(nu)

    try:
        return asyncio.run(_fetch_profiles_async(clean_urls))
    except RuntimeError:
        # fallback (rare, Streamlit loop collision)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            _fetch_profiles_async(clean_urls)
        )

### END crawler.py — PART 3 / 4

### START crawler.py — PART 4 / 4

# -------------------------------
# Expected output contract
# -------------------------------
#
# Each returned profile has:
#
# {
#   "url": canonical_url,
#   "requested_url": original_input_url,
#   "slug_tokens": set[str],
#   "title_tokens": set[str],
#   "h1_tokens": set[str],
#   "h2h3_tokens": set[str],
#   "meta_tokens": set[str],
#   "lead_text": str,       # phrase checks ONLY
#   "veo_ready": bool       # secondary flag
# }
#
# NO body tokens
# NO nav/footer bleed
# NO site discovery
# NO sitemap usage
#

def _profile_token_union(profile: Dict) -> set:
    """
    Utility helper (optional use in mapping):
    Combines all structural tokens into one set.
    """
    return (
        set(profile.get("slug_tokens", []))
        | set(profile.get("title_tokens", []))
        | set(profile.get("h1_tokens", []))
        | set(profile.get("h2h3_tokens", []))
        | set(profile.get("meta_tokens", []))
    )


def build_token_index(profiles: List[Dict]) -> Dict[str, List[int]]:
    """
    Optional inverted index:
    token -> list of page indices
    """
    index: Dict[str, List[int]] = {}
    for i, p in enumerate(profiles):
        for tok in _profile_token_union(p):
            index.setdefault(tok, []).append(i)
    return index

### END crawler.py — PART 4 / 4
