import io
import re
import asyncio
import contextlib
import gzip
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from urllib.parse import urlparse, urljoin
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------- Optional deps ----------
try:
    import aiohttp  # type: ignore
    HAVE_AIOHTTP = True
except Exception:
    HAVE_AIOHTTP = False

try:
    import requests  # type: ignore
except Exception:  # Streamlit Cloud has requests preinstalled; this is a guard
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # noqa: F401
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

# ---------- Brand / Theme ----------
BRAND_BG = "#747474"     # background
BRAND_INK = "#242F40"    # blue/ink
BRAND_ACCENT = "#329662" # accent (green, replaces yellow)
BRAND_LIGHT = "#FFFFFF"  # white

st.set_page_config(page_title="OutrankIQ", page_icon="🔎", layout="centered")

# ---------- Global CSS ----------
st.markdown(
    f"""
<style>
:root {{
  --bg: {BRAND_BG};
  --ink: {BRAND_INK};
  --accent: {BRAND_ACCENT};
  --accent-rgb: 50,150,98; /* #329662 */
  --light: {BRAND_LIGHT};
}}
/* App background */
.stApp {{ background-color: var(--bg); }}

/* Base text on dark bg */
html, body, [class^="css"], [class*=" css"] {{ color: var(--light) !important; }}

/* Headings */
h1, h2, h3, h4, h5, h6 {{ color: var(--light) !important; }}

/* Inputs / selects / numbers: white surface with ink text */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--light) !important;
  color: var(--ink) !important;
  border-radius: 8px !important;
}}

/* Hand cursor for select + number inputs (including +/-) */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stNumberInput button {{ cursor: pointer !important; }}

/* Selectbox: caret inside + green focus */
.stSelectbox div[data-baseweb="select"] > div {{
  border: 2px solid var(--light) !important;
  position: relative;
}}
.stSelectbox div[data-baseweb="select"]:focus-within > div {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
}}
.stSelectbox div[data-baseweb="select"] > div::after {{
  content: "▾";
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--ink);
  pointer-events: none;
  font-size: 14px;
  font-weight: 700;
}}

/* Number inputs: ensure GREEN focus + blue steppers */
.stNumberInput input {{
  border: 2px solid var(--light) !important;
  outline: none !important;
}}
.stNumberInput input:focus,
.stNumberInput input:focus-visible {{
  outline: none !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
}}
.stNumberInput:focus-within input {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
}}
.stNumberInput button {{
  background: var(--ink) !important;        /* blue default */
  color: #ffffff !important;
  border: 1px solid var(--ink) !important;
}}
.stNumberInput button:hover,
.stNumberInput button:active,
.stNumberInput button:focus-visible {{
  background: var(--accent) !important;     /* green on interaction */
  color: #000 !important;
  border-color: var(--accent) !important;
}}

/* Text inputs: ensure GREEN focus */
.stTextInput input {
  border: 2px solid var(--light) !important;
  outline: none !important;
  border-radius: 8px !important;
}
.stTextInput input:focus,
.stTextInput input:focus-visible {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
  outline: none !important;
}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {{
  background: rgba(255,255,255,0.98);
  border: 2px dashed var(--accent);
}}
/* Text in uploader area is dark for readability */
[data-testid="stFileUploader"] * {{ color: var(--ink) !important; }}

/* “Browse files” button: blue default, transparent on hover (like Calculate) */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] [role="button"] {{
  background-color: var(--ink) !important;  /* #242F40 */
  color: #ffffff !important;
  border: 2px solid var(--ink) !important;
  border-radius: 8px !important;
  padding: 2px 10px !important;
  font-weight: 700 !important;
  transition: background-color .15s ease, color .15s ease, border-color .15s ease;
}}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] label:hover,
[data-testid="stFileUploaderDropzone"] [role="button"]:hover {{
  background-color: transparent !important; /* transparent on hover */
  color: var(--ink) !important;
  border-color: var(--ink) !important;
}}

/* Tables/readability */
.stDataFrame, .stDataFrame * , .stTable, .stTable * {{ color: var(--ink) !important; }}

/* Action buttons (download & calculate) — transparent on hover */
.stButton > button, .stDownloadButton > button {{
  background-color: var(--accent) !important;  /* green default */
  color: var(--ink) !important;
  border: 2px solid var(--light) !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 0 rgba(0,0,0,.15);
  transition: background-color .15s ease, color .15s ease, border-color .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background-color: transparent !important; /* transparent on hover */
  color: var(--light) !important;
  border-color: var(--accent) !important;
}}

/* Strategy banner helper */
.info-banner {{
  background: linear-gradient(90deg, var(--ink) 0%, var(--accent) 100%);
  padding: 16px; border-radius: 12px; color: var(--light);
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Title + tagline ----------
st.title("OutrankIQ")
st.caption("Score keywords by Search Volume (A) and Keyword Difficulty (B) — with selectable scoring strategies.")

# ---------- Helpers ----------
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        if any(k in c.lower() for k in candidates):
            return c
    return None

LABEL_MAP = {
    6: "Elite",
    5: "Excellent",
    4: "Good",
    3: "Fair",
    2: "Low",
    1: "Very Low",
    0: "Not rated",
}

COLOR_MAP = {
    6: "#2ecc71",
    5: "#a3e635",
    4: "#facc15",
    3: "#fb923c",
    2: "#f87171",
    1: "#ef4444",
    0: "#9ca3af",
}

strategy_descriptions = {
    "Low Hanging Fruit": "Keywords that can be used to rank quickly with minimal effort. Ideal for new content or low-authority sites. Try targeting long-tail keywords, create quick-win content, and build a few internal links.",
    "In The Game": "Moderate difficulty keywords that are within reach for growing sites. Focus on optimizing content, earning backlinks, and matching search intent to climb the ranks.",
    "Competitive": "High-volume, high-difficulty keywords dominated by authoritative domains. Requires strong content, domain authority, and strategic SEO to compete. Great for long-term growth.",
}

# ---------- Strategy selector ----------
scoring_mode = st.selectbox("Choose Scoring Strategy", ["Low Hanging Fruit", "In The Game", "Competitive"])

if scoring_mode == "Low Hanging Fruit":
    MIN_VALID_VOLUME = 10
    KD_BUCKETS = [(0, 15, 6), (16, 20, 5), (21, 25, 4), (26, 50, 3), (51, 75, 2), (76, 100, 1)]
elif scoring_mode == "In The Game":
    MIN_VALID_VOLUME = 1500
    KD_BUCKETS = [(0, 30, 6), (31, 45, 5), (46, 60, 4), (61, 70, 3), (71, 80, 2), (81, 100, 1)]
else:
    MIN_VALID_VOLUME = 3000
    KD_BUCKETS = [(0, 40, 6), (41, 60, 5), (61, 75, 4), (76, 85, 3), (86, 95, 2), (96, 100, 1)]

st.markdown(
    f"""
<div class="info-banner" style="margin-bottom:16px;">
  <div style='margin-bottom:6px; font-size:13px;'>
    Minimum Search Volume Required: <strong>{MIN_VALID_VOLUME}</strong>
  </div>
  <strong style='font-size:18px;'>{scoring_mode}</strong><br>
  <span style='font-size:15px;'>{strategy_descriptions[scoring_mode]}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Category tagging (multi-label) ----------
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|step[- ]?by[- ]?step|tutorial|guide|is)\b", re.I)
AEO_PAT = re.compile(r"^\s*(who|what|when|where|why|how|which|can|should)\b", re.I)
VEO_PAT = re.compile(r"\b(near me|open now|closest|call now|directions|ok google|alexa|siri|hey google)\b", re.I)
GEO_PAT = re.compile(r"\b(how to|best way to|steps? to|examples? of|checklist|framework|template)\b", re.I)
SXO_PAT = re.compile(r"\b(best|top|compare|comparison|vs\.?|review|pricing|cost|cheap|free download|template|examples?)\b", re.I)
LLM_PAT = re.compile(r"\b(prompt|prompting|prompt[- ]?engineering|chatgpt|gpt[- ]?\d|llm|rag|embedding|vector|few[- ]?shot|zero[- ]?shot)\b", re.I)
CATEGORY_ORDER = ["SEO", "AIO", "VEO", "GEO", "AEO", "SXO", "LLM"]

def categorize_keyword(kw: str) -> List[str]:
    if not isinstance(kw, str) or not kw.strip():
        return ["SEO"]
    text = kw.strip().lower()
    cats = set()
    if AIO_PAT.search(text): cats.add("AIO")
    if AEO_PAT.search(text): cats.add("AEO")
    if VEO_PAT.search(text): cats.add("VEO")
    if GEO_PAT.search(text): cats.add("GEO")
    if SXO_PAT.search(text): cats.add("SXO")
    if LLM_PAT.search(text): cats.add("LLM")
    if not cats:
        cats.add("SEO")
    else:
        if "LLM" not in cats:
            cats.add("SEO")
    return [c for c in CATEGORY_ORDER if c in cats]

# ---------- Scoring ----------
def calculate_score(volume: float, kd: float) -> int:
    if pd.isna(volume) or pd.isna(kd):
        return 0
    if volume < MIN_VALID_VOLUME:
        return 0
    kd = max(0.0, min(100.0, float(kd)))
    for low, high, score in KD_BUCKETS:
        if low <= kd <= high:
            return score
    return 0


def add_scoring_columns(df: pd.DataFrame, volume_col: str, kd_col: str, kw_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()

    def _eligibility_reason(vol, kd):
        if pd.isna(vol) or pd.isna(kd):
            return "No", "Invalid Volume/KD"
        if vol < MIN_VALID_VOLUME:
            return "No", f"Below min volume for {scoring_mode} ({MIN_VALID_VOLUME})"
        return "Yes", ""

    eligible, reason = zip(*(_eligibility_reason(v, k) for v, k in zip(out[volume_col], out[kd_col])))
    out["Eligible"] = list(eligible)
    out["Reason"] = list(reason)
    out["Score"] = [calculate_score(v, k) for v, k in zip(out[volume_col], out[kd_col])]
    out["Tier"] = out["Score"].map(LABEL_MAP).fillna("Not rated")

    kw_series = out[kw_col] if kw_col else pd.Series([""] * len(out), index=out.index)
    out["Category"] = [", ".join(categorize_keyword(str(k))) for k in kw_series]

    ordered = ([kw_col] if kw_col else []) + [volume_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]

# ---------- Tokenization & profile helpers ----------
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Dev-only defaults (not shown in UI)
_MAX_PAGES = 120
_MAX_BYTES = 350_000
_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 8
_TOTAL_BUDGET_SECS = 60
_CONCURRENCY = 16  # aiohttp
_THREADS = 12      # requests fallback
_MIN_FIT_THRESHOLD = 0.0

_DEF_HEADERS = {
    "User-Agent": "OutrankIQMapper/1.0 (+https://example.com)"
}

def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower()) if text else []


def _same_site(url: str, base_host: str, base_root: str, include_subdomains: bool) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        if include_subdomains:
            # Compare against the exact base host; include any subdomain of it
            return host == base_host or host.endswith("." + base_host)
        else:
            return host == base_host
    except Exception:
        return False


def _derive_roots(base_url: str) -> Tuple[str, str]:
    """Return (base_host, base_root) where base_root is eTLD+1-ish.
    We avoid extra deps; simple heuristic: last two labels.
    """
    host = urlparse(base_url).netloc.lower()
    parts = host.split(".")
    if len(parts) >= 2:
        base_root = ".".join(parts[-2:])
    else:
        base_root = host
    return host, base_root


def _normalize_base(base_url: str) -> str:
    if not base_url:
        return ""
    u = base_url.strip()
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    # remove path/query/fragment to stay at site root for discovery
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}"


# ---------- URL discovery ----------
@contextlib.contextmanager
def _session():
    if HAVE_AIOHTTP:
        yield None  # aiohttp created inside async
    else:
        yield requests.Session() if requests else None


def _fetch_text_requests(url: str, session, timeout: Tuple[int, int]) -> Optional[str]:
    try:
        if session is not None:
            resp = session.get(url, headers=_DEF_HEADERS, timeout=timeout, stream=True, allow_redirects=True)
        elif requests is not None:
            resp = requests.get(url, headers=_DEF_HEADERS, timeout=timeout, stream=True, allow_redirects=True)
        else:
            return None
        ctype = resp.headers.get("Content-Type", "").lower()
        if resp.status_code >= 400:
            return None
        # Accept HTML, text, XML, and gzipped sitemaps
        raw = resp.content[:_MAX_BYTES]
        if ("gzip" in ctype) or url.lower().endswith(".gz"):
            with contextlib.suppress(Exception):
                raw = gzip.decompress(raw)
            ctype = "text/xml"
        if ("text" not in ctype) and ("html" not in ctype) and ("xml" not in ctype):
            return None
        return raw.decode(resp.apparent_encoding or "utf-8", errors="ignore")
    except Exception:
        return None


def _extract_sitemaps_from_robots(base_root_url: str, session) -> List[str]:
    robots_url = urljoin(base_root_url + "/", "robots.txt")
    txt = _fetch_text_requests(robots_url, session, (_CONNECT_TIMEOUT, _READ_TIMEOUT))
    if not txt:
        return []
    maps = []
    for line in txt.splitlines():
        if line.lower().startswith("sitemap:"):
            sm = line.split(":", 1)[1].strip()
            if sm:
                maps.append(sm)
    return maps


def _parse_sitemap_xml(xml_text: str) -> List[str]:
    urls = []
    if not xml_text:
        return urls
    # Simple regex fallback to avoid heavy XML edge cases
    for m in re.finditer(r"<loc>\s*([^<]+)\s*</loc>", xml_text, re.I):
        urls.append(m.group(1).strip())
    return urls


def discover_urls(base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> List[str]:
    base = _normalize_base(base_url)
    if not base:
        return []
    base_host, base_root = _derive_roots(base)
    discovered: List[str] = []

    with _session() as sess:
        if use_sitemap_first:
            maps = _extract_sitemaps_from_robots(base, sess)
            if not maps:
                # Try common sitemap paths
                maps = [
                    urljoin(base + "/", "sitemap.xml"),
                    urljoin(base + "/", "sitemap_index.xml"),
                ]
            seen = set()
            for sm in maps:
                if len(discovered) >= _MAX_PAGES:
                    break
                xml = _fetch_text_requests(sm, sess, (_CONNECT_TIMEOUT, _READ_TIMEOUT))
                if not xml:
                    continue
                for u in _parse_sitemap_xml(xml):
                    if u in seen:
                        continue
                    if _same_site(u, base_host, base_root, include_subdomains):
                        seen.add(u)
                        discovered.append(u)
                        if len(discovered) >= _MAX_PAGES:
                            break
        # Fallback shallow crawl if still empty
        if not discovered:
            discovered = shallow_crawl(base, include_subdomains)

    return discovered[:_MAX_PAGES]


# ---------- Shallow crawl (depth ≤ 2) ----------
def _extract_links(html: str, current_url: str) -> List[str]:
    links: List[str] = []
    if not html:
        return links
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                links.append(urljoin(current_url, a["href"]))
            return links
        except Exception:
            pass
    # regex fallback
    for m in re.finditer(r"href=\"([^\"]+)\"|href='([^']+)'", html, re.I):
        href = m.group(1) or m.group(2)
        if href:
            links.append(urljoin(current_url, href))
    return links


def shallow_crawl(base_url: str, include_subdomains: bool) -> List[str]:
    base = _normalize_base(base_url)
    base_host, base_root = _derive_roots(base)
    start = base
    frontier = [(start, 0)]
    seen = set([start])
    out: List[str] = []

    def ok(u: str) -> bool:
        if not _same_site(u, base_host, base_root, include_subdomains):
            return False
        p = urlparse(u)
        if not p.scheme.startswith("http"):
            return False
        if any(u.lower().endswith(ext) for ext in (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".rar")):
            return False
        return True

    if HAVE_AIOHTTP:
        async def _run():
            timeout = aiohttp.ClientTimeout(total=_TOTAL_BUDGET_SECS, sock_connect=_CONNECT_TIMEOUT, sock_read=_READ_TIMEOUT)
            conn = aiohttp.TCPConnector(limit=_CONCURRENCY, ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=conn, headers=_DEF_HEADERS) as session:
                idx = 0
                while frontier and len(out) < _MAX_PAGES:
                    url, depth = frontier.pop(0)
                    idx += 1
                    try:
                        async with session.get(url, allow_redirects=True) as resp:
                            if resp.status >= 400:
                                continue
                            ctype = resp.headers.get("Content-Type", "").lower()
                            if "html" not in ctype and "text" not in ctype:
                                continue
                            b = await resp.content.read(_MAX_BYTES)
                            html = b.decode(errors="ignore")
                    except Exception:
                        continue
                    out.append(url)
                    if depth < 2 and len(out) < _MAX_PAGES:
                        for link in _extract_links(html, url):
                            if link not in seen and ok(link):
                                seen.add(link)
                                frontier.append((link, depth + 1))
            return out
        try:
            return asyncio.run(_run())[:_MAX_PAGES]
        except RuntimeError:
            # Fallback to requests crawl when an event loop is already running
            if not requests:
                return [base]
            sess = requests.Session()
            try:
                while frontier and len(out) < _MAX_PAGES:
                    url, depth = frontier.pop(0)
                    try:
                        r = sess.get(url, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
                        if r.status_code >= 400:
                            continue
                        ctype = r.headers.get("Content-Type", "").lower()
                        if "html" not in ctype and "text" not in ctype:
                            continue
                        html = r.content[:_MAX_BYTES].decode(r.apparent_encoding or "utf-8", errors="ignore")
                    except Exception:
                        continue
                    out.append(url)
                    if depth < 2 and len(out) < _MAX_PAGES:
                        for link in _extract_links(html, url):
                            if link not in seen and ok(link):
                                seen.add(link)
                                frontier.append((link, depth + 1))
            finally:
                sess.close()
            return out[:_MAX_PAGES]
    else:
        if not requests:
            return [base]
        sess = requests.Session()
        try:
            while frontier and len(out) < _MAX_PAGES:
                url, depth = frontier.pop(0)
                try:
                    r = sess.get(url, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
                    if r.status_code >= 400:
                        continue
                    ctype = r.headers.get("Content-Type", "").lower()
                    if "html" not in ctype and "text" not in ctype:
                        continue
                    html = r.content[:_MAX_BYTES].decode(r.apparent_encoding or "utf-8", errors="ignore")
                except Exception:
                    continue
                out.append(url)
                if depth < 2 and len(out) < _MAX_PAGES:
                    for link in _extract_links(html, url):
                        if link not in seen and ok(link):
                            seen.add(link)
                            frontier.append((link, depth + 1))
        finally:
            sess.close()
        return out[:_MAX_PAGES]


# ---------- Content profiling ----------
def _extract_profile(html: str, url: str) -> Dict:
    title = ""
    h1_texts: List[str] = []
    h2h3_texts: List[str] = []
    body_text = ""
    canonical = ""

    if HAVE_BS4 and html:
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Title
            t = soup.find("title")
            title = t.get_text(" ", strip=True) if t else ""
            # Canonical
            link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
            if link and link.get("href"):
                canonical = urljoin(url, link["href"])  # resolve relative
            # Headings
            for h in soup.find_all(["h1", "h2", "h3"]):
                txt = h.get_text(" ", strip=True)
                if not txt:
                    continue
                if h.name == "h1":
                    h1_texts.append(txt)
                else:
                    h2h3_texts.append(txt)
            # Body (first ~800 words)
            for tag in soup(["script", "style", "noscript", "template", "nav", "footer", "header", "aside"]):
                tag.extract()
            body_text = soup.get_text(" ", strip=True)
        except Exception:
            pass
    # Fallback naive extraction
    if not title:
        m = re.search(r"<title>(.*?)</title>", html or "", re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
    if not canonical:
        canonical = url

    # Trim body to ~800 words
    if body_text:
        words = body_text.split()
        if len(words) > 800:
            body_text = " ".join(words[:800])

    weights: Dict[str, float] = defaultdict(float)
    for tok in _tokenize(title):
        weights[tok] += 3.0
    for t in h1_texts:
        for tok in _tokenize(t):
            weights[tok] += 2.0
    for t in h2h3_texts:
        for tok in _tokenize(t):
            weights[tok] += 1.5
    for tok in _tokenize(body_text):
        weights[tok] += 1.0

    title_h1 = " ".join([title] + h1_texts)

    return {
        "url": canonical or url,
        "title": title or "",
        "title_h1": title_h1.lower(),
        "weights": dict(weights),
    }


def _fetch_profiles(urls: List[str]) -> List[Dict]:
    profiles: List[Dict] = []
    if not urls:
        return profiles

    if HAVE_AIOHTTP:
        async def _run():
            timeout = aiohttp.ClientTimeout(total=_TOTAL_BUDGET_SECS, sock_connect=_CONNECT_TIMEOUT, sock_read=_READ_TIMEOUT)
            conn = aiohttp.TCPConnector(limit=_CONCURRENCY, ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=conn, headers=_DEF_HEADERS) as session:
                sem = asyncio.Semaphore(_CONCURRENCY)
                async def fetch(u: str):
                    async with sem:
                        try:
                            async with session.get(u, allow_redirects=True) as resp:
                                if resp.status >= 400:
                                    return None
                                ctype = resp.headers.get("Content-Type", "").lower()
                                if "html" not in ctype and "text" not in ctype:
                                    return None
                                b = await resp.content.read(_MAX_BYTES)
                                html = b.decode(errors="ignore")
                                return _extract_profile(html, str(resp.url))
                        except Exception:
                            return None
                tasks = [fetch(u) for u in urls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, dict):
                        profiles.append(r)
            return profiles
        try:
            return asyncio.run(_run())
        except RuntimeError:
            # Fallback to requests when an event loop is already running
            if not requests:
                return profiles
            sess = requests.Session()
            try:
                for u in urls[:_MAX_PAGES]:
                    try:
                        r = sess.get(u, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
                        if r.status_code >= 400:
                            continue
                        ctype = r.headers.get("Content-Type", "").lower()
                        if "html" not in ctype and "text" not in ctype:
                            continue
                        html = r.content[:_MAX_BYTES].decode(r.apparent_encoding or "utf-8", errors="ignore")
                        prof = _extract_profile(html, str(r.url))
                        if prof and prof.get("weights"):
                            profiles.append(prof)
                    except Exception:
                        continue
            finally:
                sess.close()
            return profiles
    else:
        if not requests:
            return profiles
        sess = requests.Session()
        try:
            for u in urls[:_MAX_PAGES]:
                try:
                    r = sess.get(u, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT), allow_redirects=True)
                    if r.status_code >= 400:
                        continue
                    ctype = r.headers.get("Content-Type", "").lower()
                    if "html" not in ctype and "text" not in ctype:
                        continue
                    html = r.content[:_MAX_BYTES].decode(r.apparent_encoding or "utf-8", errors="ignore")
                    profiles.append(_extract_profile(html, str(r.url)))
                except Exception:
                    continue
        finally:
            sess.close()
        return profiles


# ---------- Fit scoring ----------
def _fit_score(keyword: str, profile: Dict) -> float:
    tokens = _tokenize(keyword)
    if not tokens:
        return 0.0
    w = profile.get("weights", {})
    overlap = sum(w.get(t, 0.0) for t in tokens) / max(1, len(tokens))
    title_h1 = profile.get("title_h1", "")
    # Title/H1 boosts
    covered = sum(1 for t in tokens if t in title_h1)
    if covered == len(tokens):
        overlap += 0.25
    elif covered / len(tokens) >= 0.5:
        overlap += 0.10
    # Phrase hint
    phrase = " ".join(tokens)
    if phrase and phrase in title_h1:
        overlap += 0.15
    return max(0.0, min(2.0, overlap))


# ---------- Mapping algorithm ----------
def map_keywords_to_urls(df: pd.DataFrame, kw_col: Optional[str], vol_col: str, base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> pd.Series:
    # Prepare pages
    url_list = discover_urls(base_url, include_subdomains=include_subdomains, use_sitemap_first=use_sitemap_first)
    profiles = _fetch_profiles(url_list)
    profiles = [p for p in profiles if p.get("weights")]
    if not profiles:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    # Precompute best URL per keyword
    best_for_kw: Dict[int, Tuple[str, float, str]] = {}

    for idx, row in df.iterrows():
        kw = str(row.get(kw_col, "")) if kw_col else str(row.get("Keyword", ""))
        cats = set(categorize_keyword(kw))
        # Slot priority: VEO > AIO > SEO
        if "VEO" in cats:
            slot = "VEO"
        elif "AIO" in cats:
            slot = "AIO"
        else:
            slot = "SEO"
        best_url = ""
        best_fit = 0.0
        for p in profiles:
            f = _fit_score(kw, p)
            if f > best_fit:
                best_fit = f
                best_url = p.get("url", "")
        if best_fit >= _MIN_FIT_THRESHOLD and best_url:
            best_for_kw[idx] = (best_url, best_fit, slot)
        else:
            best_for_kw[idx] = ("", 0.0, slot)

    # Build candidate buckets per URL
    candidates = defaultdict(lambda: {"VEO": [], "AIO": [], "SEO": []})
    for idx, (u, f, slot) in best_for_kw.items():
        if not u:
            continue
        score = float(df.loc[idx, "Score"]) if "Score" in df.columns else 0.0
        vol = float(df.loc[idx, vol_col]) if vol_col in df.columns else 0.0
        candidates[u][slot].append((idx, f, score, vol))

    assigned: Dict[str, Dict[str, List[int] | Optional[int]]] = {}
    for u in candidates.keys():
        assigned[u] = {"VEO": None, "AIO": None, "SEO": []}

    mapped = {i: "" for i in df.index}

    # VEO pass
    for u, bucket in candidates.items():
        veos = sorted(bucket["VEO"], key=lambda x: (-x[1], -x[2], -x[3]))
        if veos:
            idx, *_ = veos[0]
            assigned[u]["VEO"] = idx
            mapped[idx] = u

    # AIO pass
    for u, bucket in candidates.items():
        if assigned[u]["VEO"] is not None and len(candidates[u]["SEO"]) >= 2:
            # capacity check later; still allow AIO if room remains after SEO
            pass
        aios = sorted(bucket["AIO"], key=lambda x: (-x[1], -x[2], -x[3]))
        if aios:
            idx, *_ = aios[0]
            # ensure not already used by VEO assignment elsewhere (a keyword can only map once)
            if not mapped.get(idx):
                assigned[u]["AIO"] = idx
                mapped[idx] = u

    # SEO pass (Primary + Secondary per URL)
    for u, bucket in candidates.items():
        seos = sorted(bucket["SEO"], key=lambda x: (-x[2], -x[1], -x[3]))  # Score desc, then fit desc, then volume desc
        take = []
        for item in seos:
            if len(take) >= 2:
                break
            idx = item[0]
            if not mapped.get(idx):
                take.append(idx)
        assigned[u]["SEO"] = take
        for idx in take:
            mapped[idx] = u

    # Capacity check: max 4 per URL (1 VEO, 1 AIO, 2 SEO). If exceeded (shouldn't), trim SEO extras.
    for u in list(assigned.keys()):
        cnt = (1 if assigned[u]["VEO"] is not None else 0) + (1 if assigned[u]["AIO"] is not None else 0) + len(assigned[u]["SEO"])  # type: ignore
        if cnt > 4:
            # Trim SEO
            extras = cnt - 4
            while extras > 0 and assigned[u]["SEO"]:
                drop_idx = assigned[u]["SEO"].pop()  # type: ignore
                mapped[drop_idx] = ""
                extras -= 1

    return pd.Series([mapped[i] for i in df.index], index=df.index, dtype="string")


# ---------- Single Keyword ----------
st.subheader("Single Keyword Score")
with st.form("single"):
    c1, c2 = st.columns(2)
    with c1:
        vol_val = st.number_input("Search Volume (A)", min_value=0, step=10, value=0)
    with c2:
        kd_val = st.number_input("Keyword Difficulty (B)", min_value=0, step=1, value=0)

    if st.form_submit_button("Calculate Score"):
        sc = calculate_score(vol_val, kd_val)
        label = LABEL_MAP.get(sc, "Not rated")
        color = COLOR_MAP.get(sc, "#9ca3af")
        st.markdown(
            f"""
            <div style='background-color:{color}; padding:16px; border-radius:12px; text-align:center;'>
                <span style='font-size:22px; font-weight:bold; color:#000;'>Score: {sc} • Tier: {label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if vol_val < MIN_VALID_VOLUME:
            st.warning(f"The selected strategy requires a minimum search volume of {MIN_VALID_VOLUME}. Please enter a volume that meets the threshold.")

st.markdown("---")
st.subheader("Bulk Scoring (CSV Upload)")

# Mapping controls (minimal UI)
base_site_url = st.text_input("Base site URL (for URL mapping)", placeholder="https://example.com")
use_sitemap_first = True
include_subdomains = True

uploaded = st.file_uploader("Upload CSV", type=["csv"])
example = pd.DataFrame({"Keyword":["best running shoes","seo tools","crm software"], "Volume":[5400,880,12000], "KD":[38,72,18]})
with st.expander("See example CSV format"):
    st.dataframe(example, use_container_width=True)

# ---------- Robust CSV reader + numeric cleaning ----------
if uploaded is not None:
    raw = uploaded.getvalue()

    def try_read(bytes_data: bytes) -> pd.DataFrame:
        trials = [
            {"encoding": None, "sep": None, "engine": "python"},
            {"encoding": "utf-8", "sep": None, "engine": "python"},
            {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
            {"encoding": "ISO-8859-1", "sep": None, "engine": "python"},
            {"encoding": "cp1252", "sep": None, "engine": "python"},
            {"encoding": "utf-16", "sep": None, "engine": "python"},
            {"encoding": None, "sep": ",", "engine": "python"},
            {"encoding": None, "sep": "\t", "engine": "python"},
        ]
        last_err = None
        for t in trials:
            try:
                return pd.read_csv(io.BytesIO(bytes_data), **{k:v for k,v in t.items() if v is not None})
            except Exception as e:
                last_err = e
        raise last_err

    try:
        df = try_read(raw)
    except Exception:
        st.error("Could not read the file. Please ensure it's a CSV (or TSV) exported from Excel/Sheets and try again.")
        st.stop()

    vol_col = find_column(df, ["volume","search volume","sv"])
    kd_col  = find_column(df, ["kd","difficulty","keyword difficulty"])
    kw_col  = find_column(df, ["keyword","query","term"])

    missing = []
    if vol_col is None: missing.append("Volume")
    if kd_col  is None: missing.append("Keyword Difficulty")

    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
    else:
        # Clean numbers
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\s]","",regex=True).str.replace("%","",regex=False)
        df[kd_col]  = df[kd_col].astype(str).str.replace(r"[,\s]","",regex=True).str.replace("%","",regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col]  = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

        scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

        # ---------- CSV DOWNLOAD ----------
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes":1,"No":0}).fillna(0)
        export_df = export_df.sort_values(
            by=["_EligibleSort", kd_col, vol_col],
            ascending=[False, True, False],
            kind="mergesort"
        ).drop(columns=["_EligibleSort"])

        # -------- URL Mapping (appends last column) --------
        map_series = pd.Series([""] * len(export_df), index=export_df.index, dtype="string")
        if base_site_url.strip():
            loader = st.empty()
            loader.markdown(
                """
                <div style='display:flex;align-items:center;gap:12px;'>
                  <div style='font-size:28px'>🚀</div>
                  <div style='font-weight:700;'>Mapping keywords to your site…</div>
                </div>
                <style>
                  @keyframes bob { from { transform: translateY(0); } to { transform: translateY(-6px); } }
                  div[style*="font-size:28px"] { animation: bob .6s ease-in-out infinite alternate; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("Launching fast crawl & scoring fit…"):
                map_series = map_keywords_to_urls(
                    export_df,
                    kw_col=kw_col,
                    vol_col=vol_col,
                    base_url=base_site_url.strip(),
                    include_subdomains=include_subdomains,
                    use_sitemap_first=use_sitemap_first,
                )
            loader.empty()
        else:
            st.info("Enter a Base site URL to enable mapping.")

        export_df["Map URL"] = map_series

        export_cols = base_cols + ["Strategy", "Map URL"]  # Map URL as rightmost column
        export_df = export_df[export_cols]

        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download scored CSV",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            help="Sorted by eligibility (Yes first), KD ascending, Volume descending"
        )

st.markdown("---")
st.caption(f"© {datetime.now().year} OutrankIQ")
