# app.py  (OutrankIQ)
# Full code — site-agnostic sitemap-first mapping with dynamic thesaurus.
# IMPORTANT: Concept overrides (careers/program) are ENABLED ONLY for "In The Game".
# Competitive uses generic mapping (no concept nudges / relaxed acceptance).

import io
import re
import gzip
import math
import asyncio
import contextlib
import hashlib
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict
from urllib.parse import urlparse, urljoin, unquote

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
except Exception:
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # noqa: F401
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False


# ---------- Brand / Theme ----------
BRAND_BG = "#FFFFFF"     # white background
BRAND_INK = "#242F40"    # "blue" ink
BRAND_ACCENT = "#329662" # green
BRAND_LIGHT = "#FFFFFF"  # white

st.set_page_config(page_title="OutrankIQ", page_icon="🔎", layout="wide")

# ---------- Global CSS ----------
st.markdown(
    f"""
<style>
:root {{
  --bg: {BRAND_BG};
  --ink: {BRAND_INK};
  --accent: {BRAND_ACCENT};
  --accent-rgb: 50,150,98;
  --light: {BRAND_LIGHT};
}}

/* App background */
.stApp {{
  background-color: var(--bg);
}}

/* Full-bleed green header bar */
.header-wrap {{
  width: 100vw;
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  background: var(--accent);
  color: #fff;
  padding: 28px 0 18px 0;
}}
.header-inner {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 24px;
}}
.header-title {{
  font-size: 40px; 
  font-weight: 800; 
  letter-spacing: .3px;
  line-height: 1.05;
  margin: 0;
}}
.header-sub {{
  font-size: 16px; 
  opacity: .95;
  margin-top: 6px;
}}

/* Base text on white */
html, body, [class^="css"], [class*=" css"] {{ color: var(--ink) !important; }}

/* Headings */
h1, h2, h3, h4, h5, h6 {{ color: var(--ink) !important; }}

/* Inputs — white surface with INK borders and focus also INK (blue) */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--light) !important;
  color: var(--ink) !important;
  border-radius: 10px !important;
  border: 2px solid var(--ink) !important;  /* blue border default */
}}

/* Hand cursor for select + number inputs (including +/-) */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stNumberInput button {{ cursor: pointer !important; }}

/* Selectbox: keep caret and focus in INK (blue) */
.stSelectbox div[data-baseweb="select"] > div {{
  position: relative;
  box-shadow: none !important;
}}
.stSelectbox div[data-baseweb="select"]:focus-within > div {{
  border-color: var(--ink) !important;           /* blue focus */
  box-shadow: 0 0 0 3px rgba(36,47,64,.20) !important; /* faint blue glow */
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

/* Number inputs: steppers + focus in blue */
.stNumberInput input {{
  outline: none !important;
}}
.stNumberInput input:focus,
.stNumberInput input:focus-visible {{
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 3px rgba(36,47,64,.20) !important;
}}
.stNumberInput:focus-within input {{
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 3px rgba(36,47,64,.20) !important;
}}
.stNumberInput button {{
  background: var(--ink) !important;        
  color: #ffffff !important;
  border: 1px solid var(--ink) !important;
}}
.stNumberInput button:hover,
.stNumberInput button:active,
.stNumberInput button:focus-visible {{
  background: var(--accent) !important;     
  color: #000 !important;
  border-color: var(--accent) !important;
}}

/* Text inputs: focus in blue */
.stTextInput input {{
  outline: none !important;
}}
.stTextInput input:focus,
.stTextInput input:focus-visible {{
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 3px rgba(36,47,64,.20) !important;
  outline: none !important;
}}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {{
  background: rgba(255,255,255,0.98);
  border: 2px dashed var(--accent);
}}
/* Text in uploader area */
[data-testid="stFileUploader"] * {{ color: var(--ink) !important; }}

/* “Browse files” button: INK default, invert on hover */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] [role="button"] {{
  background-color: var(--ink) !important;  
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
  background-color: transparent !important; 
  color: var(--ink) !important;
  border-color: var(--ink) !important;
}}

/* Tables/readability */
.stDataFrame, .stDataFrame * , .stTable, .stTable * {{ color: var(--ink) !important; }}

/* Action buttons (Calculate & Download): green default; on hover -> blue border + blue text */
.stButton > button, .stDownloadButton > button {{
  background-color: var(--accent) !important;  
  color: #ffffff !important;
  border: 2px solid var(--accent) !important;
  border-radius: 10px !important;
  font-weight: 800 !important;
  box-shadow: 0 2px 0 rgba(0,0,0,.15);
  transition: background-color .15s ease, color .15s ease, border-color .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background-color: transparent !important; 
  color: var(--ink) !important;               /* blue font on hover */
  border-color: var(--ink) !important;        /* blue border on hover */
}}

/* Primary Map button: force-in-blue, invert on hover */
.map-btn > button {{
  background-color: var(--ink) !important;
  color: #fff !important;
  border: 2px solid var(--ink) !important;
}}
.map-btn > button:hover {{
  background-color: transparent !important;
  color: var(--ink) !important;
  border-color: var(--ink) !important;
}}

/* Expander header (example CSV): make label + arrow blue/visible */
[data-testid="stExpander"] summary {{ color: var(--ink) !important; }}
[data-testid="stExpander"] summary svg {{ stroke: var(--ink) !important; color: var(--ink) !important; }}

/* Strategy banner helper */
.info-banner {{
  background: linear-gradient(90deg, var(--ink) 0%, var(--accent) 100%);
  padding: 14px; border-radius: 12px; color: var(--light);
}}

/* Blue loader text + blue circle */
.loader-row {{
  display:flex; align-items:center; gap:12px; color: var(--ink);
  font-weight:700;
}}
.loader-circle {{
  width:18px; height:18px; border-radius:50%;
  border:3px solid rgba(36,47,64,.35);
  border-top-color: var(--ink);
  animation: spin .8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
<div class="header-wrap">
  <div class="header-inner">
    <h1 class="header-title">OutrankIQ</h1>
    <div class="header-sub">Score keywords by Search Volume (A) and Keyword Difficulty (B) — with selectable scoring strategies.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

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


LABEL_MAP = {6:"Elite",5:"Excellent",4:"Good",3:"Fair",2:"Low",1:"Very Low",0:"Not rated"}
COLOR_MAP = {6:"#2ecc71",5:"#a3e635",4:"#facc15",3:"#fb923c",2:"#f87171",1:"#ef4444",0:"#9ca3af"}

strategy_descriptions = {
    "Low Hanging Fruit": "Keywords that can be used to rank quickly with minimal effort. Ideal for new content or low-authority sites. Try targeting long-tail keywords, create quick-win content, and build a few internal links.",
    "In The Game": "Moderate difficulty keywords that are within reach for growing sites. Focus on optimizing content, earning backlinks, and matching search intent to climb the ranks.",
    "Competitive": "High-volume, high-difficulty keywords dominated by authoritative domains. Requires strong content, domain authority, and strategic SEO to compete. Great for long-term growth.",
}

# ---------- Strategy selector ----------
st.subheader("Choose Scoring Strategy")
scoring_mode = st.selectbox("", ["Low Hanging Fruit", "In The Game", "Competitive"], label_visibility="collapsed")

if scoring_mode == "Low Hanging Fruit":
    MIN_VALID_VOLUME = 10
    KD_BUCKETS = [(0,15,6),(16,20,5),(21,25,4),(26,50,3),(51,75,2),(76,100,1)]
elif scoring_mode == "In The Game":
    MIN_VALID_VOLUME = 1500
    KD_BUCKETS = [(0,30,6),(31,45,5),(46,60,4),(61,70,3),(71,80,2),(81,100,1)]
else:
    MIN_VALID_VOLUME = 3000
    KD_BUCKETS = [(0,40,6),(41,60,5),(61,75,4),(76,85,3),(86,95,2),(96,100,1)]

st.markdown(
    f"""
<div class="info-banner" style="margin:12px 0 18px 0;">
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
VEO_PAT = re.compile(r"\b(near me|open now|closest|call now|directions|ok google|alexa|siri|hey google)\b", re.I)  # Voice Engine Optimization
GEO_PAT = re.compile(r"\b(how to|best way to|steps? to|examples? of|checklist|framework|template)\b", re.I)
SXO_PAT = re.compile(r"\b(best|top|compare|comparison|vs\.?|review|pricing|cost|cheap|free download|template|examples?)\b", re.I)
LLM_PAT = re.compile(r"\b(prompt|prompting|prompt[- ]?engineering|chatgpt|gpt[- ]?\d|llm|rag|embedding|vector|few[- ]?shot|zero[- ]?shot)\b", re.I)
CATEGORY_ORDER = ["SEO","AIO","VEO","GEO","AEO","SXO","LLM"]

def categorize_keyword(kw: str) -> List[str]:
    if not isinstance(kw, str) or not kw.strip():
        return ["SEO"]
    text = kw.strip().lower()
    cats = set()
    if AIO_PAT.search(text): cats.add("AIO")
    if AEO_PAT.search(text): cats.add("AEO")
    if VEO_PAT.search(text): cats.add("VEO")  # voice
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

    eligible, reason = zip(*(_eligibility_reason(v,k) for v,k in zip(out[volume_col], out[kd_col])))
    out["Eligible"] = list(eligible)
    out["Reason"] = list(reason)
    out["Score"] = [calculate_score(v,k) for v,k in zip(out[volume_col], out[kd_col])]
    out["Tier"]  = out["Score"].map(LABEL_MAP).fillna("Not rated")

    kw_series = out[kw_col] if kw_col else pd.Series([""]*len(out), index=out.index)
    out["Category"] = [", ".join(categorize_keyword(str(k))) for k in kw_series]

    ordered = ([kw_col] if kw_col else []) + [volume_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]


# ---------- Tokenization & normalizers ----------
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower()) if text else []

def _normalize_token(tok: str) -> str:
    t = tok.lower()
    t = re.sub(r"(ing|ed|es|s)$", "", t)  # cheap stems
    return t

def _head_noun(tokens: List[str]) -> str:
    for t in reversed(tokens):
        if len(t) > 2:
            return _normalize_token(t)
    return tokens[-1] if tokens else ""


# ---------- Dev-only defaults ----------
_MAX_PAGES = 140
_MAX_BYTES = 400_000
_CONNECT_TIMEOUT = 6
_READ_TIMEOUT = 9
_TOTAL_BUDGET_SECS = 65
_CONCURRENCY = 16
_THREADS = 12
_ALT_FIT_MIN = 0.20  # generic fallback minimum if not top-1

_DEF_HEADERS = {"User-Agent": "OutrankIQMapper/1.1 (+https://example.com)"}

# Role lexicon (site-agnostic, extensible)
ROLE_TOKS = {
    "paraprofessional","aide","assistant","associate","coordinator","specialist","therapist",
    "technician","teacher","educator","counselor","clinician","caregiver","pca","dsp",
    "driver","instructor","advocate","manager","supervisor","director","analyst",
    "recruiter","caseworker","behavior","bcba","ot","pt","slp","rn","lpn","administrator"
}

# Program/service seeds (canonical cues)
PROGRAM_SEEDS = {
    "waiver","cadi","cdcs","ltss","pca","respite","housing","employment","transportation",
    "day","habilitation","therapy","head","early","intervention","iep","504","guardianship",
    "management","vocational","rehabilitation","ssi","medicaid","tefra","case"
}

CONTACT_CUES = {"contact","connect","touch","call","email","location","directions","visit"}
DONATE_CUES  = {"donate","give","support","contribute","donation","gift","ways","fund"}
ABOUT_CUES   = {"about","mission","history","story","leadership","team","board","values"}
CAREER_CUES  = {"career","careers","job","jobs","employment","openings","positions","vacancies","hiring","apply","join","team","work"}

# ---------- Site normalization ----------
def _same_site(url: str, base_host: str, base_root: str, include_subdomains: bool) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        apex = base_root
        www_host = f"www.{base_root}"
        if include_subdomains:
            if host in {base_host, apex, www_host}:
                return True
            return host.endswith("." + base_root)
        else:
            return host in {base_host, apex, www_host}
    except Exception:
        return False

def _derive_roots(base_url: str) -> Tuple[str, str]:
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
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}"


# ---------- HTTP helpers ----------
@contextlib.contextmanager
def _session():
    if HAVE_AIOHTTP:
        yield None
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
        if resp.status_code >= 400:
            return None
        ctype = resp.headers.get("Content-Type", "").lower()
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


# ---------- Sitemaps (page-first) ----------
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

_LOC_RE = re.compile(r"<loc>\s*([^<]+)\s*</loc>", re.I)
def _parse_sitemap_xml(xml_text: str) -> List[str]:
    if not xml_text:
        return []
    return [m.group(1).strip() for m in _LOC_RE.finditer(xml_text)]

def _is_page_sitemap(url: str) -> bool:
    u = url.lower()
    return ("page-sitemap" in u) or u.endswith("/page-sitemap.xml")

def _is_post_sitemap(url: str) -> bool:
    u = url.lower()
    return ("post-sitemap" in u) or ("news" in u) or ("blog" in u)

def _collect_sitemap_urls(sm_url: str, session, base_host: str, base_root: str, include_subdomains: bool, seen: set, out: List[str], depth: int = 0):
    if depth > 4 or len(out) >= _MAX_PAGES or sm_url in seen:
        return
    seen.add(sm_url)
    xml = _fetch_text_requests(sm_url, session, (_CONNECT_TIMEOUT, _READ_TIMEOUT))
    if not xml:
        return
    locs = _parse_sitemap_xml(xml)
    is_index = bool(re.search(r"<sitemapindex", xml, re.I))
    if is_index:
        children = sorted(locs, key=lambda u: (0 if _is_page_sitemap(u) else (1 if not _is_post_sitemap(u) else 2), u))
        for u in children:
            if len(out) >= _MAX_PAGES:
                break
            _collect_sitemap_urls(u, session, base_host, base_root, include_subdomains, seen, out, depth + 1)
    else:
        for u in locs:
            if len(out) >= _MAX_PAGES:
                break
            if _same_site(u, base_host, base_root, include_subdomains):
                out.append(u)

def discover_urls(base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> List[str]:
    base = _normalize_base(base_url)
    if not base:
        return []
    base_host, base_root = _derive_roots(base)
    discovered: List[str] = []

    with _session() as sess:
        if use_sitemap_first:
            maps = _extract_sitemaps_from_robots(base, sess)
            maps = list(dict.fromkeys(maps + [
                urljoin(base + "/", "sitemap.xml"),
                urljoin(base + "/", "sitemap_index.xml"),
                urljoin(base + "/", "page-sitemap.xml"),
            ]))
            maps = sorted(maps, key=lambda u: (0 if _is_page_sitemap(u) else (1 if not _is_post_sitemap(u) else 2), u))
            seen = set()
            for sm in maps:
                if len(discovered) >= _MAX_PAGES:
                    break
                _collect_sitemap_urls(sm, sess, base_host, base_root, include_subdomains, seen, discovered, depth=0)

        if not discovered:
            discovered = shallow_crawl(base, include_subdomains)

    seenu = set()
    uniq = []
    for u in discovered:
        if u not in seenu:
            uniq.append(u); seenu.add(u)
    return uniq[:_MAX_PAGES]


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
    for m in re.finditer(r'href="([^"]+)"|href=\'([^\']+)\'', html, re.I):
        href = m.group(1) or m.group(2)
        if href:
            links.append(urljoin(current_url, href))
    return links

def shallow_crawl(base_url: str, include_subdomains: bool) -> List[str]:
    base = _normalize_base(base_url)
    base_host, base_root = _derive_roots(base)
    frontier = [(base, 0)]
    seen = set([base])
    out: List[str] = []

    def ok(u: str) -> bool:
        if not _same_site(u, base_host, base_root, include_subdomains):
            return False
        p = urlparse(u)
        if not p.scheme.startswith("http"):
            return False
        if any(u.lower().endswith(ext) for ext in (".pdf",".jpg",".jpeg",".png",".gif",".webp",".svg",".zip",".rar",".mp4",".avi",".mov",".wmv")):
            return False
        return True

    if HAVE_AIOHTTP:
        async def _run():
            timeout = aiohttp.ClientTimeout(total=_TOTAL_BUDGET_SECS, sock_connect=_CONNECT_TIMEOUT, sock_read=_READ_TIMEOUT)
            conn = aiohttp.TCPConnector(limit=_CONCURRENCY, ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=conn, headers=_DEF_HEADERS) as session:
                while frontier and len(out) < _MAX_PAGES:
                    url, depth = frontier.pop(0)
                    try:
                        async with session.get(url, allow_redirects=True) as resp:
                            if resp.status >= 400:
                                continue
                            ctype = resp.headers.get("Content-Type","").lower()
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
                                frontier.append((link, depth+1))
            return out
        try:
            return asyncio.run(_run())[:_MAX_PAGES]
        except RuntimeError:
            pass

    if not requests:
        return [base]
    sess = requests.Session()
    try:
        while frontier and len(out) < _MAX_PAGES:
            url, depth = frontier.pop(0)
            try:
                r = sess.get(url, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT,_READ_TIMEOUT), allow_redirects=True)
                if r.status_code >= 400:
                    continue
                ctype = r.headers.get("Content-Type","").lower()
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
                        frontier.append((link, depth+1))
    finally:
        sess.close()
    return out[:_MAX_PAGES]


# ---------- Page profiling ----------
def _path_tokens(u: str) -> List[str]:
    path = unquote(urlparse(u).path.lower())
    parts = [p for p in path.strip("/").split("/") if p]
    toks = []
    for p in parts:
        toks.extend(_tokenize(p.replace("-", " ")))
    return toks

def _is_probably_post(u: str, title: str) -> bool:
    u2 = u.lower()
    if any(seg in u2 for seg in ("/category/","/tag/","/tags/","/blog/","/news/","/event","/events/")):
        return True
    if re.search(r"/20\d{2}/\d{2}/", u2):  # date paths
        return True
    return False

def _extract_profile(html: str, url: str) -> Dict:
    title = ""
    h1_texts: List[str] = []
    h2h3_texts: List[str] = []
    body_text = ""
    canonical = ""

    if HAVE_BS4 and html:
        try:
            soup = BeautifulSoup(html, "html.parser")
            t = soup.find("title")
            title = t.get_text(" ", strip=True) if t else ""
            link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
            if link and link.get("href"):
                canonical = urljoin(url, link["href"])
            for h in soup.find_all(["h1","h2","h3"]):
                txt = h.get_text(" ", strip=True)
                if not txt: continue
                if h.name == "h1": h1_texts.append(txt)
                else: h2h3_texts.append(txt)
            for tag in soup(["script","style","noscript","template","nav","footer","header","aside"]):
                tag.extract()
            body_text = soup.get_text(" ", strip=True)
        except Exception:
            pass
    if not title:
        m = re.search(r"<title>(.*?)</title>", html or "", re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
    if not canonical:
        canonical = url

    if body_text:
        words = body_text.split()
        if len(words) > 900:
            body_text = " ".join(words[:900])

    weights: Dict[str, float] = defaultdict(float)
    for tok in _tokenize(title): weights[tok] += 3.0
    for t in h1_texts:
        for tok in _tokenize(t): weights[tok] += 2.0
    for t in h2h3_texts:
        for tok in _tokenize(t): weights[tok] += 1.5
    for tok in _tokenize(body_text): weights[tok] += 1.0
    for tok in _path_tokens(canonical or url): weights[tok] += 2.2

    title_h1 = " ".join([title] + h1_texts).lower()
    is_post = _is_probably_post(canonical, title)

    return {
        "url": canonical or url,
        "title": title or "",
        "title_h1": title_h1,
        "weights": dict(weights),
        "is_post": is_post,
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
                tasks = [fetch(u) for u in urls[:_MAX_PAGES]]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, dict):
                        profiles.append(r)
            return profiles
        try:
            return asyncio.run(_run())
        except RuntimeError:
            pass

    if not requests:
        return profiles
    sess = requests.Session()
    try:
        for u in urls[:_MAX_PAGES]:
            try:
                r = sess.get(u, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT,_READ_TIMEOUT), allow_redirects=True)
                if r.status_code >= 400: continue
                ctype = r.headers.get("Content-Type","").lower()
                if "html" not in ctype and "text" not in ctype: continue
                html = r.content[:_MAX_BYTES].decode(r.apparent_encoding or "utf-8", errors="ignore")
                prof = _extract_profile(html, str(r.url))
                if prof and prof.get("weights"):
                    profiles.append(prof)
            except Exception:
                continue
    finally:
        sess.close()
    return profiles


# ---------- Build a site-aware thesaurus (lightweight) ----------
def _flag_careers_url(u: str, title_h1: str) -> bool:
    u2 = u.lower()
    if any(seg in u2 for seg in ("/careers","/jobs","/employment","/work-with-us","/join-our-team")):
        return True
    if any(w in title_h1 for w in CAREER_CUES):
        return True
    return False

def _flag_contact(u: str, title_h1: str) -> bool:
    u2 = u.lower()
    if any(seg in u2 for seg in ("/contact","/get-in-touch","/connect")):
        return True
    if any(w in title_h1 for w in CONTACT_CUES):
        return True
    return False

def _flag_donate(u: str, title_h1: str) -> bool:
    u2 = u.lower()
    if any(seg in u2 for seg in ("/donate","/give","/donation","/support")):
        return True
    if any(w in title_h1 for w in DONATE_CUES):
        return True
    return False

def _prob_program(u: str, title_h1: str) -> bool:
    u2 = u.lower()
    if any(seg in u2 for seg in ("/program","/programs","/services","/what-we-do","/our-services")):
        return True
    if any(w in title_h1.split() for w in PROGRAM_SEEDS):
        return True
    return False

def build_site_thesaurus(profiles: List[Dict]) -> Dict[str, Set[str]]:
    th: Dict[str, Set[str]] = {
        "careers": set(CAREER_CUES),
        "contact": set(CONTACT_CUES),
        "donate": set(DONATE_CUES),
        "about": set(ABOUT_CUES),
        "programs": set(),
    }
    for p in profiles:
        u = p["url"]
        title_h1 = p.get("title_h1","")
        toks = set(_tokenize(p.get("title","")) + _path_tokens(u))
        if _flag_careers_url(u, title_h1): th["careers"].update(toks)
        if _flag_contact(u, title_h1):      th["contact"].update(toks)
        if _flag_donate(u, title_h1):       th["donate"].update(toks)
        if _flag_about(u, title_h1):        th["about"].update(toks)
        if _prob_program(u, title_h1):      th["programs"].update(toks)
    for k in th:
        th[k] = {t for t in th[k] if len(t) >= 3}
    return th


# ---------- Caching wrappers ----------
@st.cache_data(show_spinner=False, ttl=3600)
def cached_discover_urls(base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> List[str]:
    return discover_urls(base_url, include_subdomains=include_subdomains, use_sitemap_first=use_sitemap_first)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_profiles(urls: Tuple[str, ...]) -> List[Dict]:
    return _fetch_profiles(list(urls))

@st.cache_data(show_spinner=False, ttl=3600)
def cached_site_thesaurus(urls: Tuple[str, ...]) -> Dict[str, Set[str]]:
    profs = cached_fetch_profiles(urls)
    return build_site_thesaurus(profs)


# ---------- Fit scoring (with IG-only concept bias) ----------
def _fit_score(keyword: str, profile: Dict, th: Dict[str, Set[str]],
               kw_is_careers: bool, kw_prog_hits: Set[str], enable_overrides: bool) -> float:
    tokens = [_normalize_token(t) for t in _tokenize(keyword)]
    if not tokens:
        return 0.0
    w = profile.get("weights", {})
    overlap = sum(w.get(t, 0.0) for t in tokens) / max(1, len(tokens))
    title_h1 = profile.get("title_h1", "")
    covered = sum(1 for t in tokens if t in title_h1)
    if covered == len(tokens):
        overlap += 0.25
    elif covered / len(tokens) >= 0.5:
        overlap += 0.10
    phrase = " ".join(tokens)
    if phrase and phrase in title_h1:
        overlap += 0.15

    # always prefer pages over posts a bit
    if not profile.get("is_post", False):
        overlap += 0.06

    # IG-only concept nudges
    if enable_overrides:
        url = profile["url"].lower()
        is_careers = _flag_careers_url(url, title_h1)
        is_program_like = _prob_program(url, title_h1)
        if kw_is_careers and is_careers:
            overlap += 0.08
        if kw_prog_hits and is_program_like:
            overlap += 0.06

    return max(0.0, min(2.2, overlap))


# ---------- Mapping algorithm ----------
def _keyword_flags(kw: str) -> Tuple[bool, Set[str]]:
    toks = set(_normalize_token(t) for t in _tokenize(kw))
    is_role = any(t in ROLE_TOKS for t in toks)
    has_career_word = any(t in CAREER_CUES for t in toks)
    kw_is_careers = is_role or has_career_word
    prog_hits = {t for t in toks if t in PROGRAM_SEEDS}
    return kw_is_careers, prog_hits

def map_keywords_to_urls(df: pd.DataFrame, kw_col: Optional[str], vol_col: str, kd_col: str,
                         base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> pd.Series:
    # Discover & profile (cached)
    url_list = cached_discover_urls(base_url, include_subdomains=include_subdomains, use_sitemap_first=use_sitemap_first)
    if not url_list:
        return pd.Series([""] * len(df), index=df.index, dtype="string")
    url_tuple = tuple(url_list)
    profiles = cached_fetch_profiles(url_tuple)
    profiles = [p for p in profiles if p.get("weights")]
    if not profiles:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    # Site thesaurus
    th = cached_site_thesaurus(url_tuple)

    prof_by_url = {p["url"]: p for p in profiles}
    prof_list = list(prof_by_url.values())

    vols = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).clip(lower=0)
    max_log = float((vols + 1).apply(lambda x: math.log(1 + x)).max()) or 1.0

    def strat_weights():
        if scoring_mode == "Low Hanging Fruit":
            return 0.20, 0.45, 0.35  # fit, kd_norm, vol
        elif scoring_mode == "In The Game":
            return 0.30, 0.35, 0.35
        else:
            return 0.30, 0.20, 0.50
    W_FIT, W_KD, W_VOL = strat_weights()

    TOP_K = 4
    kw_candidates: Dict[int, List[Tuple[str, float]]] = {}
    kw_slot: Dict[int, str] = {}
    kw_rank: Dict[int, float] = {}

    is_ig = scoring_mode == "In The Game"
    is_comp = scoring_mode == "Competitive"

    # Build candidates with IG-only overrides
    for idx, row in df.iterrows():
        # IG/Competitive: only map eligible rows
        if (is_ig or is_comp) and str(row.get("Eligible", "")) != "Yes":
            kw_candidates[idx] = []
            kw_slot[idx] = "SEO"  # placeholder
            kw_rank[idx] = 0.0
            continue

        kw = str(row.get(kw_col, "")) if kw_col else str(row.get("Keyword", ""))
        cats = set(categorize_keyword(kw))
        if "VEO" in cats:   slot = "VEO"
        elif "AIO" in cats: slot = "AIO"
        else:               slot = "SEO"

        kw_is_careers, prog_hits = _keyword_flags(kw)

        fits: List[Tuple[str, float]] = []
        for p in prof_list:
            f = _fit_score(
                kw, p, th,
                kw_is_careers=kw_is_careers,
                kw_prog_hits=prog_hits,
                enable_overrides=is_ig  # <-- IG ONLY
            )
            if f > 0:
                fits.append((p["url"], f))
        fits.sort(key=lambda x: x[1], reverse=True)
        kw_candidates[idx] = fits[:TOP_K]
        kw_slot[idx] = slot

        kd_val = float(pd.to_numeric(row.get(kd_col, 0), errors="coerce") or 0)
        vol_val = float(pd.to_numeric(row.get(vol_col, 0), errors="coerce") or 0)
        best_fit = fits[0][1] if fits else 0.0
        fit_norm = best_fit / 2.2
        kd_norm = max(0.0, 1.0 - kd_val / 100.0)
        vol_norm = math.log(1 + max(0.0, vol_val)) / max_log
        kw_rank[idx] = W_FIT * fit_norm + W_KD * kd_norm + W_VOL * vol_norm

    # Caps per URL (Primary SEO 2 + AIO 1 + VEO 1)
    caps = {"VEO": 1, "AIO": 1, "SEO": 2}
    assigned: Dict[str, Dict[str, List[int] | Optional[int]]] = {}
    for p in prof_list:
        assigned[p["url"]] = {"VEO": None, "AIO": None, "SEO": []}

    mapped = {i: "" for i in df.index}

    # Acceptance minima
    if is_ig:
        CAREERS_MIN = 0.22
        PROGRAM_MIN = 0.20
        GENERIC_MIN = _ALT_FIT_MIN
    else:
        # LHF & Competitive: no special minima; all generic
        CAREERS_MIN = PROGRAM_MIN = GENERIC_MIN = _ALT_FIT_MIN

    def assign_slot(slot_name: str):
        ids = [i for i, s in kw_slot.items() if s == slot_name]
        ids.sort(key=lambda i: (-kw_rank.get(i, 0.0), i))
        for i in ids:
            choices = kw_candidates.get(i, [])
            if not choices:
                continue
            kw_text = str(df.loc[i, kw_col]) if kw_col else str(df.loc[i, "Keyword"])
            kw_is_careers, prog_hits = _keyword_flags(kw_text)
            for j, (u, fit) in enumerate(choices):
                title_h1 = prof_by_url[u].get("title_h1","")
                is_careers_url = _flag_careers_url(u, title_h1)
                is_program_like = _prob_program(u, title_h1)

                # choose threshold (IG-only special; others generic)
                if is_ig and kw_is_careers and is_careers_url:
                    min_fit = CAREERS_MIN
                elif is_ig and prog_hits and is_program_like:
                    min_fit = PROGRAM_MIN
                else:
                    min_fit = GENERIC_MIN

                if slot_name in {"VEO","AIO"}:
                    if assigned[u][slot_name] is None and (j == 0 or fit >= min_fit):
                        assigned[u][slot_name] = i
                        mapped[i] = u
                        break
                else:  # SEO
                    if len(assigned[u]["SEO"]) < caps["SEO"] and (j == 0 or fit >= min_fit):
                        assigned[u]["SEO"].append(i)
                        mapped[i] = u
                        break

    assign_slot("VEO")
    assign_slot("AIO")
    assign_slot("SEO")

    # Enforce SEO cap hard
    for u, slots in assigned.items():
        if isinstance(slots["SEO"], list) and len(slots["SEO"]) > caps["SEO"]:
            for drop_idx in slots["SEO"][caps["SEO"]:]:
                mapped[drop_idx] = ""
            slots["SEO"] = slots["SEO"][:caps["SEO"]]

    return pd.Series([mapped[i] for i in df.index], index=df.index, dtype="string")


# ---------- Single Keyword ----------
st.subheader("Single Keyword Score")
with st.form("single"):
    c1, c2 = st.columns(2)
    with c1:
        vol_val = st.number_input("Search Volume (A)", min_value=0, step=10, value=0)
    with c2:
        kd_val  = st.number_input("Keyword Difficulty (B)", min_value=0, step=1, value=0)

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

# Minimal Mapping controls
base_site_url = st.text_input("Base site URL (for URL mapping)", placeholder="https://example.com")
use_sitemap_first = True
include_subdomains = True  # always include; UI toggle removed as requested

uploaded = st.file_uploader("Upload CSV", type=["csv"])
example = pd.DataFrame({"Keyword":["best running shoes","seo tools","crm software"], "Volume":[5400,880,12000], "KD":[38,72,18]})
with st.expander("See example CSV format"):
    st.dataframe(example, use_container_width=True)

# ---------- Robust CSV reader + numeric cleaning ----------
map_clicked = False
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
            {"encoding": None, "sep": "\\t", "engine": "python"},
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
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\\s]","",regex=True).str.replace("%","",regex=False)
        df[kd_col]  = df[kd_col].astype(str).str.replace(r"[,\\s]","",regex=True).str.replace("%","",regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col]  = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

        scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

        # ---------- CSV base ----------
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        # Sort for readability
        export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes":1,"No":0}).fillna(0)
        export_df = export_df.sort_values(
            by=["_EligibleSort", kd_col, vol_col],
            ascending=[False, True, False],
            kind="mergesort"
        ).drop(columns=["_EligibleSort"])
        export_cols = base_cols + ["Strategy"]
        export_df = export_df[export_cols]

        # Map button
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        map_clicked = st.button("🚀 Map Keywords to Site", key="map_btn", use_container_width=False, help="Uses page-sitemap first, then other sitemaps/crawl.")

        # -------- URL Mapping (appends last column WHEN button clicked) --------
        map_series = pd.Series([""] * len(export_df), index=export_df.index, dtype="string")
        if map_clicked and base_site_url.strip():
            sig_cols = [c for c in [kw_col, vol_col, kd_col] if c]
            try:
                sig_df = export_df[sig_cols].copy()
            except Exception:
                sig_df = export_df[[col for col in sig_cols if col in export_df.columns]].copy()
            sig_csv = sig_df.fillna("").astype(str).to_csv(index=False)
            sig_base = f"{_normalize_base(base_site_url.strip()).lower()}|{scoring_mode}|{kw_col}|{vol_col}|{kd_col}|{len(export_df)}"
            signature = hashlib.md5((sig_base + "\\n" + sig_csv).encode("utf-8")).hexdigest()

            # Strategy-specific cache key so each strategy reserves URL capacity independently
            cache_key = (signature, scoring_mode)

            if "map_cache" not in st.session_state:
                st.session_state["map_cache"] = {}
            cache = st.session_state["map_cache"]

            if cache_key in cache and len(cache[cache_key]) == len(export_df):
                map_series = pd.Series(cache[cache_key], index=export_df.index, dtype="string")
            else:
                loader = st.empty()
                loader.markdown(
                    """
                    <div class="loader-row"><div class="loader-circle"></div><div>Mapping keywords to your site…</div></div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.spinner("Launching fast crawl & scoring fit…"):
                    map_series = map_keywords_to_urls(
                        export_df,
                        kw_col=kw_col,
                        vol_col=vol_col,
                        kd_col=kd_col,
                        base_url=base_site_url.strip(),
                        include_subdomains=include_subdomains,
                        use_sitemap_first=use_sitemap_first,
                    )
                loader.empty()
                cache[cache_key] = list(map_series.fillna("").astype(str).values)

        # Add column (always present; possibly blank if not clicked)
        export_df["Map URL"] = map_series

        # Export
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
