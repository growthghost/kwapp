import io
import re
import asyncio
import contextlib
import gzip
import math
import hashlib
from typing import Optional, List, Dict, Tuple, Set
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
except Exception:
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # noqa: F401
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

# ---------- Brand / Theme ----------
BRAND_BG = "#747474"     # background
BRAND_INK = "#242F40"    # blue/ink
BRAND_ACCENT = "#329662" # accent (green)
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
  --accent-rgb: 50,150,98;
  --light: {BRAND_LIGHT};
}}
/* App background + base text */
.stApp {{ background-color: var(--bg); }}
html, body, [class^="css"], [class*=" css"] {{ color: var(--light) !important; }}
h1, h2, h3, h4, h5, h6 {{ color: var(--light) !important; }}

/* Inputs: white surface, GREEN focus (no red outlines) */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--light) !important;
  color: var(--ink) !important;
  border-radius: 8px !important;
}}
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {{ border: 2px solid var(--light) !important; }}
.stNumberInput input:focus,
.stNumberInput input:focus-visible,
.stNumberInput:focus-within input,
.stTextInput input:focus,
.stTextInput input:focus-visible,
.stSelectbox div[data-baseweb="select"]:focus-within > div {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
  outline: none !important;
}}
.stNumberInput button {{
  background: var(--ink) !important; color:#fff !important; border:1px solid var(--ink) !important;
}}
.stNumberInput button:hover,.stNumberInput button:active,.stNumberInput button:focus-visible {{
  background: var(--accent) !important; color:#000 !important; border-color: var(--accent) !important;
}}
/* Select caret */
.stSelectbox div[data-baseweb="select"] > div {{ position: relative; }}
.stSelectbox div[data-baseweb="select"] > div::after {{
  content:"▾"; position:absolute; right:12px; top:50%; transform:translateY(-50%); color:var(--ink); pointer-events:none; font-size:14px; font-weight:700;
}}

/* Uploader area + Browse button (BLUE -> WHITE on hover) */
[data-testid="stFileUploaderDropzone"] {{ background: rgba(255,255,255,0.98); border: 2px dashed var(--accent); }}
[data-testid="stFileUploader"] * {{ color: var(--ink) !important; }}
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
  background-color: var(--light) !important; /* white on hover */
  color: var(--ink) !important;             /* blue text */
  border-color: var(--ink) !important;      /* blue border */
}}

/* Tables */
.stDataFrame, .stDataFrame *, .stTable, .stTable * {{ color: var(--ink) !important; }}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
  background-color: var(--accent) !important; color: var(--ink) !important;
  border: 2px solid var(--light) !important; border-radius: 10px !important; font-weight: 700 !important;
  box-shadow: 0 2px 0 rgba(0,0,0,.15);
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background-color: transparent !important; color: var(--light) !important; border-color: var(--accent) !important;
}}
/* Banner */
.info-banner {{ background: linear-gradient(90deg, var(--ink) 0%, var(--accent) 100%); padding:16px; border-radius:12px; color:var(--light); }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Title ----------
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

LABEL_MAP = {6:"Elite",5:"Excellent",4:"Good",3:"Fair",2:"Low",1:"Very Low",0:"Not rated"}
COLOR_MAP = {6:"#2ecc71",5:"#a3e635",4:"#facc15",3:"#fb923c",2:"#f87171",1:"#ef4444",0:"#9ca3af"}
strategy_descriptions = {
    "Low Hanging Fruit":"Keywords that can be used to rank quickly with minimal effort. Ideal for new content or low-authority sites. Try targeting long-tail keywords, create quick-win content, and build a few internal links.",
    "In The Game":"Moderate difficulty keywords that are within reach for growing sites. Focus on optimizing content, earning backlinks, and matching search intent to climb the ranks.",
    "Competitive":"High-volume, high-difficulty keywords dominated by authoritative domains. Requires strong content, domain authority, and strategic SEO to compete. Great for long-term growth.",
}

# ---------- Strategy ----------
scoring_mode = st.selectbox("Choose Scoring Strategy", ["Low Hanging Fruit","In The Game","Competitive"])
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
<div class="info-banner" style="margin-bottom:16px;">
  <div style='margin-bottom:6px; font-size:13px;'>Minimum Search Volume Required: <strong>{MIN_VALID_VOLUME}</strong></div>
  <strong style='font-size:18px;'>{scoring_mode}</strong><br>
  <span style='font-size:15px;'>{strategy_descriptions[scoring_mode]}</span>
</div>
""", unsafe_allow_html=True)

# ---------- Category Tagging ----------
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|step[- ]?by[- ]?step|tutorial|guide|is)\b", re.I)
AEO_PAT = re.compile(r"^\s*(who|what|when|where|why|how|which|can|should)\b", re.I)
VEO_PAT = re.compile(r"\b(near me|open now|closest|call now|directions|ok google|alexa|siri|hey google)\b", re.I)
GEO_PAT = re.compile(r"\b(how to|best way to|steps? to|examples? of|checklist|framework|template)\b", re.I)
SXO_PAT = re.compile(r"\b(best|top|compare|comparison|vs\.?|review|pricing|cost|cheap|free download|template|examples?)\b", re.I)
LLM_PAT = re.compile(r"\b(prompt|prompting|prompt[- ]?engineering|chatgpt|gpt[- ]?\d|llm|rag|embedding|vector|few[- ]?shot|zero[- ]?shot)\b", re.I)
AIO_PAGE_SIG = re.compile(r"\b(what is|how to|guide|tutorial|step[- ]?by[- ]?step|checklist|framework|template|examples?)\b", re.I)
CATEGORY_ORDER = ["SEO","AIO","VEO","GEO","AEO","SXO","LLM"]

def categorize_keyword(kw: str) -> List[str]:
    if not isinstance(kw,str) or not kw.strip():
        return ["SEO"]
    text = kw.strip().lower()
    cats = set()
    if AIO_PAT.search(text): cats.add("AIO")
    if AEO_PAT.search(text): cats.add("AEO")
    if VEO_PAT.search(text): cats.add("VEO")
    if GEO_PAT.search(text): cats.add("GEO")
    if SXO_PAT.search(text): cats.add("SXO")
    if LLM_PAT.search(text): cats.add("LLM")
    if not cats: cats.add("SEO")
    else:
        if "LLM" not in cats: cats.add("SEO")
    return [c for c in CATEGORY_ORDER if c in cats]

# ---------- Score ----------
def calculate_score(volume: float, kd: float) -> int:
    if pd.isna(volume) or pd.isna(kd): return 0
    if volume < MIN_VALID_VOLUME: return 0
    kd = max(0.0, min(100.0, float(kd)))
    for low, high, score in KD_BUCKETS:
        if low <= kd <= high: return score
    return 0

def add_scoring_columns(df: pd.DataFrame, volume_col: str, kd_col: str, kw_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    def _eligibility_reason(vol,kd):
        if pd.isna(vol) or pd.isna(kd): return "No","Invalid Volume/KD"
        if vol < MIN_VALID_VOLUME: return "No", f"Below min volume for {scoring_mode} ({MIN_VALID_VOLUME})"
        return "Yes",""
    eligible, reason = zip(*(_eligibility_reason(v,k) for v,k in zip(out[volume_col], out[kd_col])))
    out["Eligible"] = list(eligible); out["Reason"] = list(reason)
    out["Score"] = [calculate_score(v,k) for v,k in zip(out[volume_col], out[kd_col])]
    out["Tier"]  = out["Score"].map(LABEL_MAP).fillna("Not rated")
    kw_series = out[kw_col] if kw_col else pd.Series([""]*len(out), index=out.index)
    out["Category"] = [", ".join(categorize_keyword(str(k))) for k in kw_series]
    ordered = ([kw_col] if kw_col else []) + [volume_col, kd_col, "Score","Tier","Eligible","Reason","Category"]
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]

# ---------- Tokenization & constants ----------
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_MAX_PAGES = 120
_MAX_BYTES = 350_000
_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 8
_TOTAL_BUDGET_SECS = 60
_CONCURRENCY = 16
_THREADS = 12

# per-class minimums (page vs post) — RELAXED
_MIN_SEO_PAGE = 0.20   # was 0.25
_MIN_SEO_POST = 0.50   # was 0.55
_MIN_AIO_PAGE = 0.16   # was 0.18
_MIN_AIO_POST = 0.22   # unchanged
_MIN_VEO_PAGE = 0.16   # was 0.18
_MIN_VEO_POST = 0.28   # was 0.30

_ALT_FIT_MIN = 0.20    # was 0.22

_DEF_HEADERS = {"User-Agent": "OutrankIQMapper/1.3 (+https://example.com)"}
_MAPPER_VERSION = "site-map-v12-pages-first-soften-1"

# ---------- Synonyms / phrase normalization ----------
PHRASE_MAP = [
    (re.compile(r"\bget[ -]?in[ -]?touch\b", re.I), "contact"),
    (re.compile(r"\breach[ -]?out\b", re.I), "contact"),
    (re.compile(r"\bemail\s*us\b", re.I), "contact"),
    (re.compile(r"\bcall\s*us\b", re.I), "contact"),
    (re.compile(r"\bnon[- ]?profit(s)?\b", re.I), "nonprofit"),
    (re.compile(r"\bnear\s*me\b", re.I), "nearme"),
    (re.compile(r"\bnear\s*you\b", re.I), "nearyou"),
]

_SYN_MAP = {
    "connect":"contact","connected":"contact","connecting":"contact","contacts":"contact",
    "support":"contact","helpdesk":"contact","help-line":"contact","helpline":"contact",
    "nearby":"nearme","nearyou":"nearme","directions":"nearme","address":"nearme",
    "offices":"locations","office":"locations","locations":"locations","visit":"locations","findus":"locations",
    "organisations":"organization","organisation":"organization","organizations":"organization","orgs":"organization","org":"organization",
    "nonprofit":"organization","ngo":"organization","charity":"organization","foundation":"organization",
    "assist":"help","assists":"help","assistance":"help","caregiving":"help","caregiver":"help","caregivers":"help",
    "disabilities":"disability","disabled":"disability",
    "programs":"program","programme":"program","services":"service",
    "contribute":"donate","give":"donate","giving":"donate",
    "involved":"involved","volunteering":"volunteer","volunteers":"volunteer",
}

STOPWORDS = {
    "the","and","for","to","a","an","of","with","in","on","at","by","from","about",
    "is","are","be","can","should","how","what","who","where","why","when","which",
    "me","you","us","my","our","your","we","i","it","they","them","near","now","open",
    "vs","or","and","as"
}

def _normalize_phrases(text: str) -> str:
    if not text: return ""
    out = text
    for pat, rep in PHRASE_MAP:
        out = pat.sub(rep, out)
    return out

def _norm_token(t: str) -> str:
    t = t.lower()
    if t in _SYN_MAP: t = _SYN_MAP[t]
    if t.endswith("ies") and len(t) > 3: t = t[:-3] + "y"
    elif t.endswith("es") and len(t) > 4: t = t[:-2]
    elif t.endswith("s") and len(t) > 3: t = t[:-1]
    if t.endswith("ing") and len(t) > 5: t = t[:-3]
    elif t.endswith("ed") and len(t) > 4: t = t[:-2]
    return t

def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower()) if text else []

def _ntokens(text: str) -> List[str]:
    text = _normalize_phrases(text or "")
    return [_norm_token(t) for t in _tokenize(text)]

def _head_noun(tokens: List[str]) -> str:
    for t in reversed(tokens):
        if t and t not in STOPWORDS and t.isalpha() and len(t) >= 3:
            return t
    return ""

# ---------- Domain helpers ----------
def _derive_roots(base_url: str) -> Tuple[str,str]:
    host = urlparse(base_url).netloc.lower()
    parts = host.split(".")
    base_root = ".".join(parts[-2:]) if len(parts)>=2 else host
    return host, base_root

def _domain_tokens(base_url: str) -> set:
    host = urlparse(_normalize_base(base_url)).netloc.lower()
    parts = re.split(r"[.\-]", host)
    return { _norm_token(p) for p in parts if p and p.isalpha() and len(p) >= 2 }

def _normalize_base(base_url: str) -> str:
    if not base_url: return ""
    u = base_url.strip()
    if not u.startswith(("http://","https://")):
        u = "https://" + u
    p = urlparse(u)
    return f"{p.scheme}://{p.netloc}"

def _same_site(url: str, base_host: str, base_root: str, include_subdomains: bool) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if not host: return False
        apex = base_root; www_host = f"www.{base_root}"
        if include_subdomains:
            if host in {base_host, apex, www_host}: return True
            return host.endswith("."+base_root)
        else:
            return host in {base_host, apex, www_host}
    except Exception:
        return False

# ---------- URL normalization & page-like helpers ----------
def _url_key(u: str) -> str:
    try:
        p = urlparse(u)
        scheme = 'https' if p.scheme in ('', 'http', 'https') else p.scheme
        host = p.netloc.lower()
        path = re.sub(r'/+', '/', p.path or '/')
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        return f"{scheme}://{host}{path}"
    except Exception:
        return (u or "").strip().lower()

def _is_home(u: str) -> bool:
    p = urlparse(u)
    path = re.sub(r'/+', '/', p.path or '/')
    return path == '/'

def _is_contact_like(u: str) -> bool:
    path = urlparse(u).path.lower()
    return any(seg in path for seg in ("/contact", "/contact-us", "/locations", "/find-us", "/visit"))

def _is_post_like(stype: str, url: str) -> bool:
    path = urlparse(url).path.lower()
    return (stype == "post") or ("/blog/" in path) or ("/news/" in path) or bool(re.search(r"/\d{4}/\d{2}/", path))

def _is_page_like(source_type: str, url: str, is_nav: bool) -> bool:
    if source_type == 'page': return True
    path = urlparse(url).path.lower()
    if _is_home(url): return True
    if is_nav and not any(seg in path for seg in ('/blog/', '/news/', '/category/', '/tag/', '/author/', '/archive/')):
        depth = len([seg for seg in path.split('/') if seg])
        if depth <= 2: return True
    depth = len([seg for seg in path.split('/') if seg])
    if depth <= 1 and not re.search(r"/\d{4}/\d{2}/", path) and not any(s in path for s in ('/blog/', '/news/')):
        return True
    return False

def _slug_tokens(u: str) -> Set[str]:
    path = urlparse(u).path.lower()
    segs = [s for s in path.split('/') if s]
    toks = set()
    for s in segs:
        for t in re.split(r"[-_]+", s):
            tt = _norm_token(t)
            if tt and tt not in STOPWORDS:
                toks.add(tt)
    return toks

# ---------- Robots & sitemaps ----------
@contextlib.contextmanager
def _session():
    if HAVE_AIOHTTP:
        yield None
    else:
        yield requests.Session() if requests else None

def _fetch_text_requests(url: str, session, timeout: Tuple[int,int]) -> Optional[str]:
    try:
        if session is not None:
            resp = session.get(url, headers=_DEF_HEADERS, timeout=timeout, stream=True, allow_redirects=True)
        elif requests is not None:
            resp = requests.get(url, headers=_DEF_HEADERS, timeout=timeout, stream=True, allow_redirects=True)
        else:
            return None
        if resp.status_code >= 400: return None
        ctype = resp.headers.get("Content-Type","").lower()
        raw = resp.content[:_MAX_BYTES]
        if ("gzip" in ctype) or url.lower().endswith(".gz"):
            with contextlib.suppress(Exception):
                raw = gzip.decompress(raw)
            ctype = "text/xml"
        if ("text" not in ctype) and ("html" not in ctype) and ("xml" not in ctype): return None
        return raw.decode(resp.apparent_encoding or "utf-8", errors="ignore")
    except Exception:
        return None

def _extract_sitemaps_from_robots(base_root_url: str, session) -> List[str]:
    robots_url = urljoin(base_root_url + "/", "robots.txt")
    txt = _fetch_text_requests(robots_url, session, (_CONNECT_TIMEOUT,_READ_TIMEOUT))
    if not txt: return []
    maps = []
    for line in txt.splitlines():
        if line.lower().startswith("sitemap:"):
            sm = line.split(":",1)[1].strip()
            if sm: maps.append(sm)
    return maps

def _parse_sitemap_xml_entries(xml_text: str) -> List[Tuple[str, Optional[float], Optional[str]]]:
    if not xml_text: return []
    entries: List[Tuple[str, Optional[float], Optional[str]]] = []
    for m in re.finditer(r"<url>\s*(.*?)\s*</url>", xml_text, re.I | re.S):
        block = m.group(1)
        lm = re.search(r"<loc>\s*([^<]+)\s*</loc>", block, re.I)
        if not lm: continue
        loc = lm.group(1).strip()
        pr = None; lastmod = None
        pm = re.search(r"<priority>\s*([0-9.]+)\s*</priority>", block, re.I)
        if pm:
            with contextlib.suppress(Exception):
                pr = float(pm.group(1))
        lmm = re.search(r"<lastmod>\s*([^<]+)\s*</lastmod>", block, re.I)
        if lmm:
            lastmod = lmm.group(1).strip()
        entries.append((loc, pr, lastmod))
    for m in re.finditer(r"<sitemap>\s*(.*?)\s*</sitemap>", xml_text, re.I | re.S):
        block = m.group(1)
        lm = re.search(r"<loc>\s*([^<]+)\s*</loc>", block, re.I)
        if lm:
            entries.append((lm.group(1).strip(), None, None))
    if not entries:
        for lm in re.finditer(r"<loc>\s*([^<]+)\s*</loc>", xml_text, re.I):
            entries.append((lm.group(1).strip(), None, None))
    return entries

def _sm_classify(u: str) -> str:
    ul = u.lower()
    if "page" in ul: return "page"
    if ("post" in ul) or ("blog" in ul) or ("news" in ul): return "post"
    if ("category" in ul) or ("tag" in ul) or ("author" in ul) or ("archive" in ul): return "tax"
    return "other"

def _sm_bucket(u: str) -> int:
    t = _sm_classify(u)
    return {"page":0,"post":1,"tax":2,"other":3}.get(t,3)

def _collect_sitemap_urls(sm_url: str, session, base_host: str, base_root: str,
                          include_subdomains: bool, seen: set, out: List[str],
                          srcmap: Dict[str,str], meta: Dict[str, Dict[str, Optional[str]]],
                          parent_type: Optional[str] = None, depth: int = 0):
    if depth > 3 or len(out) >= _MAX_PAGES or sm_url in seen: return
    seen.add(sm_url)
    xml = _fetch_text_requests(sm_url, session, (_CONNECT_TIMEOUT,_READ_TIMEOUT))
    if not xml: return
    entries = _parse_sitemap_xml_entries(xml)
    is_index = bool(re.search(r"<sitemapindex", xml, re.I)) or any(u.lower().endswith((".xml",".xml.gz")) for (u,_,_) in entries)

    if is_index:
        children = [u for (u,_,_) in entries if u]
        children.sort(key=_sm_bucket)  # prefer page sitemaps first
        for child in children:
            if len(out) >= _MAX_PAGES: break
            if child.lower().endswith((".xml",".xml.gz")):
                _collect_sitemap_urls(child, session, base_host, base_root, include_subdomains,
                                      seen, out, srcmap, meta, parent_type=None, depth=depth+1)
        return

    stype = parent_type if parent_type else _sm_classify(sm_url)
    for (u, pr, lm) in entries:
        if len(out) >= _MAX_PAGES: break
        if _same_site(u, base_host, base_root, include_subdomains):
            out.append(u)
            key = _url_key(u)
            srcmap.setdefault(key, stype)
            md = meta.setdefault(key, {})
            if pr is not None:
                md["priority"] = f"{pr:.3f}"
            if lm:
                md["lastmod"] = lm

def discover_urls_with_sources(base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> Tuple[List[str], Dict[str,str], Dict[str, Dict[str, Optional[str]]]]:
    base = _normalize_base(base_url)
    if not base: return [], {}, {}
    base_host, base_root = _derive_roots(base)
    discovered: List[str] = []
    srcmap: Dict[str,str] = {}
    meta: Dict[str, Dict[str, Optional[str]]] = {}

    with _session() as sess:
        if use_sitemap_first:
            maps = _extract_sitemaps_from_robots(base, sess)
            if not maps:
                maps = [urljoin(base + "/", "sitemap.xml"), urljoin(base + "/", "sitemap_index.xml")]
            maps.sort(key=_sm_bucket)
            seen = set()
            for sm in maps:
                if len(discovered) >= _MAX_PAGES: break
                _collect_sitemap_urls(sm, sess, base_host, base_root, include_subdomains,
                                      seen, discovered, srcmap, meta, parent_type=_sm_classify(sm), depth=0)

        if not discovered:
            discovered = shallow_crawl(base, include_subdomains)
            for u in discovered:
                srcmap.setdefault(_url_key(u), "other")

    discovered = list(dict.fromkeys(discovered))
    def _path_depth(u: str) -> int:
        return len([seg for seg in urlparse(u).path.split("/") if seg])
    prio = {"page":0,"post":1,"tax":2,"other":3}
    discovered.sort(key=lambda u: (prio.get(srcmap.get(_url_key(u),"other"),3), _path_depth(u)))
    trimmed = discovered[:_MAX_PAGES]
    trimmed_meta = {}
    for u in trimmed:
        key = _url_key(u)
        trimmed_meta[u] = meta.get(key, {})
    return trimmed, {u: srcmap.get(_url_key(u),"other") for u in trimmed}, trimmed_meta

# ---------- Shallow crawl ----------
def _extract_links(html: str, current_url: str) -> List[str]:
    links: List[str] = []
    if not html: return links
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
        if href: links.append(urljoin(current_url, href))
    return links

def shallow_crawl(base_url: str, include_subdomains: bool) -> List[str]:
    base = _normalize_base(base_url)
    base_host, base_root = _derive_roots(base)
    start = base
    frontier = [(start,0)]
    seen = set([start])
    out: List[str] = []

    def ok(u: str) -> bool:
        if not _same_site(u, base_host, base_root, include_subdomains): return False
        p = urlparse(u)
        if not p.scheme.startswith("http"): return False
        if any(u.lower().endswith(ext) for ext in (".pdf",".jpg",".jpeg",".png",".gif",".webp",".svg",".zip",".rar")): return False
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
                            if resp.status >= 400: continue
                            ctype = resp.headers.get("Content-Type","").lower()
                            if "html" not in ctype and "text" not in ctype: continue
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
            if not requests: return [base]
            sess = requests.Session()
            try:
                while frontier and len(out) < _MAX_PAGES:
                    url, depth = frontier.pop(0)
                    try:
                        r = sess.get(url, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT,_READ_TIMEOUT), allow_redirects=True)
                        if r.status_code >= 400: continue
                        ctype = r.headers.get("Content-Type","").lower()
                        if "html" not in ctype and "text" not in ctype: continue
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
    else:
        if not requests: return [base]
        sess = requests.Session()
        try:
            while frontier and len(out) < _MAX_PAGES:
                url, depth = frontier.pop(0)
                try:
                    r = sess.get(url, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT,_READ_TIMEOUT), allow_redirects=True)
                    if r.status_code >= 400: continue
                    ctype = r.headers.get("Content-Type","").lower()
                    if "html" not in ctype and "text" not in ctype: continue
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

# ---------- Nav harvesting ----------
def _harvest_nav(base_url: str) -> Tuple[Dict[str, Set[str]], Set[str]]:
    home = _normalize_base(base_url)
    html = ""
    with _session() as sess:
        html = _fetch_text_requests(home, sess, (_CONNECT_TIMEOUT,_READ_TIMEOUT)) or ""
    url_to_tokens: Dict[str, Set[str]] = defaultdict(set)
    all_tokens: Set[str] = set()
    if not html:
        return {}, set()
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            areas = []
            areas.extend(soup.find_all("nav"))
            header = soup.find("header"); footer = soup.find("footer")
            if header: areas.append(header)
            if footer: areas.append(footer)
            base = home
            for area in areas:
                for a in area.find_all("a", href=True):
                    href = urljoin(base, a["href"])
                    key = _url_key(href)
                    text = a.get_text(" ", strip=True)
                    toks = set(_ntokens(text))
                    if toks:
                        url_to_tokens[key].update(toks)
                        all_tokens.update(toks)
            t = soup.find("title")
            if t:
                all_tokens.update(_ntokens(t.get_text(" ", strip=True)))
        except Exception:
            pass
    else:
        base = home
        for m in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.I | re.S):
            href = urljoin(base, (m.group(1) or ""))
            key = _url_key(href)
            text = re.sub(r"<.*?>", "", m.group(2) or "")
            toks = set(_ntokens(text))
            if toks:
                url_to_tokens[key].update(toks)
                all_tokens.update(toks)
    return dict(url_to_tokens), set(all_tokens)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_nav(base_url: str) -> Tuple[Dict[str, Set[str]], Set[str]]:
    return _harvest_nav(base_url)

# ---------- Content profiling ----------
_PHONE_PAT = re.compile(r"(\+?\d{1,2}[\s.-]?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})")
_ADDR_PAT = re.compile(r"\b(suite|ste\.?|unit|rd\.?|road|st\.?|street|ave\.?|avenue|blvd\.?|boulevard|ln\.?|lane|dr\.?|drive|ct\.?|court)\b", re.I)

def _answerability_score(html: str, title_h1_text: str) -> float:
    score = 0.0
    t = title_h1_text.lower()
    if AIO_PAGE_SIG.search(t):
        score += 0.2
    if re.search(r"<(ol|ul)[\s>].*?<li", html or "", re.I | re.S):
        score += 0.1
    if re.search(r"\bFAQ\b|\bQ:\b|\bA:\b", html or "", re.I):
        score += 0.1
    first = re.sub(r"<.*?>", " ", (html or ""))
    first = re.sub(r"\s+", " ", first).strip()
    head = " ".join(first.split()[:200]).lower()
    if re.search(r"\b(is|are)\b.+?\.", head):
        score += 0.05
    if re.search(r"FAQPage|HowTo|QAPage", html or "", re.I):
        score += 0.1
    return min(0.5, score)

def _extract_profile(html: str, final_url: str, requested_url: Optional[str] = None) -> Dict:
    title = ""; h1_texts: List[str] = []; h2h3_texts: List[str] = []; body_text = ""; canonical = ""

    if HAVE_BS4 and html:
        try:
            soup = BeautifulSoup(html, "html.parser")
            t = soup.find("title"); title = t.get_text(" ", strip=True) if t else ""
            link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v,list) else [v]))
            if link and link.get("href"): canonical = urljoin(final_url, link["href"])
            for h in soup.find_all(["h1","h2","h3"]):
                txt = h.get_text(" ", strip=True)
                if not txt: continue
                if h.name == "h1": h1_texts.append(txt)
                else: h2h3_texts.append(txt)
            for tag in soup(["script","style","noscript","template","nav","footer","header","aside"]): tag.extract()
            body_text = soup.get_text(" ", strip=True)
        except Exception:
            pass
    if not title:
        m = re.search(r"<title>(.*?)</title>", html or "", re.I|re.S)
        if m: title = re.sub(r"\s+"," ",m.group(1)).strip()
    if not canonical: canonical = final_url
    if body_text:
        words = body_text.split()
        lead_text = " ".join(words[:200])
        if len(words) > 800: body_text = " ".join(words[:800])
    else:
        lead_text = ""

    weights: Dict[str,float] = defaultdict(float)
    for tok in _ntokens(title): weights[tok] += 3.0
    for t in h1_texts:
        for tok in _ntokens(t): weights[tok] += 2.5
    for t in h2h3_texts:
        for tok in _ntokens(t): weights[tok] += 1.5
    for tok in _ntokens(body_text): weights[tok] += 1.0

    title_h1 = " ".join([title] + h1_texts)
    title_h1_norm = " ".join(_ntokens(title_h1))
    lead_norm = " ".join(_ntokens(lead_text))

    veo_ready = False
    if _is_contact_like(canonical): veo_ready = True
    if _PHONE_PAT.search(html or "") or _ADDR_PAT.search(html or ""): veo_ready = True
    if re.search(r"LocalBusiness|Organization", html or "", re.I): veo_ready = True

    a_score = _answerability_score(html or "", title_h1)

    return {
        "url": canonical or final_url,
        "title": title or "",
        "title_h1": title_h1.lower(),
        "title_h1_norm": title_h1_norm,
        "lead_norm": lead_norm,
        "weights": dict(weights),
        "final_url": final_url,
        "requested_url": requested_url or final_url,
        "veo_ready": veo_ready,
        "a_score": a_score,
    }

def _fetch_profiles(urls: List[str]) -> List[Dict]:
    profiles: List[Dict] = []
    if not urls: return profiles

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
                                if resp.status >= 400: return None
                                ctype = resp.headers.get("Content-Type","").lower()
                                if "html" not in ctype and "text" not in ctype: return None
                                b = await resp.content.read(_MAX_BYTES)
                                html = b.decode(errors="ignore")
                                return _extract_profile(html, str(resp.url), requested_url=u)
                        except Exception:
                            return None
                results = await asyncio.gather(*[fetch(u) for u in urls], return_exceptions=True)
                for r in results:
                    if isinstance(r, dict): profiles.append(r)
            return profiles
        try:
            return asyncio.run(_run())
        except RuntimeError:
            if not requests: return profiles
            sess = requests.Session()
            try:
                for u in urls[:_MAX_PAGES]:
                    try:
                        r = sess.get(u, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT,_READ_TIMEOUT), allow_redirects=True)
                        if r.status_code >= 400: continue
                        ctype = r.headers.get("Content-Type","").lower()
                        if "html" not in ctype and "text" not in ctype: continue
                        html = r.content[:_MAX_BYTES].decode(r.apparent_encoding or "utf-8", errors="ignore")
                        prof = _extract_profile(html, str(r.url), requested_url=u)
                        if prof and prof.get("weights"): profiles.append(prof)
                    except Exception:
                        continue
            finally:
                sess.close()
            return profiles
    else:
        if not requests: return profiles
        sess = requests.Session()
        try:
            for u in urls[:_MAX_PAGES]:
                try:
                    r = sess.get(u, headers=_DEF_HEADERS, timeout=(_CONNECT_TIMEOUT,_READ_TIMEOUT), allow_redirects=True)
                    if r.status_code >= 400: continue
                    ctype = r.headers.get("Content-Type","").lower()
                    if "html" not in ctype and "text" not in ctype: continue
                    html = r.content[:_MAX_BYTES].decode(r.apparent_encoding or "utf-8", errors="ignore")
                    profiles.append(_extract_profile(html, str(r.url), requested_url=u))
                except Exception:
                    continue
        finally:
            sess.close()
        return profiles

# ---------- Caching ----------
@st.cache_data(show_spinner=False, ttl=3600)
def cached_discover_and_sources(base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> Tuple[List[str], Dict[str,str], Dict[str, Dict[str, Optional[str]]]]:
    return discover_urls_with_sources(base_url, include_subdomains, use_sitemap_first)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_profiles(urls: Tuple[str, ...]) -> List[Dict]:
    return _fetch_profiles(list(urls))

# ---------- Fit & priority ----------
def _fit_score(keyword: str, profile: Dict) -> float:
    tokens = _ntokens(keyword)
    if not tokens: return 0.0
    w = profile.get("weights", {})
    overlap = 0.0
    for t in tokens:
        tf = w.get(t, 0.0)
        if tf > 0:
            overlap += math.sqrt(tf)
    overlap /= max(1, len(tokens))
    title_tokens = set((profile.get("title_h1_norm") or "").split())
    covered = sum(1 for t in tokens if t in title_tokens)
    if covered == len(tokens): overlap += 0.25
    elif covered/len(tokens) >= 0.5: overlap += 0.12
    phrase = " ".join(tokens)
    thn = profile.get("title_h1_norm","")
    if phrase and phrase in thn: overlap += 0.22
    if covered == 0 and not _is_home(profile.get("url","")) and not _is_contact_like(profile.get("url","")):
        overlap -= 0.15
    return max(0.0, min(2.0, overlap))

def _url_priority_bonus(u: str, is_nav: bool, source_type: Optional[str]) -> float:
    """Page-first bias (kept from previous guardrails)."""
    path = urlparse(u).path.lower()
    depth = len([seg for seg in path.split("/") if seg])
    bonus = 0.0
    if source_type == "page":
        bonus += 0.35       # page sitemap bonus retained
        bonus += 0.05
    elif source_type == "post":
        bonus -= 0.20
    elif source_type == "tax":
        bonus -= 0.25
    if "/blog/" in path or "/news/" in path:
        bonus -= 0.20
    if re.search(r"/\d{4}/\d{2}/", path):
        bonus -= 0.20
    if depth <= 1:
        bonus += 0.12
    if is_nav:
        bonus += 0.15
    return bonus

# ---------- Out-of-domain detection ----------
POL_CONTACT_PAT = re.compile(r"\b(contact|email|call)\s+(senator|representative|rep|governor|mayor|mp|president)\b", re.I)

def _capitalized_tokens(original_kw: str) -> set:
    caps = set()
    for m in re.finditer(r"\b[A-Z][a-z]+(?:[-'][A-Za-z]+)?\b", original_kw or ""):
        caps.add(_norm_token(m.group(0)))
    return caps

def _looks_out_of_domain(original_kw: str, site_lex: set, base_url: str) -> bool:
    tokens = _ntokens(original_kw)
    if not tokens: return False
    if POL_CONTACT_PAT.search(original_kw or ""):
        return True
    domtoks = _domain_tokens(base_url)
    distinctive = [t for t in tokens if t.isalpha() and len(t) >= 4 and t not in STOPWORDS]
    unknown = [t for t in distinctive if (t not in site_lex and t not in domtoks)]
    if not unknown:
        return False
    cap = _capitalized_tokens(original_kw)
    if any(t in unknown for t in cap):
        return True
    if "contact" in tokens and unknown:
        return True
    if len(unknown) >= 2:
        return True
    return False

# ---------- VEO intent detection ----------
VEO_NAV_TOKS = {"contact","contacts","phone","call","address","directions","hours","location","locations","visit","map","email"}

def _veo_intent(profile: Dict, nav_anchor_map: Dict[str, Set[str]]) -> Tuple[bool, bool]:
    u = profile.get("url","")
    key = _url_key(u)
    nav_toks = nav_anchor_map.get(key, set())
    nav_hit = any(t in nav_toks for t in VEO_NAV_TOKS)
    page_hit = profile.get("veo_ready", False) or _is_contact_like(u)
    home_nap = _is_home(u) and profile.get("veo_ready", False)
    return (nav_hit or page_hit), home_nap

# ---------- Helpers for sitemap tie-break ----------
def _parse_lastmod_ts(s: Optional[str]) -> float:
    if not s: return 0.0
    ss = s.strip()
    with contextlib.suppress(Exception):
        if ss.endswith("Z"): ss = ss[:-1] + "+00:00"
        return datetime.fromisoformat(ss).timestamp()
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return datetime.fromisoformat(m.group(1)).timestamp()
        except Exception:
            return 0.0
    return 0.0

# ---------- Mapping ----------
def map_keywords_to_urls(df: pd.DataFrame, kw_col: Optional[str], vol_col: str, kd_col: str,
                         base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> pd.Series:
    url_list, srcmap, smeta = cached_discover_and_sources(base_url, include_subdomains, use_sitemap_first)
    profiles = cached_fetch_profiles(tuple(url_list))
    profiles = [p for p in profiles if p.get("weights")]
    if not profiles:
        return pd.Series([""]*len(df), index=df.index, dtype="string")

    nav_anchor_map, all_nav_tokens = cached_nav(base_url)

    domain_toks = _domain_tokens(base_url)
    page_counts: Dict[str,int] = defaultdict(int)
    post_counts: Dict[str,int] = defaultdict(int)
    src_by_key = { _url_key(k): v for k, v in srcmap.items() }

    def _src_type_for_profile(p: Dict) -> str:
        return (
            src_by_key.get(_url_key(p.get("url",""))) or
            src_by_key.get(_url_key(p.get("final_url",""))) or
            src_by_key.get(_url_key(p.get("requested_url",""))) or
            "other"
        )

    nav_keys = set(nav_anchor_map.keys())
    def _is_nav(p: Dict) -> bool:
        return (_url_key(p.get("url","")) in nav_keys) or (_url_key(p.get("requested_url","")) in nav_keys)

    global_counts: Dict[str,int] = defaultdict(int)
    for p in profiles:
        stype = _src_type_for_profile(p)
        is_nav_flag = _is_nav(p)
        is_page = _is_page_like(stype, p["url"], is_nav_flag)
        is_post = _is_post_like(stype, p["url"])
        toks = set(p.get("weights", {}).keys())
        toks.update((p.get("title_h1_norm") or "").split())
        toks.update(_ntokens(p.get("title","") or ""))
        for t in toks:
            global_counts[t] += 1
            if is_page: page_counts[t] += 1
            if is_post: post_counts[t] += 1

    for key, toks in nav_anchor_map.items():
        for t in toks:
            page_counts[t] += 3
    for t in domain_toks:
        page_counts[t] += 2

    page_core_lex = set(page_counts.keys())
    post_lex = set(post_counts.keys())
    site_lex = set(page_core_lex) | set(post_lex) | set(all_nav_tokens) | set(domain_toks)

    post_heavy: Dict[str,bool] = {}
    very_common: Dict[str,bool] = {}
    total_docs = max(1, len(profiles))
    for t in set(list(page_counts.keys()) + list(post_counts.keys()) + list(global_counts.keys())):
        r = post_counts.get(t,0) / max(1, page_counts.get(t,0))
        post_heavy[t] = (r >= 5.0)
        very_common[t] = (global_counts.get(t,0) >= max(3, total_docs // 3))

    vols = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).clip(lower=0)
    max_log = float((vols + 1).apply(lambda x: math.log(1 + x)).max()) or 1.0
    def strat_weights():
        if scoring_mode == "Low Hanging Fruit": return 0.30, 0.40, 0.30
        elif scoring_mode == "In The Game":     return 0.30, 0.35, 0.35
        else:                                    return 0.30, 0.20, 0.50
    W_FIT, W_KD, W_VOL = strat_weights()

    def _has_core_token(tokens: List[str]) -> bool:
        for t in tokens:
            if len(t) >= 4 and t not in STOPWORDS and t in page_core_lex:
                return True
        return False

    # RELAXED penalties
    def _post_heavy_penalty(tokens: List[str]) -> float:
        cnt = sum(1 for t in tokens if post_heavy.get(t, False))
        if cnt >= 2: return -0.25   # was -0.30
        if cnt == 1: return -0.15   # was -0.20
        return 0.0

    def _commonness_penalty(tokens: List[str]) -> float:
        cnt = sum(1 for t in tokens if very_common.get(t, False))
        return -0.03 * cnt          # was -0.05 * cnt

    TOP_K = 5
    kw_candidates: Dict[int, List[Tuple[str,float,float,str,float]]] = {}
    kw_slot: Dict[int,str] = {}
    kw_rank: Dict[int,float] = {}
    head_noun_by_kw: Dict[int, str] = {}

    for idx, row in df.iterrows():
        kw = str(row.get(kw_col, "")) if kw_col else str(row.get("Keyword",""))

        if _looks_out_of_domain(kw, site_lex, base_url):
            kw_candidates[idx] = []
            kw_slot[idx] = "SEO"
            kw_rank[idx] = 0.0
            continue

        cats = set(categorize_keyword(kw))
        if "VEO" in cats: slot = "VEO"
        elif "AIO" in cats: slot = "AIO"
        else: slot = "SEO"

        tokens_norm = _ntokens(kw)
        head = _head_noun(tokens_norm)
        head_noun_by_kw[idx] = head

        # ----- RELAXED core requirement -----
        # SEO: core OR head present in nav/domain; VEO: no core requirement; AIO: keep core
        need_core = False
        if slot == "SEO":
            in_core = _has_core_token(tokens_norm)
            head_ok = bool(head) and (head in all_nav_tokens or head in domain_toks)
            need_core = not (in_core or head_ok)
        elif slot == "VEO":
            need_core = False
        else:
            need_core = not _has_core_token(tokens_norm)
        if need_core:
            kw_candidates[idx] = []
            kw_slot[idx] = slot
            kw_rank[idx] = 0.0
            continue

        fits_page: List[Tuple[str,float,float,str,float]] = []
        fits_other: List[Tuple[str,float,float,str,float]] = []
        best_page_probe: Optional[Tuple[str,float,float,str,float]] = None  # keep best page even if gated

        for p in profiles:
            base_fit = _fit_score(kw, p)
            if base_fit <= 0:
                continue
            stype = _src_type_for_profile(p)
            is_nav_flag = _is_nav(p)
            is_page_like = _is_page_like(stype, p["url"], is_nav_flag)

            # coverage & salience
            title_tokens = set((p.get("title_h1_norm") or "").split())
            title_raw = (p.get("title") or "").strip().lower()
            covered = sum(1 for t in tokens_norm if t in title_tokens)
            covered_ratio = covered / max(1, len(tokens_norm))
            lead_tokens = set((p.get("lead_norm") or "").split())
            lead_cov = sum(1 for t in tokens_norm if t in lead_tokens) / max(1, len(tokens_norm))

            v_intent, home_nap = _veo_intent(p, nav_anchor_map)
            a_score = p.get("a_score", 0.0)

            # bonuses/penalties
            bonus = _url_priority_bonus(p["url"], is_nav_flag, stype)
            if is_page_like:
                bonus += _post_heavy_penalty(tokens_norm)

            slug_toks = _slug_tokens(p["url"])
            nav_anchor_toks = nav_anchor_map.get(_url_key(p["url"]), set())
            if head and (head in slug_toks or head in title_tokens): bonus += 0.12
            if head and (head in nav_anchor_toks): bonus += 0.12
            phrase_str = " ".join(tokens_norm)
            if head and (title_raw.startswith(head + " ") or title_raw == head):
                bonus += 0.12
            elif phrase_str and (title_raw.startswith(phrase_str + " ") or title_raw == phrase_str):
                bonus += 0.12

            a_bonus = 0.0
            v_bonus = 0.0
            if slot == "AIO":
                a_bonus = min(0.22, a_score)
            if slot == "VEO":
                if v_intent: v_bonus += 0.18
                if home_nap: v_bonus += 0.08

            bonus += _commonness_penalty(tokens_norm)

            f = max(0.0, min(2.0, base_fit + bonus + a_bonus + v_bonus))

            # VEO soft gate — RELAXED threshold 0.52 (from 0.60)
            if slot == "VEO" and not v_intent:
                f = max(0.0, f - 0.20)
                if f < 0.52:
                    if is_page_like:
                        if (best_page_probe is None) or (f > best_page_probe[1]):
                            best_page_probe = (p["url"], f, covered_ratio, stype, a_score)
                    continue

            # Class-aware salience gates
            passed = True
            if slot == "SEO":
                if (lead_cov < 0.18) and (covered_ratio < 0.50):
                    passed = False
            elif slot == "AIO":
                if (lead_cov < 0.12) and (covered_ratio < 0.40) and (a_score < 0.30):
                    passed = False
            else:  # VEO
                if lead_cov < 0.10 and v_intent and covered_ratio < 0.33:
                    passed = False

            # SEO head rule — SOFTER
            if passed and slot == "SEO" and head:
                phrase = " ".join(tokens_norm)
                head_in_nav = head in nav_anchor_map.get(_url_key(p["url"]), set())
                if (head not in title_tokens) and (head not in slug_toks) and not head_in_nav:
                    # allow if phrase present OR decent coverage OR decent fit
                    if not phrase or phrase not in (p.get("title_h1_norm") or ""):
                        if (covered_ratio < 0.45) and (f < 0.60):
                            passed = False

            if not passed:
                if is_page_like:
                    if (best_page_probe is None) or (f > best_page_probe[1]):
                        best_page_probe = (p["url"], f, covered_ratio, stype, a_score)
                continue

            if is_page_like:
                fits_page.append((p["url"], f, covered_ratio, stype, a_score))
            else:
                fits_other.append((p["url"], f, covered_ratio, stype, a_score))

        fits_page.sort(key=lambda x: x[1], reverse=True)
        fits_other.sort(key=lambda x: x[1], reverse=True)

        # ---------- Forced include: ensure at least one PAGE candidate (RELAXED) ----------
        if not fits_page and fits_other and best_page_probe is not None:
            best_fit = fits_other[0][1]
            page_fit = best_page_probe[1]
            # was: >= 50% and >= (min - 0.03)
            if (page_fit >= 0.45 * best_fit) and (page_fit >= (_MIN_SEO_PAGE - 0.05 if slot=="SEO" else _MIN_AIO_PAGE - 0.05)):
                fits_page = [best_page_probe]

        # ----- Selection with class-aware epsilon -----
        fits: List[Tuple[str,float,float,str,float]] = []
        if slot == "AIO" and fits_page and fits_other:
            best_page = fits_page[0][1]
            # AIO post can win if a_score >= 0.33 (was 0.35) and fit >= page + 0.06 (was +0.08)
            aio_posts = [t for t in fits_other if _is_post_like(t[3], t[0]) and t[4] >= 0.33]
            if aio_posts:
                aio_posts.sort(key=lambda x: x[1], reverse=True)
                best_post = aio_posts[0][1]
                if best_post >= (best_page + 0.06):
                    fits.extend(aio_posts[:TOP_K])
                    if len(fits) < TOP_K:
                        fits.extend(fits_page[:TOP_K - len(fits)])
                else:
                    # prefer page when close (epsilon widened lightly to 0.12)
                    if best_page >= (best_post - 0.12):
                        fits.extend(fits_page[:TOP_K])
                        if len(fits) < TOP_K:
                            fits.extend(fits_other[:TOP_K - len(fits)])
                    else:
                        fits.extend(fits_page[:TOP_K])
                        if len(fits) < TOP_K:
                            fits.extend(fits_other[:TOP_K - len(fits)])
            else:
                fits.extend(fits_page[:TOP_K])
                if len(fits) < TOP_K:
                    fits.extend(fits_other[:TOP_K - len(fits)])
        else:
            # Non-AIO: page-first with epsilon widened to 0.14 (was 0.10)
            if fits_page and fits_other:
                best_page = fits_page[0][1]
                best_other = fits_other[0][1]
                best_overall = max(best_page, best_other)
                if best_page >= (best_overall - 0.14):
                    fits.extend(fits_page[:TOP_K])
                    if len(fits) < TOP_K:
                        fits.extend(fits_other[:TOP_K - len(fits)])
                else:
                    fits.extend(fits_page[:TOP_K])
                    if len(fits) < TOP_K:
                        fits.extend(fits_other[:TOP_K - len(fits)])
            elif fits_page:
                fits = fits_page[:TOP_K]
            else:
                fits = fits_other[:TOP_K]

        if not fits:
            kw_candidates[idx] = []
            kw_slot[idx] = slot
            kw_rank[idx] = 0.0
            continue

        # sitemap tie-breakers within 0.04
        top_fit = fits[0][1]
        def _smeta_boost(u: str, base_fit: float) -> float:
            b = 0.0
            md = smeta.get(u, {})
            close = (top_fit - base_fit) <= 0.04
            if close and md.get("priority"):
                try:
                    pr = float(md.get("priority") or 0)
                    if pr >= 0.8: b += 0.03
                    elif pr >= 0.5: b += 0.015
                except Exception:
                    pass
            if close and md.get("lastmod") and (slot == "AIO"):
                if _parse_lastmod_ts(md.get("lastmod")) > 0:
                    b += 0.01
            return b

        fits = [(u, f + _smeta_boost(u, f), cr, st, a) for (u, f, cr, st, a) in fits]
        fits.sort(key=lambda x: x[1], reverse=True)

        kw_candidates[idx] = fits[:TOP_K]
        kw_slot[idx] = slot

        best_fit = fits[0][1]
        kd_val = float(pd.to_numeric(row.get(kd_col,0), errors="coerce") or 0)
        vol_val = float(pd.to_numeric(row.get(vol_col,0), errors="coerce") or 0)
        fit_norm = best_fit / 2.0
        kd_norm = max(0.0, 1.0 - kd_val/100.0)
        vol_norm = math.log(1 + max(0.0, vol_val)) / max_log

        lhf_opportunity = vol_norm * kd_norm
        if scoring_mode == "Low Hanging Fruit":
            kw_rank[idx] = 0.30*fit_norm + 0.30*kd_norm + 0.20*vol_norm + 0.20*lhf_opportunity
        elif scoring_mode == "In The Game":
            kw_rank[idx] = 0.35*fit_norm + 0.30*kd_norm + 0.35*vol_norm
        else:
            kw_rank[idx] = 0.40*fit_norm + 0.20*kd_norm + 0.40*vol_norm

        # --- class-specific minimum fit across candidates (RELAXED mins already set) ---
        def _class_min_for_type(slot_name: str, is_post: bool) -> float:
            if slot_name == "SEO": return _MIN_SEO_POST if is_post else _MIN_SEO_PAGE
            if slot_name == "AIO": return _MIN_AIO_POST if is_post else _MIN_AIO_PAGE
            return _MIN_VEO_POST if is_post else _MIN_VEO_PAGE

        any_ok = False
        for (u, f, cr, st, a) in fits:
            is_post = _is_post_like(st, u)
            if f >= _class_min_for_type(slot, is_post):
                any_ok = True
                break
        if not any_ok:
            kw_candidates[idx] = []
            kw_rank[idx] = 0.0

    # ---------- Assignment (no backfill; caps enforced) ----------
    caps = {"VEO":1, "AIO":1, "SEO":2}
    assigned: Dict[str, Dict[str, object]] = {}
    for p in profiles:
        assigned[p["url"]] = {"VEO":None, "AIO":None, "SEO":[]}

    mapped = {i:"" for i in df.index}

    def _seo_allowed(stype: str, u: str, fit: float, covered_ratio: float, head: str, title_tokens: Set[str]) -> bool:
        if _is_post_like(stype, u):
            # slightly looser than before, but still tough
            strong = (covered_ratio >= 0.58 and fit >= 0.78)
            head_ok = head and (head in title_tokens or head in _slug_tokens(u))
            phrase_ok = (head is None) or (head == "")  # if no obvious head noun
            if strong and (head_ok or phrase_ok):
                return True
            return False
        return True

    def assign_slot(slot_name: str):
        ids = [i for i,s in kw_slot.items() if s == slot_name]
        if scoring_mode == "Low Hanging Fruit":
            opp = {}
            for i in ids:
                row = df.loc[i]
                kd_val = float(pd.to_numeric(row.get(kd_col,0), errors="coerce") or 0)
                vol_val = float(pd.to_numeric(row.get(vol_col,0), errors="coerce") or 0)
                kd_norm = max(0.0, 1.0 - kd_val/100.0)
                vol_norm = math.log(1 + max(0.0, vol_val)) / (float((vols + 1).apply(lambda x: math.log(1 + x)).max()) or 1.0)
                opp[i] = vol_norm * kd_norm
            ids.sort(key=lambda i: (-opp.get(i,0.0), -kw_rank.get(i,0.0), i))
        else:
            ids.sort(key=lambda i: (-kw_rank.get(i,0.0), i))

        for i in ids:
            choices = kw_candidates.get(i, [])
            if not choices: continue
            head = head_noun_by_kw.get(i, "")
            for j, (u, fit, covered_ratio, stype, a_score) in enumerate(choices):
                title_tokens = set((next((p["title_h1_norm"] for p in profiles if p["url"]==u), "")).split())
                if slot_name == "SEO":
                    if not _seo_allowed(stype, u, fit, covered_ratio, head, title_tokens):
                        continue
                    if head and (head not in _slug_tokens(u)) and (head not in title_tokens):
                        continue
                if slot_name in {"VEO","AIO"}:
                    if assigned[u][slot_name] is None and (j == 0 or fit >= _ALT_FIT_MIN):
                        assigned[u][slot_name] = i; mapped[i] = u; break
                else:
                    if (len(assigned[u]["SEO"]) < caps["SEO"]) and (j == 0 or fit >= _ALT_FIT_MIN):
                        assigned[u]["SEO"].append(i); mapped[i] = u; break

    assign_slot("VEO"); assign_slot("AIO"); assign_slot("SEO")

    for u, slots in assigned.items():
        if isinstance(slots["SEO"], list) and len(slots["SEO"]) > caps["SEO"]:
            for drop_idx in slots["SEO"][caps["SEO"]:]: mapped[drop_idx] = ""
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
        sc = calculate_score(vol_val, kd_val); label = LABEL_MAP.get(sc,"Not rated"); color = COLOR_MAP.get(sc,"#9ca3af")
        st.markdown(f"<div style='background-color:{color}; padding:16px; border-radius:12px; text-align:center;'><span style='font-size:22px; font-weight:bold; color:#000;'>Score: {sc} • Tier: {label}</span></div>", unsafe_allow_html=True)
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

# ---------- CSV ingest ----------
if uploaded is not None:
    raw = uploaded.getvalue()
    def try_read(bytes_data: bytes) -> pd.DataFrame:
        trials = [
            {"encoding": None, "sep": None, "engine":"python"},
            {"encoding":"utf-8", "sep": None, "engine":"python"},
            {"encoding":"utf-8-sig", "sep": None, "engine":"python"},
            {"encoding":"ISO-8859-1", "sep": None, "engine":"python"},
            {"encoding":"cp1252", "sep": None, "engine":"python"},
            {"encoding":"utf-16", "sep": None, "engine":"python"},
            {"encoding": None, "sep": ",", "engine":"python"},
            {"encoding": None, "sep": "\t", "engine":"python"},
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
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\s]","",regex=True).str.replace("%","",regex=False)
        df[kd_col]  = df[kd_col].astype(str).str.replace(r"[,\s]","",regex=True).str.replace("%","",regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col]  = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

        scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score","Tier","Eligible","Reason","Category"]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes":1,"No":0}).fillna(0)
        export_df = export_df.sort_values(by=["_EligibleSort", kd_col, vol_col], ascending=[False, True, False], kind="mergesort").drop(columns=["_EligibleSort"])

        # -------- URL Mapping --------
        map_series = pd.Series([""]*len(export_df), index=export_df.index, dtype="string")
        if base_site_url.strip():
            sig_cols = [c for c in [kw_col, vol_col, kd_col] if c]
            try: sig_df = export_df[sig_cols].copy()
            except Exception: sig_df = export_df[[col for col in sig_cols if col in export_df.columns]].copy()
            sig_csv = sig_df.fillna("").astype(str).to_csv(index=False)
            sig_base = f"{_MAPPER_VERSION}|{_normalize_base(base_site_url.strip()).lower()}|{scoring_mode}|{kw_col}|{vol_col}|{kd_col}|{len(export_df)}"
            signature = hashlib.md5((sig_base + "\n" + sig_csv).encode("utf-8")).hexdigest()
            if "map_cache" not in st.session_state: st.session_state["map_cache"] = {}
            cache = st.session_state["map_cache"]

            if signature in cache and len(cache[signature]) == len(export_df):
                map_series = pd.Series(cache[signature], index=export_df.index, dtype="string")
            else:
                loader = st.empty()
                loader.markdown(
                    """
                    <div style='display:flex;align-items:center;gap:12px;'>
                      <div style='font-size:28px'>🚀</div>
                      <div style='font-weight:700;'>Mapping keywords to your site…</div>
                    </div>
                    <style>@keyframes bob { from { transform: translateY(0); } to { transform: translateY(-6px); } }
                    div[style*="font-size:28px"] { animation: bob .6s ease-in-out infinite alternate; }</style>
                    """, unsafe_allow_html=True)
                with st.spinner("Launching fast crawl & scoring fit…"):
                    map_series = map_keywords_to_urls(
                        export_df, kw_col=kw_col, vol_col=vol_col, kd_col=kd_col,
                        base_url=base_site_url.strip(), include_subdomains=True, use_sitemap_first=True
                    )
                loader.empty()
                cache[signature] = map_series.fillna("").astype(str).tolist()
        else:
            st.info("Enter a Base site URL to enable mapping.")

        export_df["Map URL"] = map_series
        export_cols = base_cols + ["Strategy","Map URL"]
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
