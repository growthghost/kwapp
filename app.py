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
from mapping import weighted_map_keywords
from crawler import fetch_profiles


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
BRAND_BG = "#FFFFFF"     # page background (white)
BRAND_INK = "#242F40"    # dark blue text on white
BRAND_ACCENT = "#329662" # green
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
html, body, [class^="css"], [class*=" css"] {{ color: var(--ink) !important; }}
h1, h2, h3, h4, h5, h6 {{ color: var(--ink) !important; }}

/* OutrankIQ green header bar — full-bleed */
.oiq-header {{
  background: var(--accent);
  color: #ffffff;
  padding: 22px 20px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
}}
.oiq-bleed {{
  margin-left: calc(50% - 50vw);
  margin-right: calc(50% - 50vw);
  width: 100vw;
  border-radius: 0 !important;
}}
.oiq-header-inner {{
  max-width: 1000px;
  margin: 0 auto;
  padding-left: 16px;
  text-align: left;
}}
.oiq-header .oiq-title {{
  font-size: 40px;
  font-weight: 800;
  letter-spacing: 0.2px;
}}
.oiq-header .oiq-sub {{
  margin-top: 6px;
  font-size: 16px;
  opacity: .98;
}}

/* Inputs: white surface */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--light) !important;
  color: var(--ink) !important;
  border-radius: 8px !important;
}}

/* Base border (light) */
.stNumberInput input,
.stTextInput input {{
  border: 2px solid rgba(36,47,64,0.08) !important;
}}
/* Select focus = BLUE glow */
.stSelectbox div[data-baseweb="select"]:focus-within > div {{
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 3px rgba(36,47,64,.35) !important;
  outline: none !important;
}}
/* Number & text focus = GREEN glow */
.stNumberInput input:focus,
.stNumberInput input:focus-visible,
.stNumberInput:focus-within input,
.stTextInput input:focus,
.stTextInput input:focus-visible {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
  outline: none !important;
}}

/* --- FORCE BLUE BASE BORDERS (requested) --- */
.stSelectbox div[data-baseweb="select"] > div {{
  border: 2px solid var(--ink) !important;        /* dropdown default border = BLUE */
}}
.stNumberInput input {{
  border: 2px solid var(--ink) !important;        /* A/B boxes default border = BLUE */
}}

/* Number steppers */
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

/* BLUE labels */
div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label,
div[data-testid="stTextInput"] > label,
div[data-testid="stCheckbox"] > label {{ color: var(--ink) !important; font-weight: 700; }}

/* Expander header ("See example") — blue text + blue arrow */
[data-testid="stExpander"] > details > summary {{
  background: #ffffff !important;
  border: 1px solid rgba(36,47,64,0.12) !important;
  border-radius: 10px !important;
  color: var(--ink) !important;
  font-weight: 700;
}}
[data-testid="stExpander"] > details > summary::-webkit-details-marker {{ color: var(--ink) !important; }}
[data-testid="stExpander"] > details > summary::marker {{ color: var(--ink) !important; }}

/* Expander body */
[data-testid="stExpander"] .st-emotion-cache-1h9usn1,
[data-testid="stExpander"] > details > div {{
  background: #ffffff !important;
}}

/* Uploader + Browse button (BLUE -> WHITE on hover) */
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
  transition: background-color .15s ease, color .15s ease, border-color .15s ease, box-shadow .15s ease;
}}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] label:hover,
[data-testid="stFileUploaderDropzone"] [role="button"]:hover {{
  background-color: var(--light) !important; 
  color: var(--ink) !important;             
  border-color: var(--ink) !important;      
  box-shadow: 0 0 0 3px rgba(36,47,64,.15) !important;
}}

/* Make Streamlit spinner text blue */
[data-testid="stSpinner"] * {{ color: var(--ink) !important; }}

/* Custom loader (blue circular spinner) */
.oiq-loader {{ display:flex; align-items:center; gap:12px; }}
.oiq-spinner {{
  width:22px; height:22px; border:3px solid rgba(36,47,64,0.25);
  border-top-color: var(--ink);
  border-radius:50%; animation: oiq-spin 1s linear infinite;
}}
@keyframes oiq-spin {{ to {{ transform: rotate(360deg); }} }}
.oiq-loader-text {{ color: var(--ink); font-weight:700; }}

/* Tables */
.stDataFrame, .stDataFrame *, .stTable, .stTable * {{ color: var(--ink) !important; }}

/* Action buttons */
.stButton > button, .stDownloadButton > button {{
  background-color: var(--accent) !important; 
  color: var(--ink) !important;
  border: 2px solid rgba(36,47,64,0.08) !important; 
  border-radius: 10px !important; 
  font-weight: 700 !important;
  box-shadow: 0 2px 0 rgba(0,0,0,.12);
  transition: background-color .15s ease, color .15s ease, border-color .15s ease, box-shadow .15s ease;
}}
/* Hover/focus/active = transparent, BLUE text, BLUE outline */
.stButton > button:hover, .stDownloadButton > button:hover,
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
.stButton > button:active, .stDownloadButton > button:active {{
  background-color: transparent !important; 
  color: var(--ink) !important; 
  border-color: var(--ink) !important;
  box-shadow: 0 0 0 3px rgba(36,47,64,.15) !important;
}}

/* Strategy banner */
.info-banner {{
  background: linear-gradient(90deg, var(--ink) 0%, var(--accent) 100%);
  padding: 16px; border-radius: 12px; color: var(--light);
}}

/* Footer */
.oiq-footer {{
  color: var(--ink);
  font-size: 13px;
  opacity: .95;
  text-align: left;
  margin-top: 24px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
<div class="oiq-header oiq-bleed">
  <div class="oiq-header-inner">
    <div class="oiq-title">OutrankIQ</div>
    <div class="oiq-sub">Score keywords by Search Volume (A) and Keyword Difficulty (B) — with selectable scoring strategies.</div>
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
    "Low Hanging Fruit":"Keywords that can be used to rank quickly with minimal effort. Ideal for new content or low-authority sites. Try targeting long-tail keywords, create quick-win content, and build a few internal links.",
    "In The Game":"Moderate difficulty keywords that are within reach for growing sites. Focus on optimizing content, earning backlinks, and matching search intent to climb the ranks.",
    "Competitive":"High-volume, high-difficulty keywords dominated by authoritative domains. Requires strong content, domain authority, and strategic SEO to compete. Great for long-term growth.",
}

# ---------- Strategy ----------
scoring_mode = st.selectbox("Choose Scoring Strategy", ["Low Hanging Fruit","In The Game","Competitive"])

# Reset mapping state on strategy switch
if "last_strategy" not in st.session_state:
    st.session_state["last_strategy"] = scoring_mode
if st.session_state.get("last_strategy") != scoring_mode:
    st.session_state["last_strategy"] = scoring_mode
    for k in ["map_cache","map_result","map_signature","map_ready","mapping_running"]:
        st.session_state.pop(k, None)

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

# ---------- Tokenization & normalization ----------
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

PHRASE_MAP = [
    (re.compile(r"\bget[ -]?in[ -]?touch\b", re.I), "contact"),
    (re.compile(r"\breach[ -]?out\b", re.I), "contact"),
    (re.compile(r"\bemail\s*us\b", re.I), "contact"),
    (re.compile(r"\bcall\s*us\b", re.I), "contact"),
    (re.compile(r"\btalk\s*to\s*us\b", re.I), "contact"),
    (re.compile(r"\bspeak\s*with\b", re.I), "contact"),
    (re.compile(r"\brequest\s+(info|information)\b", re.I), "contact"),
    (re.compile(r"\bbook\s+(a\s*)?call\b", re.I), "contact"),
    (re.compile(r"\bschedule\s+(a\s*)?call\b", re.I), "contact"),
    (re.compile(r"\bappointment\b", re.I), "contact"),

    (re.compile(r"\bnon\s*profit(s)?\b", re.I), "nonprofit"),
    (re.compile(r"\bnon[- ]?profit(s)?\b", re.I), "nonprofit"),

    (re.compile(r"\bnear\s*me\b", re.I), "nearme"),
    (re.compile(r"\bnear\s*you\b", re.I), "nearyou"),
    (re.compile(r"\bnearby\b", re.I), "nearme"),
    (re.compile(r"\bclosest\b", re.I), "nearme"),

    (re.compile(r"\bfind\s*us\b", re.I), "locations"),
    (re.compile(r"\bvisit\s*us\b", re.I), "locations"),
    (re.compile(r"\bour\s*office(s)?\b", re.I), "locations"),
    (re.compile(r"\bour\s*location(s)?\b", re.I), "locations"),
    (re.compile(r"\bwhere\s*we\s*are\b", re.I), "locations"),

    (re.compile(r"\bmake\s+(a\s*)?gift\b", re.I), "donate"),
    (re.compile(r"\bgive\s*now\b", re.I), "donate"),
    (re.compile(r"\bsupport\s+our\s+work\b", re.I), "donate"),
    (re.compile(r"\bcontribute\b", re.I), "donate"),
    (re.compile(r"\bmake\s+(a\s*)?donation\b", re.I), "donate"),

    (re.compile(r"\bget\s+involved\b", re.I), "volunteer"),
    (re.compile(r"\bserve\s+with\s+us\b", re.I), "volunteer"),
    (re.compile(r"\bjoin\s+us\b", re.I), "volunteer"),
    (re.compile(r"\btake\s+action\b", re.I), "volunteer"),
    (re.compile(r"\bsign\s+up\s+to\s+help\b", re.I), "volunteer"),

    (re.compile(r"\bcalendar\b", re.I), "events"),
    (re.compile(r"\b(upcoming|workshops|trainings|classes|webinars)\b", re.I), "events"),

    (re.compile(r"\bwho\s+we\s+are\b", re.I), "about"),
    (re.compile(r"\bour\s+story\b", re.I), "about"),
    (re.compile(r"\bmission\b", re.I), "about"),
    (re.compile(r"\bvision\b", re.I), "about"),
    (re.compile(r"\bleadership\b", re.I), "about"),
    (re.compile(r"\b(team|board|staff)\b", re.I), "about"),

    (re.compile(r"\binitiatives\b", re.I), "program"),
    (re.compile(r"\bofferings\b", re.I), "service"),
    (re.compile(r"\bsolutions\b", re.I), "service"),
    (re.compile(r"\bministries\b", re.I), "service"),

    (re.compile(r"\blibrary\b", re.I), "resources"),
    (re.compile(r"\btoolkit\b", re.I), "resources"),
    (re.compile(r"\bdownloads?\b", re.I), "resources"),
    (re.compile(r"\btemplates?\b", re.I), "resources"),
    (re.compile(r"\bguides?\b", re.I), "resources"),
    (re.compile(r"\bplaybook\b", re.I), "resources"),

    (re.compile(r"\bjobs?\b", re.I), "careers"),
    (re.compile(r"\bemployment\b", re.I), "careers"),
    (re.compile(r"\b(opening|openings)\b", re.I), "careers"),
    (re.compile(r"\bwe'?re\s+hiring\b", re.I), "careers"),
]

_SYN_MAP = {
    "connect":"contact","connected":"contact","connecting":"contact","contacts":"contact",
    "support":"contact","helpdesk":"contact","helpline":"contact","help-line":"contact",
    "nearby":"nearme","nearyou":"nearme","directions":"directions","address":"address",
    "offices":"locations","office":"locations","locations":"locations","visit":"locations","findus":"locations",
    "organisations":"organization","organisation":"organization","organizations":"organization","orgs":"organization","org":"organization",
    "nonprofit":"nonprofit","ngo":"organization","charity":"organization","foundation":"organization",
    "assist":"help","assists":"help","assistance":"help","caregiving":"help","caregiver":"help","caregivers":"help",
    "disabilities":"disability","disabled":"disability",
    "programs":"program","programme":"program","services":"service",
    "contribute":"donate","give":"donate","giving":"donate","donation":"donate","donations":"donate",
    "involved":"volunteer","volunteering":"volunteer","volunteers":"volunteer",
    "calendar":"events","workshops":"events","trainings":"events","classes":"events","webinars":"events",
    "jobs":"careers","employment":"careers","opening":"careers","openings":"careers","hiring":"careers",
    "library":"resources","toolkit":"resources","templates":"resources","guides":"resources","playbook":"resources","downloads":"resources",
    "aboutus":"about","mission":"about","vision":"about","team":"about","board":"about","staff":"about",
}

STOPWORDS = {
    "the","and","for","to","a","an","of","with"," in","on","at","by","from","about",
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
    if t in _SYN_MAP:
        t = _SYN_MAP[t]
    if t.endswith("ies") and len(t) > 3: t = t[:-3] + "y"
    elif t.endswith("es") and len(t) > 4: t = t[:-2]
    elif t.endswith("s") and len(t) > 3: t = t[:-1]
    if t.endswith("ing") and len(t) > 5: t = t[:-3]
    elif t.endswith("ed") and len(t) > 4: t = t[:-2]
    if t in _SYN_MAP:
        t = _SYN_MAP[t]
    return t

def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower()) if text else []

def _ntokens(text: str) -> List[str]:
    text = _normalize_phrases(text or "")
    return [_norm_token(t) for t in _tokenize(text)]

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
            resp = session.get(url, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}, timeout=timeout, stream=True, allow_redirects=True)
        elif requests is not None:
            resp = requests.get(url, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}, timeout=timeout, stream=True, allow_redirects=True)
        else:
            return None
        if resp.status_code >= 400: return None
        ctype = resp.headers.get("Content-Type","").lower()
        raw = resp.content[:350_000]
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
    txt = _fetch_text_requests(robots_url, session, (5,8))
    if not txt: return []
    maps = []
    for line in txt.splitlines():
        if line.lower().startswith("sitemap:"):
            sm = line.split(":",1)[1].strip()
            if sm: maps.append(sm)
    return maps

def _parse_sitemap_xml_entries(xml_text: str) -> List[Tuple[str, Optional[float], Optional[str]]]:
    if not xml_text:
        return []
    entries: List[Tuple[str, Optional[float], Optional[str]]] = []
    for m in re.finditer(r"<url>\s*(.*?)</url>", xml_text, re.I | re.S):
        block = m.group(1)
        lm = re.search(r"<loc>\s*([^<]+)\s*</loc>", block, re.I)
        if not lm:
            continue
        loc = lm.group(1).strip()
        pr: Optional[float] = None
        lastmod: Optional[str] = None
        pm = re.search(r"<priority>\s*([0-9.]+)\s*</priority>", block, re.I)
        if pm:
            with contextlib.suppress(Exception):
                pr = float(pm.group(1))
        lmm = re.search(r"<lastmod>\s*([^<]+)\s*</lastmod>", block, re.I)
        if lmm:
            lastmod = lmm.group(1).strip()
        entries.append((loc, pr, lastmod))
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
    if depth > 3 or len(out) >= 120 or sm_url in seen: return
    seen.add(sm_url)
    xml = _fetch_text_requests(sm_url, session, (5,8))
    if not xml: return
    entries = _parse_sitemap_xml_entries(xml)
    is_index = bool(re.search(r"<sitemapindex", xml, re.I)) or any(u.lower().endswith((".xml",".xml.gz")) for (u,_,_) in entries)

    if is_index:
        children = [u for (u,_,_) in entries if u]
        children.sort(key=_sm_bucket)  # prefer page sitemaps first
        for child in children:
            if len(out) >= 120: break
            if child.lower().endswith((".xml",".xml.gz")):
                _collect_sitemap_urls(child, session, base_host, base_root, include_subdomains,
                                      seen, out, srcmap, meta, parent_type=None, depth=depth+1)
        return

    stype = parent_type if parent_type else _sm_classify(sm_url)
    for (u, pr, lm) in entries:
        if len(out) >= 120: break
        if _same_site(u, base_host, base_root, include_subdomains):
            out.append(u)
            key = _url_key(u)
            srcmap.setdefault(key, stype)
            md = meta.setdefault(key, {})
            if pr is not None: md["priority"] = f"{pr:.3f}"
            if lm: md["lastmod"] = lm

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
            maps.sort(key=_sm_bucket)  # pages before posts/tax/other
            seen = set()
            for sm in maps:
                if len(discovered) >= 120: break
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
    trimmed = discovered[:120]
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
            timeout = aiohttp.ClientTimeout(total=60, sock_connect=5, sock_read=8)
            conn = aiohttp.TCPConnector(limit=16, ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=conn, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}) as session:
                while frontier and len(out) < 120:
                    url, depth = frontier.pop(0)
                    try:
                        async with session.get(url, allow_redirects=True) as resp:
                            if resp.status >= 400: continue
                            ctype = resp.headers.get("Content-Type","").lower()
                            if "html" not in ctype and "text" not in ctype: continue
                            b = await resp.content.read(350_000)
                            html = b.decode(errors="ignore")
                    except Exception:
                        continue
                    out.append(url)
                    if depth < 2 and len(out) < 120:
                        for link in _extract_links(html, url):
                            if link not in seen and ok(link):
                                seen.add(link)
                                frontier.append((link, depth+1))
            return out
        try:
            return asyncio.run(_run())[:120]
        except RuntimeError:
            if not requests: return [base]
            sess = requests.Session()
            try:
                while frontier and len(out) < 120:
                    url, depth = frontier.pop(0)
                    try:
                        r = sess.get(url, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}, timeout=(5,8), allow_redirects=True)
                        if r.status_code >= 400: continue
                        ctype = r.headers.get("Content-Type","").lower()
                        if "html" not in ctype and "text" not in ctype: continue
                        html = r.content[:350_000].decode(r.apparent_encoding or "utf-8", errors="ignore")
                    except Exception:
                        continue
                    out.append(url)
                    if depth < 2 and len(out) < 120:
                        for link in _extract_links(html, url):
                            if link not in seen and ok(link):
                                seen.add(link)
                                frontier.append((link, depth+1))
            finally:
                sess.close()
            return out[:120]
    else:
        if not requests: return [base]
        sess = requests.Session()
        try:
            while frontier and len(out) < 120:
                url, depth = frontier.pop(0)
                try:
                    r = sess.get(url, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}, timeout=(5,8), allow_redirects=True)
                    if r.status_code >= 400: continue
                    ctype = r.headers.get("Content-Type","").lower()
                    if "html" not in ctype and "text" not in ctype: continue
                    html = r.content[:350_000].decode(r.apparent_encoding or "utf-8", errors="ignore")
                except Exception:
                    continue
                out.append(url)
                if depth < 2 and len(out) < 120:
                    for link in _extract_links(html, url):
                        if link not in seen and ok(link):
                            seen.add(link)
                            frontier.append((link, depth+1))
        finally:
            sess.close()
        return out[:120]

# ---------- Nav harvesting ----------
def _harvest_nav(base_url: str) -> Tuple[Dict[str, Set[str]], Set[str]]:
    home = _normalize_base(base_url)
    html = ""
    with _session() as sess:
        html = _fetch_text_requests(home, sess, (5,8)) or ""
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

    # Build normalized token weights (legacy; we'll use keys only)
    weights: Dict[str,float] = defaultdict(float)
    for tok in _ntokens(title): weights[tok] += 1.0
    for t in h1_texts + h2h3_texts:
        for tok in _ntokens(t): weights[tok] += 1.0
    for tok in _ntokens(body_text): weights[tok] += 1.0

    title_h1 = " ".join([title] + h1_texts)
    title_h1_norm = " ".join(_ntokens(title_h1))
    lead_norm = " ".join(_ntokens(lead_text))

    veo_ready = False
    if _PHONE_PAT.search(html or "") or _ADDR_PAT.search(html or ""): veo_ready = True
    if re.search(r"LocalBusiness|Organization", html or "", re.I): veo_ready = True

    # Concatenated text for exact-phrase checks (lowercased)
    text_concat = " ".join([
        (title or "").lower(),
        " ".join([t.lower() for t in h1_texts]),
        " ".join([t.lower() for t in h2h3_texts]),
        (lead_text or "").lower(),
    ])

    return {
        "url": canonical or final_url,
        "title": title or "",
        "title_h1": title_h1.lower(),
        "title_h1_norm": title_h1_norm,
        "lead_norm": lead_norm,
        "weights": dict(weights),  # we will treat keys as unweighted tokens
        "final_url": final_url,
        "requested_url": requested_url or final_url,
        "veo_ready": veo_ready,
        "text_concat": text_concat,  # NEW: phrase checks
    }

def _fetch_profiles(urls: List[str]) -> List[Dict]:
    profiles: List[Dict] = []
    if not urls: return profiles

    if HAVE_AIOHTTP:
        async def _run():
            timeout = aiohttp.ClientTimeout(total=60, sock_connect=5, sock_read=8)
            conn = aiohttp.TCPConnector(limit=16, ssl=False)
            async with aiohttp.ClientSession(timeout=timeout, connector=conn, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}) as session:
                sem = asyncio.Semaphore(16)
                async def fetch(u: str):
                    async with sem:
                        try:
                            async with session.get(u, allow_redirects=True) as resp:
                                if resp.status >= 400: return None
                                ctype = resp.headers.get("Content-Type","").lower()
                                if "html" not in ctype and "text" not in ctype: return None
                                b = await resp.content.read(350_000)
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
                for u in urls[:120]:
                    try:
                        r = sess.get(u, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}, timeout=(5,8), allow_redirects=True)
                        if r.status_code >= 400: continue
                        ctype = r.headers.get("Content-Type","").lower()
                        if "html" not in ctype and "text" not in ctype: continue
                        html = r.content[:350_000].decode(r.apparent_encoding or "utf-8", errors="ignore")
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
            for u in urls[:120]:
                try:
                    r = sess.get(u, headers={"User-Agent":"OutrankIQMapper/1.2 (voice-engine-optimization)"}, timeout=(5,8), allow_redirects=True)
                    if r.status_code >= 400: continue
                    ctype = r.headers.get("Content-Type","").lower()
                    if "html" not in ctype and "text" not in ctype: continue
                    html = r.content[:350_000].decode(r.apparent_encoding or "utf-8", errors="ignore")
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

# ---------- Mapping (UNWEIGHTED overlap with exact-phrase precedence) ----------
def map_keywords_to_urls(df: pd.DataFrame, kw_col: Optional[str], vol_col: str, kd_col: str,
                         base_url: str, include_subdomains: bool, use_sitemap_first: bool) -> pd.Series:
    url_list, srcmap, _smeta = cached_discover_and_sources(base_url, include_subdomains, use_sitemap_first)
    profiles = cached_fetch_profiles(tuple(url_list))

    # Build per-profile token sets (unweighted union) + helpers
    nav_anchor_map, _ = cached_nav(base_url)
    nav_keys = set(nav_anchor_map.keys())

    def _is_nav(purl: str) -> bool:
        k = _url_key(purl)
        return k in nav_keys

    page_tokens: List[Set[str]] = []
    page_urls: List[str] = []
    page_src_types: List[str] = []
    page_texts: List[str] = []
    page_depths: List[int] = []
    for p in profiles:
        u = p.get("url") or p.get("final_url") or p.get("requested_url")
        if not u: 
            continue
        tokens = set()
        # union of everything we know (unweighted)
        tokens |= set((p.get("title_h1_norm") or "").split())
        tokens |= set((p.get("lead_norm") or "").split())
        tokens |= set(p.get("weights", {}).keys())
        tokens |= _slug_tokens(u)

        if not tokens:
            continue

        page_urls.append(u)
        page_tokens.append(tokens)
        page_src_types.append(srcmap.get(_url_key(u), "other"))
        page_texts.append((p.get("text_concat") or "").lower() + " " + u.lower().replace('-', ' ').replace('_', ' ').replace('/', ' '))
        page_depths.append(len([seg for seg in urlparse(u).path.split('/') if seg]))

    n_pages = len(page_urls)
    if n_pages == 0:
        return pd.Series([""]*len(df), index=df.index, dtype="string")

    # Build inverted index token -> set(page indices)
    inv: Dict[str, Set[int]] = defaultdict(set)
    for i, toks in enumerate(page_tokens):
        for t in toks:
            inv[t].add(i)

    # slot detection
    def kw_slot_for(text: str) -> str:
        cats = set(categorize_keyword(text))
        if "VEO" in cats: return "VEO"
        if "AIO" in cats: return "AIO"
        return "SEO"

    # Opportunity ordering (per strategy; same as before)
    vols_local = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).clip(lower=0)
    max_log_local = float((vols_local + 1).apply(lambda x: math.log(1 + x)).max()) or 1.0

    def opportunity(idx: int) -> float:
        row = df.loc[idx]
        kd_val = float(pd.to_numeric(row.get(kd_col,0), errors="coerce") or 0)
        vol_val = float(pd.to_numeric(row.get(vol_col,0), errors="coerce") or 0)
        kd_norm = max(0.0, 1.0 - kd_val/100.0)
        vol_norm = math.log(1 + max(0.0, vol_val)) / max_log_local
        return vol_norm * kd_norm

    # Caps per URL (per current strategy run)
    caps = {"VEO":1, "AIO":1, "SEO":2}
    per_url_caps: Dict[str, Dict[str, List[int] or Optional[int]]] = {}
    for u in page_urls:
        per_url_caps[u] = {"VEO": None, "AIO": None, "SEO": []}

    mapped = {i:"" for i in df.index}

    # Prepare eligible ids by slot, sorted by opportunity (desc)
    ids_by_slot: Dict[str, List[int]] = {"VEO": [], "AIO": [], "SEO": []}
    kw_texts: Dict[int, str] = {}
    for idx, row in df.iterrows():
        vol_val = float(pd.to_numeric(row.get(vol_col,0), errors="coerce") or 0)
        if vol_val < MIN_VALID_VOLUME:
            continue  # ineligible for mapping under current strategy
        kw_text = str(row.get(kw_col, "")) if kw_col else str(row.get("Keyword",""))
        kw_texts[idx] = kw_text
        ids_by_slot[kw_slot_for(kw_text)].append(idx)

    for slot_name in ["VEO","AIO","SEO"]:
        ids = ids_by_slot[slot_name]
        ids.sort(key=lambda i: (-opportunity(i), i))

        for i in ids:
            kw_text = kw_texts.get(i, "")
            kw_tokens = set(_ntokens(kw_text))
            if not kw_tokens:
                continue

            # candidate pages via inverted index
            candidates: Set[int] = set()
            for t in kw_tokens:
                candidates |= inv.get(t, set())
            if not candidates:
                continue

            # First: exact-phrase precedence
            kw_lower = kw_text.strip().lower()
            exact_hits = []
            for pi in candidates:
                if kw_lower and (kw_lower in page_texts[pi]):
                    exact_hits.append(pi)

            def tie_key(pi: int) -> Tuple:
                # Deterministic tie-breaks:
                # 1) shallower depth
                # 2) page-like over post-like
                # 3) shorter URL
                # 4) alphabetical URL
                stype = srcmap.get(_url_key(page_urls[pi]), "other")
                is_nav_flag = _is_nav(page_urls[pi])
                page_like = _is_page_like(stype, page_urls[pi], is_nav_flag)
                return (
                    page_depths[pi],                # smaller is better
                    0 if page_like else 1,          # page-like first
                    len(page_urls[pi]),             # shorter is better
                    page_urls[pi]                   # alphabetical
                )

            chosen_index: Optional[int] = None

            if exact_hits:
                # exact phrase always wins; break ties deterministically
                exact_hits.sort(key=lambda pi: tie_key(pi))
                chosen_index = exact_hits[0]
            else:
                # No exact hit — use unweighted coverage, then overlap, then tie-break
                scored: List[Tuple[float,int,int,int]] = []
                for pi in candidates:
                    inter = kw_tokens & page_tokens[pi]
                    if not inter:
                        continue
                    coverage = len(inter) / max(1, len(kw_tokens))
                    overlap = len(inter)
                    scored.append((coverage, overlap, pi, 0))
                if not scored:
                    continue
                scored.sort(key=lambda x: (-x[0], -x[1], tie_key(x[2])))
                chosen_index = scored[0][2]

            if chosen_index is None:
                continue

            u = page_urls[chosen_index]
            # enforce per-URL caps (soft; try next candidate if cap full)
            if slot_name in {"VEO","AIO"}:
                already = per_url_caps[u][slot_name]
                if already is None:
                    per_url_caps[u][slot_name] = i
                    mapped[i] = u
                else:
                    # try next best candidate that isn't capped out
                    # build ordered list again and pick next
                    ordered_candidates = []
                    if exact_hits:
                        ordered_candidates = sorted(exact_hits, key=lambda pi: tie_key(pi))
                    else:
                        ordered_candidates = [pi for _,_,pi,_ in sorted(scored, key=lambda x: (-x[0], -x[1], tie_key(x[2])))]
                    placed = False
                    for pi in ordered_candidates:
                        u2 = page_urls[pi]
                        if per_url_caps[u2][slot_name] is None:
                            per_url_caps[u2][slot_name] = i
                            mapped[i] = u2
                            placed = True
                            break
                    if not placed:
                        # soft overflow: allow replacing only if same URL holds nobody? (No)
                        # Leave unmapped to respect caps strictly
                        pass
            else:
                # SEO list up to 2
                current_list = per_url_caps[u]["SEO"]
                assert isinstance(current_list, list)
                if len(current_list) < 2:
                    current_list.append(i)
                    mapped[i] = u
                else:
                    # try next candidate
                    ordered_candidates = []
                    if exact_hits:
                        ordered_candidates = sorted(exact_hits, key=lambda pi: tie_key(pi))
                    else:
                        ordered_candidates = [pi for _,_,pi,_ in sorted(scored, key=lambda x: (-x[0], -x[1], tie_key(x[2])))]
                    placed = False
                    for pi in ordered_candidates:
                        u2 = page_urls[pi]
                        lst = per_url_caps[u2]["SEO"]
                        assert isinstance(lst, list)
                        if len(lst) < 2:
                            lst.append(i)
                            mapped[i] = u2
                            placed = True
                            break
                    if not placed:
                        # Strict caps: leave unmapped
                        pass

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

# Mapping controls
base_site_url = st.text_input("Base site URL (for URL mapping)", placeholder="https://example.com")
include_subdomains = True
use_sitemap_first = True  # always use sitemap first

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
        # Clean numbers
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\s]","",regex=True).str.replace("%","",regex=False)
        df[kd_col]  = df[kd_col].astype(str).str.replace(r"[,\s]","",regex=True).str.replace("%","",regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col]  = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

        scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

        # ---------- Build export_df ----------
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score","Tier","Eligible","Reason","Category"]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes":1,"No":0}).fillna(0)
        export_df = export_df.sort_values(by=["_EligibleSort", kd_col, vol_col], ascending=[False, True, False], kind="mergesort").drop(columns=["_EligibleSort"])

        # ---------- Prepare signature for mapping state ----------
        sig_cols = [c for c in [kw_col, vol_col, kd_col] if c]
        try:
            sig_df = export_df[sig_cols].copy()
        except Exception:
            sig_df = export_df[[col for col in sig_cols if col in export_df.columns]].copy()
        sig_csv = sig_df.fillna("").astype(str).to_csv(index=False)
        sig_base = f"site-map-v13-unweighted-phrase-first|{_normalize_base(base_site_url.strip()).lower()}|{scoring_mode}|{kw_col}|{vol_col}|{kd_col}|{len(export_df)}|subdomains={include_subdomains}"
        curr_signature = hashlib.md5((sig_base + "\n" + sig_csv).encode("utf-8")).hexdigest()

        # Invalidate previous map if inputs changed
        if st.session_state.get("map_signature") != curr_signature:
            st.session_state["map_ready"] = False
	            # ======================= BEGIN MAPPING BLOCK (guarded) =======================
    import re
    import pandas as pd

    # Only run if export_df exists and is a DataFrame
    if isinstance(globals().get("export_df", None), pd.DataFrame) and not export_df.empty:

        # Build case-insensitive column resolver
        cols_lower_to_orig = {c.lower(): c for c in export_df.columns}
        def _resolve(cands, allow_missing=False, default_val=None):
            for c in cands:
                if c in export_df.columns:
                    return c
                cl = str(c).lower()
                if cl in cols_lower_to_orig:
                    return cols_lower_to_orig[cl]
            if allow_missing:
                return default_val
            raise ValueError(f"Missing expected column: one of {cands}")

        KW_COL       = _resolve(["Keyword","keyword","query","term"])
        VOL_COL      = _resolve(["Search Volume","search volume","volume","sv"])
        KD_COL       = _resolve(["Keyword Difficulty","keyword difficulty","kd","difficulty"])
        ELIGIBLE_COL = _resolve(["Eligible","eligible"])
        SCORE_COL    = _resolve(["Score","score"], allow_missing=True, default_val=None)
        CATEGORY_COL = _resolve(["Category","Tags","categories","tag"], allow_missing=True, default_val=None)

        # Ensure Mapped URL column exists
        MAPPED_URL_COL = "Mapped URL"
        if MAPPED_URL_COL not in export_df.columns:
            export_df[MAPPED_URL_COL] = ""

        # Ensure debug columns exist
        for col in ["Weighted Score", "Mapping Reasons", "Slug", "Title", "H1", "Meta", "Body Preview"]:
            if col not in export_df.columns:
                export_df[col] = ""

        # Pull crawl/page signals
        page_signals_by_url = (
            st.session_state.get("url_signals")
            or st.session_state.get("crawl_signals")
            or {}
        )

        # Use new weighted mapping function from mapping.py
        results = weighted_map_keywords(export_df, page_signals_by_url)

        # Apply results back into export_df
        for res in results:
            kw = res["keyword"]
            url = res["chosen_url"] or ""
            score = res.get("weighted_score", 0)
            reasons = res.get("reasons", "")

            # Find rows that match this keyword
            matches = export_df.index[export_df["Keyword"] == kw].tolist()
            for i in matches:
                export_df.at[i, MAPPED_URL_COL] = url
                export_df.at[i, "Weighted Score"] = score
                export_df.at[i, "Mapping Reasons"] = reasons
                export_df.at[i, "Slug"] = res.get("slug_text", "")
                export_df.at[i, "Title"] = res.get("title_text", "")
                export_df.at[i, "H1"] = res.get("h1_text", "")
                export_df.at[i, "Meta"] = res.get("meta_text", "")
                export_df.at[i, "Body Preview"] = res.get("body_preview", "")

        st.session_state["map_ready"] = True



        # ---------- Manual mapping button ----------
        can_map = bool(base_site_url.strip())
        map_btn = st.button("Map keywords to site", type="primary", disabled=not can_map, help="Crawls & assigns the best page per keyword for this strategy (unweighted match; exact phrase wins).")

        if map_btn and not st.session_state.get("mapping_running", False):
            st.session_state["mapping_running"] = True
            if "map_cache" not in st.session_state:
                st.session_state["map_cache"] = {}
            loader = st.empty()
            loader.markdown(
                """
                <div class="oiq-loader">
                  <div class="oiq-spinner"></div>
                  <div class="oiq-loader-text">Mapping keywords to your site…</div>
                </div>
                """, unsafe_allow_html=True)
            with st.spinner("Crawling & matching keywords…"):
                cache = st.session_state["map_cache"]
                if curr_signature in cache and len(cache[curr_signature]) == len(export_df):
                    map_series = pd.Series(cache[curr_signature], index=export_df.index, dtype="string")
                else:
                    map_series = map_keywords_to_urls(
                        export_df, kw_col=kw_col, vol_col=vol_col, kd_col=kd_col,
                        base_url=base_site_url.strip(), include_subdomains=True, use_sitemap_first=True
                    )
                    cache[curr_signature] = map_series.fillna("").astype(str).tolist()
                st.session_state["map_result"] = map_series
                st.session_state["map_signature"] = curr_signature
                st.session_state["map_ready"] = True
            loader.empty()
            st.session_state["mapping_running"] = False

                       # ---------- Build CSV for download ----------
        if st.session_state.get("map_ready") and st.session_state.get("map_signature") == curr_signature:
            export_df["Map URL"] = st.session_state["map_result"]
            # Do not show a URL where row is not eligible
            export_df.loc[export_df["Eligible"] != "Yes", "Map URL"] = ""
            can_download = True
        else:
            export_df["Map URL"] = pd.Series([""] * len(export_df), index=export_df.index, dtype="string")
            can_download = False
            if base_site_url.strip():
                st.info("Click **Map keywords to site** to generate Map URLs for this strategy and dataset.")

        # Base columns in your original CSV
        export_cols = base_cols + ["Strategy", "Map URL"]

        # Debug columns we want to include
        debug_cols = ["Weighted Score", "Mapping Reasons", "Slug", "Title", "H1", "Meta", "Body Preview"]

        # Combine, but only keep the columns that exist in export_df
        all_cols = [c for c in (export_cols + debug_cols) if c in export_df.columns]

        # Reorder DataFrame with base first, then debug columns
        export_df = export_df.loc[:, all_cols]

        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download scored CSV",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            help="Sorted by eligibility (Yes first), KD ascending, Volume descending",
            disabled=not can_download
        )

st.markdown("<div class='oiq-footer'>© 2025 OutrankIQ</div>", unsafe_allow_html=True)
