import io
import re
import math
import asyncio
from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl, urlencode

import pandas as pd
import streamlit as st

# ---------- Optional libs ----------
try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

try:
    import aiohttp
    HAVE_AIOHTTP = True
except Exception:
    HAVE_AIOHTTP = False

try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# ================== UI SETUP ==================
st.set_page_config(page_title="OutrankIQ", page_icon="🔎", layout="centered")
st.title("OutrankIQ")
st.caption(
    "Score keywords by Search Volume (A) and Keyword Difficulty (B) — then 🚀 crawl pages "
    "(domain + subdomains) and map exactly four keywords per page (Primary, Secondary, AIO, VEO)."
)

# ---------- Helpers ----------
def find_column(df: pd.DataFrame, candidates) -> str | None:
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
    "Low Hanging Fruit": "Keywords that can be used to rank quickly with minimal effort. Ideal for new content or low-authority sites.",
    "In The Game": "Moderate difficulty keywords that are within reach for growing sites.",
    "Competitive": "High-volume, high-difficulty keywords dominated by authoritative domains.",
}

# ---------- Strategy selector ----------
scoring_mode = st.selectbox("Choose Scoring Strategy", ["Low Hanging Fruit", "In The Game", "Competitive"])

if scoring_mode == "Low Hanging Fruit":
    MIN_VALID_VOLUME = 10
    KD_BUCKETS = [(0, 15, 6), (16, 20, 5), (21, 25, 4), (26, 50, 3), (51, 75, 2), (76, 100, 1)]
elif scoring_mode == "In The Game":
    MIN_VALID_VOLUME = 1500
    KD_BUCKETS = [(0, 30, 6), (31, 45, 5), (46, 60, 4), (61, 70, 3), (71, 80, 2), (81, 100, 1)]
elif scoring_mode == "Competitive":
    MIN_VALID_VOLUME = 3000
    KD_BUCKETS = [(0, 40, 6), (41, 60, 5), (61, 75, 4), (76, 85, 3), (86, 95, 2), (96, 100, 1)]

st.markdown(
    f"""
<div style='background: linear-gradient(to right, #3b82f6, #60a5fa); padding:16px; border-radius:8px; margin-bottom:16px;'>
    <div style='margin-bottom:6px; font-size:13px; color:#ffffff;'>
        Minimum Search Volume Required: <strong>{MIN_VALID_VOLUME}</strong>
    </div>
    <strong style='color:#ffffff; font-size:18px;'>{scoring_mode}</strong><br>
    <span style='color:#ffffff; font-size:15px;'>{strategy_descriptions[scoring_mode]}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Category tagging (multi-label) ----------
CATEGORY_ORDER = ["SEO", "AIO", "VEO", "GEO", "AEO", "SXO", "LLM"]
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|is|step[- ]?by[- ]?step|tutorial|guide|overview|learn)\b", re.I)
AEO_PAT = re.compile(r"^\s*(who|what|when|where|why|how|which|can|should)\b", re.I)
VEO_PAT = re.compile(r"\b(near me|nearby|local|open now|closest|phone|call|directions|address|hours|ok google|alexa|siri|hey google)\b", re.I)
GEO_PAT = re.compile(r"\b(how to|best way to|steps? to|examples? of|checklist|framework|template)\b", re.I)
SXO_PAT = re.compile(r"\b(best|top|compare|comparison|vs\.?|review|pricing|cost|cheap|free download|template|examples?)\b", re.I)
LLM_PAT = re.compile(r"\b(prompt|prompting|prompt[- ]?engineering|chatgpt|gpt[- ]?\d|llm|rag|embedding|vector|few[- ]?shot|zero[- ]?shot)\b", re.I)

def categorize_keyword(kw: str) -> list[str]:
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

def add_scoring_columns(df: pd.DataFrame, volume_col: str, kd_col: str, kw_col: str | None) -> pd.DataFrame:
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
    kw_series = out[kw_col] if kw_col else pd.Series([""] * len(out))
    out["Category"] = [", ".join(categorize_keyword(str(k))) for k in kw_series]
    ordered = ([kw_col] if kw_col else []) + [volume_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
    remaining = [c for c in out.columns if c not in ordered]
    out = out[ordered + remaining]
    return out

# ---------- Single keyword ----------
st.subheader("Single Keyword Score")
with st.form("single_form"):
    col1, col2 = st.columns(2)
    with col1:
        vol_val = st.number_input("Search Volume (A)", min_value=0, step=10, value=0)
    with col2:
        kd_val = st.number_input("Keyword Difficulty (B)", min_value=0, step=1, value=0)
    single_submit = st.form_submit_button("Calculate Score")
    if single_submit:
        sc = calculate_score(vol_val, kd_val)
        label = LABEL_MAP.get(sc, "Not rated")
        color = COLOR_MAP.get(sc, "#9ca3af")
        if vol_val < MIN_VALID_VOLUME:
            st.warning(f"The selected strategy requires a minimum search volume of {MIN_VALID_VOLUME}.")
        st.markdown(
            f"""
            <div style='background-color:{color}; padding:16px; border-radius:8px; text-align:center;'>
                <span style='font-size:22px; font-weight:bold; color:#000;'>Score: {sc} • Tier: {label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.subheader("Bulk Scoring (CSV Upload)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

# Minimal, generic example for display only
example = pd.DataFrame({"Keyword": ["example one","example two","example three"], "Volume": [5400, 880, 12000], "KD": [38, 72, 18]})
with st.expander("See example CSV format"):
    st.dataframe(example, use_container_width=True)

# ---------- CSV reader + cleaning (persist to session) ----------
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
                kwargs = {k: v for k, v in t.items() if v is not None}
                return pd.read_csv(io.BytesIO(bytes_data), **kwargs)
            except Exception as e:
                last_err = e
        raise last_err

    try:
        df_raw = try_read(raw)
    except Exception:
        st.error("Could not read the file. Please ensure it's a CSV/TSV exported from Excel/Sheets.")
        df_raw = None

    if df_raw is not None:
        vol_col = find_column(df_raw, ["volume", "search volume", "sv"])
        kd_col  = find_column(df_raw, ["kd", "difficulty", "keyword difficulty"])
        kw_col  = find_column(df_raw, ["keyword", "query", "term"])
        missing = []
        if vol_col is None: missing.append("Volume")
        if kd_col  is None: missing.append("Keyword Difficulty")
        if kw_col  is None: missing.append("Keyword")
        if missing:
            st.error("Missing required column(s): " + ", ".join(missing))
        else:
            df_raw[vol_col] = df_raw[vol_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
            df_raw[kd_col]  = df_raw[kd_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
            df_raw[vol_col] = pd.to_numeric(df_raw[vol_col], errors="coerce")
            df_raw[kd_col]  = pd.to_numeric(df_raw[kd_col], errors="coerce").clip(lower=0, upper=100)

            st.session_state["kw_df_clean"] = df_raw
            st.session_state["kw_cols"] = {"kw": kw_col, "vol": vol_col, "kd": kd_col}

# =========================================================
# ========== Crawl + Mapping (Simple & Fast) ===============
# =========================================================
DEFAULT_MAX_PAGES_AUTODISCOVER = 5          # hidden default
DEFAULT_MAX_PASTED_URLS = 10                # cap pasted URLs
CONCURRENCY = 12
PARTIAL_MAX_BYTES = 150_000
FETCH_TIMEOUT_SECS = 4
STATIC_PREFIXES = ("cdn", "static", "img", "images", "media", "assets")

DEFAULT_HEADERS = {
    "User-Agent": "OutrankIQQuickBot/1.0 (+https://outrankiq.app)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}

# 🚀 Rocket button CSS
st.markdown("""
<style>
div[data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(90deg,#2563eb,#06b6d4) !important;
  color:#fff !important; font-weight:700 !important; border:0 !important;
  border-radius:12px !important; padding:0.6rem 1rem !important;
  box-shadow:0 8px 18px rgba(2,132,199,.35) !important;
  transition: transform .08s ease, box-shadow .2s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:before { content:"🚀 "; margin-right:.35rem; }
div[data-testid="stFormSubmitButton"] > button:hover{
  transform: translateY(-1px);
  box-shadow:0 10px 22px rgba(2,132,199,.45) !important;
}
</style>
""", unsafe_allow_html=True)

# -------- Tokenization & tiny synonyms --------
STOPWORDS = set("""
a an the and or of for to in on at by with from as is are be was were this that these those it its it's your our my we you
what when where why how which who can should could would will near nearby local open now closest call directions ok google alexa siri hey
""".split())
SPLIT_RE = re.compile(r"[^a-z0-9]+")

def tok(s: str) -> list[str]:
    if not s: return []
    return [t for t in SPLIT_RE.split(str(s).lower()) if t and t not in STOPWORDS and len(t) > 1]

def micro_stem(t: str) -> str:
    if len(t) > 4:
        for suf in ("ing","ers","ies","ment","tion","s","es","ed"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
    return t

_base_syn = {
    "guide": ["tutorial","how","howto","how-to","walkthrough","step","steps","handbook"],
    "compare": ["vs","versus","comparison","against"],
    "cheap": ["affordable","budget","lowcost","low-cost","inexpensive"],
    "pricing": ["price","cost","costs","rates","fees"],
    "review": ["reviews","rating","ratings","test","tests","overview"],
    "template": ["templates","example","examples","framework","checklist","blueprint"],
    "best": ["top","leading","greatest"],
    "buy": ["purchase","order"],
    "free": ["gratis","no-cost","complimentary"],
    "software": ["tool","platform","app","application"],
    "service": ["services","agency","consulting","consultant"],
}
from collections import defaultdict as _dd
SYN = _dd(set)
for k, arr in _base_syn.items():
    k2 = micro_stem(k)
    for v in arr:
        v2 = micro_stem(v)
        SYN[k2].add(v2)
        SYN[v2].add(k2)

def expand_kw_tokens(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for t in list(tokens):
        t2 = micro_stem(t)
        out.add(t2)
        for s in SYN.get(t2, []):
            out.add(s)
    return out

# -------- URL & fetch helpers --------
def normalize_site(u: str) -> str:
    u = (u or "").strip()
    if not u: return ""
    parsed = urlparse(u if "://" in u else "https://" + u)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path
    if not netloc: return ""
    return f"{scheme}://{netloc}"

def strip_tracking(u: str) -> str:
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
             if not (k.startswith("utm_") or k in {"gclid","fbclid","mc_cid","mc_eid","ref","referrer"})]
        clean = p._replace(query=urlencode(q), fragment="")
        path = re.sub(r"/{2,}", "/", clean.path)
        if path.endswith("/") and path != "/": path = path[:-1]
        clean = clean._replace(path=path)
        return urlunparse(clean)
    except Exception:
        return u

def is_html_like(url: str) -> bool:
    lower = url.lower()
    bad_exts = (".pdf",".jpg",".jpeg",".png",".gif",".webp",".svg",".zip",".rar",".7z",".gz",
                ".mp4",".mp3",".avi",".mov",".wmv",".doc",".docx",".xls",".xlsx",".ppt",".pptx",".ics",".csv")
    return not any(lower.endswith(ext) for ext in bad_exts)

def host_in_scope(url: str, root_host: str) -> bool:
    try:
        h = urlparse(url).netloc.lower()
        if not h: return False
        first = h.split(".")[0]
        if first in STATIC_PREFIXES: return False
        return h == root_host or h.endswith("." + root_host)
    except Exception:
        return False

def content_type_ok(ctype: str) -> bool:
    ctype = (ctype or "").lower()
    return ("text/html" in ctype) or ("xml" in ctype)

async def fetch_async_partial(session, url: str) -> str | None:
    try:
        async with session.get(url) as r:
            if r.status != 200: return None
            if not content_type_ok(r.headers.get("Content-Type", "")): return None
            b = await r.content.read(PARTIAL_MAX_BYTES)
            try:
                return b.decode("utf-8", errors="ignore")
            except Exception:
                return b.decode("latin-1", errors="ignore")
    except Exception:
        return None

def fetch_sync_partial(url: str) -> str | None:
    if not HAVE_REQUESTS: return None
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=FETCH_TIMEOUT_SECS, stream=True)
        if r.status_code != 200: return None
        if not content_type_ok(r.headers.get("Content-Type", "")): return None
        b = b""
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk: break
            b += chunk
            if len(b) >= PARTIAL_MAX_BYTES: break
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return None

# -------- sitemap discovery (for auto mode) --------
SITEMAP_RE = re.compile(r"(?i)^\s*sitemap:\s*(\S+)\s*$", re.M)

async def gather_candidates_async(base: str, root_host: str) -> list[str]:
    out = []
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECS)
    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=timeout) as session:
        robots_url = base.rstrip("/") + "/robots.txt"
        robots_txt = await fetch_async_partial(session, robots_url) or ""
        sm_urls = SITEMAP_RE.findall(robots_txt or "") + [base.rstrip("/") + "/sitemap.xml"]

        tasks = [asyncio.create_task(fetch_async_partial(session, u)) for u in dict.fromkeys(sm_urls)]
        xmls = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

        def parse_sitemap(xml: str):
            if not xml: return []
            locs = re.findall(r"<loc>(.*?)</loc>", xml, flags=re.I | re.S)
            return [strip_tracking(l.strip()) for l in locs if l.strip()]

        for xml in xmls:
            for u in parse_sitemap(xml if isinstance(xml, str) else ""):
                if len(out) >= DEFAULT_MAX_PAGES_AUTODISCOVER: break
                if is_html_like(u) and host_in_scope(u, root_host):
                    out.append(u)
    return list(dict.fromkeys(out))

def gather_candidates_sync(base: str, root_host: str) -> list[str]:
    out = []
    if HAVE_REQUESTS:
        robots_url = base.rstrip("/") + "/robots.txt"
        robots_txt = fetch_sync_partial(robots_url) or ""
        sm_urls = SITEMAP_RE.findall(robots_txt or "") + [base.rstrip("/") + "/sitemap.xml"]

        def parse_sitemap(xml: str):
            if not xml: return []
            locs = re.findall(r"<loc>(.*?)</loc>", xml, flags=re.I | re.S)
            return [strip_tracking(l.strip()) for l in locs if l.strip()]

        for sm in dict.fromkeys(sm_urls):
            xml = fetch_sync_partial(sm)
            for u in parse_sitemap(xml):
                if len(out) >= DEFAULT_MAX_PAGES_AUTODISCOVER: break
                if is_html_like(u) and host_in_scope(u, root_host):
                    out.append(u)
    return list(dict.fromkeys(out))

def extract_links(html: str, base_url: str, root_host: str) -> list[str]:
    links = []
    if not html: return links
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                url = urljoin(base_url, href)
                if is_html_like(url) and host_in_scope(url, root_host):
                    links.append(strip_tracking(url))
        except Exception:
            pass
    else:
        for href in re.findall(r'href=[\'"]?([^\'" >]+)', html):
            url = urljoin(base_url, href.strip())
            if is_html_like(url) and host_in_scope(url, root_host):
                links.append(strip_tracking(url))

    def score(u: str) -> int:
        p = urlparse(u)
        path = (p.path or "/").lower()
        s = 0
        if "-" in path: s += 2
        for token in ("blog","post","article","guide","docs","learn","case","study","news"):
            if f"/{token}/" in path: s += 2
        if path.count("/") <= 3: s += 1
        if len(path) <= 80: s += 1
        return s
    links = list(dict.fromkeys(links))
    links.sort(key=score, reverse=True)
    return links[:DEFAULT_MAX_PAGES_AUTODISCOVER]

def collect_pages_auto(site: str) -> dict[str, str]:
    base = normalize_site(site)
    if not base: return {}
    root_host = urlparse(base).netloc.lower()

    urls = []
    if HAVE_AIOHTTP:
        try:
            urls = asyncio.run(gather_candidates_async(base, root_host))
        except RuntimeError:
            urls = gather_candidates_sync(base, root_host)
    else:
        urls = gather_candidates_sync(base, root_host)

    urls = [u for u in urls if host_in_scope(u, root_host)]
    urls = list(dict.fromkeys(urls))
    urls = [base] + [u for u in urls if u != base]
    urls = urls[:DEFAULT_MAX_PAGES_AUTODISCOVER]

    async def fetch_many_async(cands: list[str]) -> dict[str,str]:
        out = {}
        timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECS)
        sem = asyncio.Semaphore(CONCURRENCY)
        async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=timeout) as session:
            async def go(u: str):
                async with sem:
                    h = await fetch_async_partial(session, u)
                if h: out[u] = h
            tasks = [asyncio.create_task(go(u)) for u in cands]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        return out

    def fetch_many_sync(cands: list[str]) -> dict[str,str]:
        out = {}
        for u in cands:
            h = fetch_sync_partial(u)
            if h: out[u] = h
            if len(out) >= DEFAULT_MAX_PAGES_AUTODISCOVER: break
        return out

    cands = urls[:DEFAULT_MAX_PAGES_AUTODISCOVER]
    if HAVE_AIOHTTP:
        try:
            html_map = asyncio.run(fetch_many_async(cands))
        except RuntimeError:
            html_map = fetch_many_sync(cands)
    else:
        html_map = fetch_many_sync(cands)

    if len(html_map) < DEFAULT_MAX_PAGES_AUTODISCOVER and base not in html_map:
        home_html = fetch_sync_partial(base)
        if home_html:
            html_map[base] = home_html
    if len(html_map) < DEFAULT_MAX_PAGES_AUTODISCOVER and base in html_map:
        more = extract_links(html_map[base], base, root_host)
        need = DEFAULT_MAX_PAGES_AUTODISCOVER - len(html_map)
        more = [u for u in more if u not in html_map][:need]
        if more:
            if HAVE_AIOHTTP:
                try:
                    extra = asyncio.run(fetch_many_async(more))
                except RuntimeError:
                    extra = fetch_many_sync(more)
            else:
                extra = fetch_many_sync(more)
            html_map.update(extra)

    return dict(list(html_map.items())[:DEFAULT_MAX_PAGES_AUTODISCOVER])

def collect_pages_from_list(urls: list[str]) -> dict[str, str]:
    urls = [strip_tracking(u if "://" in u else "https://" + u.strip()) for u in urls if u.strip()]
    urls = list(dict.fromkeys(urls))[:DEFAULT_MAX_PASTED_URLS]

    async def fetch_many_async(cands: list[str]) -> dict[str,str]:
        out = {}
        timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SECS)
        sem = asyncio.Semaphore(CONCURRENCY)
        async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=timeout) as session:
            async def go(u: str):
                async with sem:
                    h = await fetch_async_partial(session, u)
                if h: out[u] = h
            tasks = [asyncio.create_task(go(u)) for u in cands]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        return out

    def fetch_many_sync(cands: list[str]) -> dict[str,str]:
        out = {}
        for u in cands:
            h = fetch_sync_partial(u)
            if h: out[u] = h
        return out

    if HAVE_AIOHTTP:
        try:
            html_map = asyncio.run(fetch_many_async(urls))
        except RuntimeError:
            html_map = fetch_many_sync(urls)
    else:
        html_map = fetch_many_sync(urls)
    return html_map

# -------- page vector (ONLY url/slug, title, h1-h3, meta) --------
def extract_text_tag(html: str, tag: str) -> list[str]:
    results = []
    if not html: return results
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for el in soup.find_all(tag):
                txt = (el.get_text(" ", strip=True) or "").strip()
                if txt: results.append(txt)
        except Exception:
            pass
    else:
        pattern = re.compile(fr"<{tag}[^>]*>(.*?)</{tag}>", re.I | re.S)
        for m in pattern.findall(html):
            t = re.sub(r"<[^>]+>", " ", m)
            t = re.sub(r"\s+", " ", t).strip()
            if t: results.append(t)
    return results

def extract_meta_desc(html: str) -> str:
    if not html: return ""
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            m = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            return (m.get("content") or "").strip() if m else ""
        except Exception:
            return ""
    m = re.search(r'<meta[^>]+name=["\']description["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I | re.S)
    return m.group(1).strip() if m else ""

def slug_tokens(url: str) -> list[str]:
    try:
        path = urlparse(url).path or "/"
        parts = [p for p in re.split(r"[\/\-_]+", path) if p]
        toks = []
        for p in parts:
            toks.extend(tok(p))
        return toks
    except Exception:
        return []

def page_vector(url: str, html: str):
    title_txt = " ".join(extract_text_tag(html, "title"))
    h1_txt   = " ".join(extract_text_tag(html, "h1"))
    h2_txt   = " ".join(extract_text_tag(html, "h2"))
    h3_txt   = " ".join(extract_text_tag(html, "h3"))
    meta_txt = extract_meta_desc(html)
    slug_toks = slug_tokens(url)

    weights = defaultdict(float)
    for t in tok(title_txt): weights[t] += 3.0
    for t in tok(h1_txt):    weights[t] += 2.0
    for t in slug_toks:      weights[t] += 2.5
    for t in tok(h2_txt):    weights[t] += 1.2
    for t in tok(h3_txt):    weights[t] += 1.0
    for t in tok(meta_txt):  weights[t] += 0.8

    norm = math.sqrt(sum(w*w for w in weights.values())) if weights else 0.0
    return dict(weights), norm

# -------- similarity + page-topics --------
SPLIT_RE2 = re.compile(r"[^a-z0-9]+")

def base_kw_page_score(kw_text: str, page_url: str, page_vec: dict[str,float], page_norm: float) -> float:
    tokens = [t for t in SPLIT_RE2.split(kw_text.lower()) if t and t not in STOPWORDS]
    if not tokens or not page_vec:
        return 0.0
    kw_tokens = set(tokens)
    kw_tokens_exp = expand_kw_tokens(kw_tokens)
    overlap = sum(page_vec.get(t, 0.0) for t in kw_tokens_exp)
    score = overlap / max(page_norm, 1e-9)

    slug_hit = any(t in slug_tokens(page_url) for t in kw_tokens)
    if slug_hit:
        score += 0.6
    if any(k in page_url.lower() for k in kw_tokens):
        score += 0.3
    return score

def top_page_terms(vec: dict[str,float], n: int = 6) -> list[str]:
    if not vec: return []
    tops = sorted(vec.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [t for t, _ in tops]

def is_aio_kw(cat: str) -> bool:
    return isinstance(cat, str) and ("AIO" in cat)

def is_veo_kw(cat: str) -> bool:
    return isinstance(cat, str) and ("VEO" in cat)

# -------- assignment: exactly 4 per page, unique across site --------
def assign_keywords_page_first(pages_html: dict[str, str], export_df: pd.DataFrame, kw_col: str, vol_col: str):
    """
    Returns:
      mapped_by_index: {row_index -> mapped_url}
      page_summary: [{url, topics, picks: [kw1,kw2,kw3,kw4]}]
    """
    mapped = {}
    page_summary = []

    if not pages_html or kw_col is None:
        return mapped, page_summary

    # Precompute page vectors and topics
    page_profiles = []
    for u, html in pages_html.items():
        vec, norm = page_vector(u, html)
        if norm > 0:
            page_profiles.append((u, vec, norm, top_page_terms(vec)))

    if not page_profiles:
        return mapped, page_summary

    # Candidate pool = ALL keywords (no gating)
    idx = export_df.index
    kw_texts = export_df[kw_col].astype(str)
    kw_scores = pd.to_numeric(export_df["Score"], errors="coerce").fillna(0).astype(float)
    kw_vols = pd.to_numeric(export_df[vol_col], errors="coerce").fillna(0).astype(float)
    kw_cats = export_df["Category"].fillna("")

    used = set()

    def tie_key(i):
        return (-kw_scores.loc[i], -kw_vols.loc[i], str(kw_texts.loc[i]).lower())

    for (url, vec, norm, topics) in page_profiles:
        # Rank all unused keywords by similarity
        sims = {}
        for i in idx:
            if i in used:
                continue
            sims[i] = base_kw_page_score(kw_texts.loc[i], url, vec, norm)
        if not sims:
            page_summary.append({"url": url, "topics": topics, "picks": []})
            continue

        ranked = sorted(sims.keys(), key=lambda i: (-sims[i],) + tie_key(i))

        def pick_next(pred=None):
            for i in ranked:
                if i in used: 
                    continue
                if (pred is None) or pred(i):
                    used.add(i)
                    mapped[i] = url
                    return i
            return None

        # Primary, Secondary
        p_idx = pick_next()
        s_idx = pick_next()

        # AIO (prefer AIO-tagged; else best remaining)
        a_idx = pick_next(lambda i: is_aio_kw(kw_cats.loc[i])) or pick_next()

        # VEO (prefer VEO-tagged; else best remaining)
        v_idx = pick_next(lambda i: is_veo_kw(kw_cats.loc[i])) or pick_next()

        picks = [x for x in [p_idx, s_idx, a_idx, v_idx] if x is not None]
        page_summary.append({
            "url": url,
            "topics": topics,
            "picks": [kw_texts.loc[i] for i in picks]
        })

    return mapped, page_summary

# =========================================================
# ================== Quick-Map Form =======================
# =========================================================
st.markdown("---")
st.subheader("Site Quick-Map")

with st.form("quickmap_form"):
    colA, colB = st.columns([1,1])
    with colA:
        site_url = st.text_input("Main domain (subdomains included automatically)", placeholder="https://example.com")
    with colB:
        pasted = st.text_area("Optional: paste up to 10 URLs (one per line). If empty, we’ll auto-discover.", height=120, placeholder="https://example.com/\nhttps://blog.example.com/post/...")

    map_submit = st.form_submit_button("score, crawl, and map")

download_area = st.empty()

if map_submit:
    df_clean = st.session_state.get("kw_df_clean", None)
    cols = st.session_state.get("kw_cols", {})
    if df_clean is None:
        st.error("Please upload your keyword CSV first.")
    elif not (site_url or pasted).strip():
        st.error("Enter a domain or paste URLs.")
    else:
        kw_col = cols.get("kw"); vol_col = cols.get("vol"); kd_col = cols.get("kd")
        scored = add_scoring_columns(df_clean, vol_col, kd_col, kw_col)

        with st.spinner("Crawling & mapping…"):
            html_map = {}
            if pasted.strip():
                url_list = [line.strip() for line in pasted.splitlines() if line.strip()]
                html_map = collect_pages_from_list(url_list)
            else:
                html_map = collect_pages_auto(site_url)

        # Prepare export (stable sort)
        filename_base = f"outrankiq_map_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes": 1, "No": 0}).fillna(0)
        export_df = export_df.sort_values(
            by=["_EligibleSort", kd_col, vol_col],
            ascending=[False, True, False],
            kind="mergesort"
        ).drop(columns=["_EligibleSort"])

        # Do the mapping
        mapped_by_index, page_summary = assign_keywords_page_first(html_map, export_df, kw_col, vol_col)

        # Add Mapped URL column (blank for unassigned)
        export_df["Mapped URL"] = [mapped_by_index.get(idx, "") for idx in export_df.index]

        # Persist CSV so it doesn't disappear
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.session_state["last_csv_bytes"] = csv_bytes
        st.session_state["last_csv_name"] = f"{filename_base}.csv"

        # Show a concise summary of what got mapped (URL + topics + 4 picks)
        if page_summary:
            st.success(f"Mapped keywords to {sum(1 for s in page_summary if s['picks'])} of {len(page_summary)} pages.")
            with st.expander("See per-page topics & selected keywords"):
                for s in page_summary:
                    st.markdown(f"**{s['url']}**")
                    topics = ", ".join(s.get("topics") or [])
                    st.write(f"Page topics: {topics if topics else '—'}")
                    picks = s.get("picks") or []
                    if picks:
                        st.write("Selected keywords: " + "; ".join(picks[:4]))
                    else:
                        st.write("No keywords selected.")
                    st.markdown("---")

# Persistent download button
if "last_csv_bytes" in st.session_state and "last_csv_name" in st.session_state:
    with download_area:
        st.download_button(
            label="⬇️ Download scored + mapped CSV",
            data=st.session_state["last_csv_bytes"],
            file_name=st.session_state["last_csv_name"],
            mime="text/csv",
            help="Each page gets up to 4 keywords (Primary, Secondary, AIO-preferred, VEO-preferred). Unassigned rows are left blank.",
            key="dl_quickmap"
        )

# Optional preview
if "last_csv_bytes" in st.session_state and st.checkbox("Preview first 10 rows (optional)", value=False, key="preview_quick"):
    try:
        df_preview = pd.read_csv(io.BytesIO(st.session_state["last_csv_bytes"]))
        def _row_style(row):
            color = COLOR_MAP.get(int(row.get("Score", 0)) if pd.notna(row.get("Score", 0)) else 0, "#9ca3af")
            return [("background-color:" + color + "; color:black;") if c in ("Score","Tier") else "" for c in row.index]
        styled = df_preview.head(10).style.apply(_row_style, axis=1)
        st.dataframe(styled, use_container_width=True)
    except Exception:
        pass

st.markdown("---")
st.caption("© 2025 OutrankIQ • Simple page-first mapping: URL/slug + title/H1–H3 + meta → 4 keywords per page (unique across site).")
