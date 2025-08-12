import io
import re
import asyncio
import math
from collections import deque, defaultdict
from datetime import datetime
from urllib.parse import urlparse, urljoin
from urllib import robotparser as _robotparser

import pandas as pd
import streamlit as st

# ---------- Optional libs ----------
try:
    from bs4 import BeautifulSoup  # noqa: F401
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

st.set_page_config(page_title="OutrankIQ", page_icon="🔎", layout="centered")

st.title("OutrankIQ")
st.caption("Score keywords by Search Volume (A) and Keyword Difficulty (B) — with selectable scoring strategies.")

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

LABEL_MAP = {
    6: "Elite",
    5: "Excellent",
    4: "Good",
    3: "Fair",
    2: "Low",
    1: "Very Low",
    0: "Not rated",
}

# Used for card + preview styling only (NOT exported)
COLOR_MAP = {
    6: "#2ecc71",  # bright green
    5: "#a3e635",  # lime
    4: "#facc15",  # yellow
    3: "#fb923c",  # orange
    2: "#f87171",  # tomato
    1: "#ef4444",  # red
    0: "#9ca3af",  # gray
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
    MIN_VALID_VOLUME = 1500  # per your correction
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
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|is|step[- ]?by[- ]?step|tutorial|guide)\b", re.I)
AEO_PAT = re.compile(r"^\s*(who|what|when|where|why|how|which|can|should)\b", re.I)
VEO_PAT = re.compile(r"\b(near me|open now|closest|call now|directions|ok google|alexa|siri|hey google)\b", re.I)
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
    """Return score 0-6, but ONLY if eligible (volume >= min)."""
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

    # Eligibility + Reason
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

    # Category (multi-label)
    kw_series = out[kw_col] if kw_col else pd.Series([""] * len(out))
    out["Category"] = [", ".join(categorize_keyword(str(k))) for k in kw_series]

    # Order columns (no color column shown)
    ordered = ([kw_col] if kw_col else []) + [volume_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
    remaining = [c for c in out.columns if c not in ordered]
    out = out[ordered + remaining]
    return out

# ---------- Single keyword ----------
st.subheader("Single Keyword Score")
with st.form("single"):
    col1, col2 = st.columns(2)
    with col1:
        vol_val = st.number_input("Search Volume (A)", min_value=0, step=10, value=0)
    with col2:
        kd_val = st.number_input("Keyword Difficulty (B)", min_value=0, step=1, value=0)

    if st.form_submit_button("Calculate Score"):
        sc = calculate_score(vol_val, kd_val)
        label = LABEL_MAP.get(sc, "Not rated")
        color = COLOR_MAP.get(sc, "#9ca3af")
        if vol_val < MIN_VALID_VOLUME:
            st.warning(f"The selected strategy requires a minimum search volume of {MIN_VALID_VOLUME}. Please enter a volume that meets the threshold.")
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
example = pd.DataFrame(
    {"Keyword": ["best running shoes", "seo tools", "crm software"], "Volume": [5400, 880, 12000], "KD": [38, 72, 18]}
)
with st.expander("See example CSV format"):
    st.dataframe(example, use_container_width=True)

# ---------- Robust CSV reader + numeric cleaning ----------
df = None
kw_col = vol_col = kd_col = None
if uploaded is not None:
    raw = uploaded.getvalue()

    def try_read(bytes_data: bytes) -> pd.DataFrame:
        trials = [
            {"encoding": None, "sep": None, "engine": "python"},  # let pandas infer
            {"encoding": "utf-8", "sep": None, "engine": "python"},
            {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
            {"encoding": "ISO-8859-1", "sep": None, "engine": "python"},
            {"encoding": "cp1252", "sep": None, "engine": "python"},
            {"encoding": "utf-16", "sep": None, "engine": "python"},
            {"encoding": None, "sep": ",", "engine": "python"},   # force comma
            {"encoding": None, "sep": "\t", "engine": "python"},  # TSV fallback
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
        df = try_read(raw)
    except Exception:
        st.error("Could not read the file. Please ensure it's a CSV (or TSV) exported from Excel/Sheets and try again.")
        st.stop()

    # Find relevant columns
    vol_col = find_column(df, ["volume", "search volume", "sv"])
    kd_col = find_column(df, ["kd", "difficulty", "keyword difficulty"])
    kw_col = find_column(df, ["keyword", "query", "term"])

    missing = []
    if vol_col is None:
        missing.append("Volume")
    if kd_col is None:
        missing.append("Keyword Difficulty")
    if kw_col is None:
        missing.append("Keyword")

    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
        df = None
    else:
        # Clean numbers (commas, spaces, percents)
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[kd_col] = df[kd_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

# =========================================================
# ========== Crawl + Mapping UI (single action) ===========
# =========================================================
st.markdown("---")
st.subheader("Site Crawl & Keyword Mapping")

with st.form("crawlmap"):
    site_url = st.text_input("Site to crawl (domain or full URL)", placeholder="https://example.com")
    include_subdomains = st.checkbox("Include subdomains", value=True)
    max_pages = st.number_input("Max pages", min_value=1, max_value=10000, value=500, step=50)
    only_assign_eligible = st.checkbox("Only assign eligible keywords", value=True)
    submit = st.form_submit_button(
        "Score, Crawl, and Map",
        disabled=(df is None or not (site_url or "").strip())
    )

# =========================================================
# ============== Crawl + Mapping Implementation ===========
# =========================================================
# ---- Utility: domain + scope checks ----
def _normalize_site(u: str) -> str:
    u = u.strip()
    if not u:
        return ""
    parsed = urlparse(u if "://" in u else "https://" + u)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path
    if not netloc:
        return ""
    return f"{scheme}://{netloc}"

def _base_hostparts(host: str) -> list[str]:
    return host.split(".")

def _in_scope(url: str, root_host: str, include_subs: bool) -> bool:
    try:
        h = urlparse(url).netloc.lower()
        if include_subs:
            return h == root_host or h.endswith("." + root_host)
        return h == root_host
    except Exception:
        return False

def _is_html_like(url: str) -> bool:
    lower = url.lower()
    bad_exts = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip",
                ".rar", ".7z", ".gz", ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".doc", ".docx", ".xls", ".xlsx")
    return not any(lower.endswith(ext) for ext in bad_exts)

# ---- robots.txt (silent; always respected) ----
def _load_robots(base: str) -> _robotparser.RobotFileParser:
    rp = _robotparser.RobotFileParser()
    try:
        rp.set_url(base.rstrip("/") + "/robots.txt")
        rp.read()
    except Exception:
        # If robots fails to load, default allow False -> be conservative and still check can_fetch
        pass
    return rp

# ---- Fetchers ----
_DEFAULT_HEADERS = {
    "User-Agent": "OutrankIQBot/1.0 (+https://outrankiq.local)"
}
FETCH_TIMEOUT = 12

async def _fetch_async(session: aiohttp.ClientSession, url: str) -> str | None:
    try:
        async with session.get(url, timeout=FETCH_TIMEOUT) as r:
            if r.status != 200:
                return None
            ctype = r.headers.get("Content-Type", "")
            if "text/html" not in ctype and "xml" not in ctype:
                return None
            return await r.text(errors="ignore")
    except Exception:
        return None

def _fetch_sync(url: str) -> str | None:
    if not HAVE_REQUESTS:
        return None
    try:
        r = requests.get(url, headers=_DEFAULT_HEADERS, timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("Content-Type", "")
        if "text/html" not in ctype and "xml" not in ctype:
            return None
        return r.text
    except Exception:
        return None

# ---- Sitemap discovery ----
_SITEMAP_RE = re.compile(r"(?i)^\s*sitemap:\s*(\S+)\s*$", re.M)

def _extract_sitemaps_from_robots(robots_txt: str) -> list[str]:
    return _SITEMAP_RE.findall(robots_txt or "") if robots_txt else []

def _parse_sitemap_xml(xml_text: str) -> list[str]:
    # Very light parser for <loc>...</loc>
    if not xml_text:
        return []
    locs = re.findall(r"<loc>(.*?)</loc>", xml_text, flags=re.I | re.S)
    # also handle self-closing tags or with namespaces
    if not locs:
        locs = re.findall(r"<\s*loc\s*>\s*(.*?)\s*<\s*/\s*loc\s*>", xml_text, flags=re.I | re.S)
    return [l.strip() for l in locs if l.strip()]

async def _gather_sitemap_urls_async(base: str, include_subs: bool, max_pages: int) -> set[str]:
    urls: set[str] = set()
    root_host = urlparse(base).netloc.lower()
    robots_url = base.rstrip("/") + "/robots.txt"
    async with aiohttp.ClientSession(headers=_DEFAULT_HEADERS) as session:
        robots_txt = await _fetch_async(session, robots_url) or ""
        sitemap_urls = _extract_sitemaps_from_robots(robots_txt)
        # Always try /sitemap.xml too
        sitemap_urls.append(base.rstrip("/") + "/sitemap.xml")
        for sm in list(dict.fromkeys(sitemap_urls)):
            xml = await _fetch_async(session, sm)
            if not xml:
                continue
            for u in _parse_sitemap_xml(xml):
                if _is_html_like(u) and _in_scope(u, root_host, include_subs):
                    urls.add(u)
                if len(urls) >= max_pages:
                    break
            if len(urls) >= max_pages:
                break
    return urls

def _gather_sitemap_urls_sync(base: str, include_subs: bool, max_pages: int) -> set[str]:
    urls: set[str] = set()
    if not HAVE_REQUESTS:
        return urls
    root_host = urlparse(base).netloc.lower()
    robots_url = base.rstrip("/") + "/robots.txt"
    robots_txt = _fetch_sync(robots_url) or ""
    sitemap_urls = _extract_sitemaps_from_robots(robots_txt)
    sitemap_urls.append(base.rstrip("/") + "/sitemap.xml")
    for sm in list(dict.fromkeys(sitemap_urls)):
        xml = _fetch_sync(sm)
        if not xml:
            continue
        for u in _parse_sitemap_xml(xml):
            if _is_html_like(u) and _in_scope(u, root_host, include_subs):
                urls.add(u)
            if len(urls) >= max_pages:
                break
        if len(urls) >= max_pages:
            break
    return urls

# ---- Link crawling (BFS) ----
_LINK_HREF_RE = re.compile(r'href=[\'"]?([^\'" >]+)')

def _extract_links(html: str, base_url: str) -> list[str]:
    links = []
    if not html:
        return links
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                links.append(urljoin(base_url, href))
        except Exception:
            pass
    else:
        for href in _LINK_HREF_RE.findall(html):
            links.append(urljoin(base_url, href.strip()))
    return links

async def _crawl_bfs_async(base: str, include_subs: bool, max_pages: int, rp: _robotparser.RobotFileParser) -> dict[str, str]:
    result: dict[str, str] = {}
    root_host = urlparse(base).netloc.lower()
    start = base
    seen = set()
    q = deque([start])
    async with aiohttp.ClientSession(headers=_DEFAULT_HEADERS) as session:
        while q and len(result) < max_pages:
            url = q.popleft()
            if url in seen:
                continue
            seen.add(url)
            if not _is_html_like(url) or not _in_scope(url, root_host, include_subs):
                continue
            if not rp.can_fetch(_DEFAULT_HEADERS["User-Agent"], url):
                continue
            html = await _fetch_async(session, url)
            if not html:
                continue
            result[url] = html
            # enqueue links
            for lk in _extract_links(html, url):
                if (lk not in seen) and _is_html_like(lk) and _in_scope(lk, root_host, include_subs):
                    q.append(lk)
            if len(result) >= max_pages:
                break
    return result

def _crawl_bfs_sync(base: str, include_subs: bool, max_pages: int, rp: _robotparser.RobotFileParser) -> dict[str, str]:
    result: dict[str, str] = {}
    if not HAVE_REQUESTS:
        return result
    root_host = urlparse(base).netloc.lower()
    start = base
    seen = set()
    q = deque([start])
    while q and len(result) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        if not _is_html_like(url) or not _in_scope(url, root_host, include_subs):
            continue
        if not rp.can_fetch(_DEFAULT_HEADERS["User-Agent"], url):
            continue
        html = _fetch_sync(url)
        if not html:
            continue
        result[url] = html
        for lk in _extract_links(html, url):
            if (lk not in seen) and _is_html_like(lk) and _in_scope(lk, root_host, include_subs):
                q.append(lk)
        if len(result) >= max_pages:
            break
    return result

# ---- Page understanding ----
_STOPWORDS = set("""
a an the and or of for to in on at by with from as is are be was were this that these those it its it's your our my we you
what when where why how which who can should could would will near open now closest call directions ok google alexa siri hey
""".split())

TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

def _tokens(text: str) -> list[str]:
    if not text:
        return []
    toks = [t for t in TOKEN_SPLIT_RE.split(text.lower()) if t and t not in _STOPWORDS and len(t) > 1]
    return toks

def _slug_tokens(url: str) -> list[str]:
    try:
        path = urlparse(url).path or "/"
        parts = [p for p in re.split(r"[\/\-_]+", path) if p]
        toks = []
        for p in parts:
            toks.extend(_tokens(p))
        return toks
    except Exception:
        return []

def _extract_text_tag(html: str, tag: str) -> list[str]:
    results = []
    if not html:
        return results
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for el in soup.find_all(tag):
                txt = (el.get_text(" ", strip=True) or "").strip()
                if txt:
                    results.append(txt)
        except Exception:
            pass
    else:
        # light regex fallback (best-effort)
        pattern = re.compile(fr"<{tag}[^>]*>(.*?)</{tag}>", re.I | re.S)
        for m in pattern.findall(html):
            t = re.sub(r"<[^>]+>", " ", m)
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                results.append(t)
    return results

def _extract_meta_desc(html: str) -> str:
    if not html:
        return ""
    if HAVE_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            m = soup.find("meta", attrs={"name": "description"})
            if not m:
                m = soup.find("meta", attrs={"property": "og:description"})
            return (m.get("content") or "").strip() if m else ""
        except Exception:
            return ""
    # regex fallback
    m = re.search(r'<meta[^>]+name=["\']description["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]*content=["\'](.*?)["\']', html, flags=re.I | re.S)
    return m.group(1).strip() if m else ""

def _page_profile(url: str, html: str) -> dict[str, float]:
    # Weighted token bag
    title_txts = _extract_text_tag(html, "title")
    h1_txts = _extract_text_tag(html, "h1")
    h2_txts = _extract_text_tag(html, "h2")
    h3_txts = _extract_text_tag(html, "h3")
    meta_desc = _extract_meta_desc(html)
    slug = _slug_tokens(url)

    weights = defaultdict(float)
    for t in _tokens(" ".join(title_txts)): weights[t] += 3.0
    for t in _tokens(" ".join(h1_txts)): weights[t] += 2.5
    for t in slug: weights[t] += 2.0
    for t in _tokens(" ".join(h2_txts)): weights[t] += 1.5
    for t in _tokens(" ".join(h3_txts)): weights[t] += 1.2
    for t in _tokens(meta_desc): weights[t] += 1.0
    return dict(weights)

def _cosine_overlap(page_vec: dict[str, float], kw_tokens: set[str]) -> float:
    if not page_vec or not kw_tokens:
        return 0.0
    num = sum(page_vec.get(t, 0.0) for t in kw_tokens)
    denom = math.sqrt(sum(w*w for w in page_vec.values())) * math.sqrt(len(kw_tokens))
    if denom <= 1e-9:
        return 0.0
    return num / denom

# ---- Keyword structures ----
class _Kw:
    __slots__ = ("idx", "text", "vol", "kd", "score", "eligible", "cats", "kw_tokens", "vol_norm")
    def __init__(self, idx, text, vol, kd, score, eligible, cats):
        self.idx = idx
        self.text = str(text or "")
        self.vol = float(vol) if pd.notna(vol) else 0.0
        self.kd = float(kd) if pd.notna(kd) else 0.0
        self.score = int(score) if pd.notna(score) else 0
        self.eligible = str(eligible) == "Yes"
        self.cats = set([c.strip() for c in str(cats or "").split(",") if c.strip()])
        self.kw_tokens = set(_tokens(self.text))
        self.vol_norm = 0.0

# ---- Mapping core ----
def _prepare_keywords_for_mapping(export_df: pd.DataFrame, kw_col: str, vol_col: str, kd_col: str) -> list[_Kw]:
    kws = []
    for i, row in export_df.reset_index(drop=False).iterrows():
        kws.append(
            _Kw(
                idx=row["index"],  # original DataFrame index
                text=row.get(kw_col, ""),
                vol=row.get(vol_col, 0),
                kd=row.get(kd_col, 0),
                score=row.get("Score", 0),
                eligible=row.get("Eligible", "No"),
                cats=row.get("Category", ""),
            )
        )
    # Volume normalization
    vols = [k.vol for k in kws]
    vmin, vmax = (min(vols), max(vols)) if vols else (0.0, 0.0)
    rng = max(vmax - vmin, 1e-9)
    for k in kws:
        k.vol_norm = (k.vol - vmin) / rng
    return kws

def _rank_score(rel: float, score: int, vol_norm: float) -> float:
    return 0.6 * rel + 0.3 * (score / 6.0) + 0.1 * vol_norm

def _assign_keywords_to_pages(pages: list[tuple[str, dict[str, float]]],
                              keywords: list[_Kw],
                              only_assign_eligible: bool) -> dict[int, tuple[str, str]]:
    """
    Returns mapping: keyword_row_index -> (url, role)
    Roles: Primary, Secondary, AIO, VEO
    Enforces uniqueness: a keyword is used at most once across the site.
    """
    mapping: dict[int, tuple[str, str]] = {}
    # Available keyword set
    available = [k for k in keywords if (k.eligible if only_assign_eligible else True)]
    assigned_kw_ids: set[int] = set()

    for url, page_vec in pages:
        # Build candidate list once for the page (skip zero-overlap)
        cands = []
        for k in available:
            if k.idx in assigned_kw_ids:
                continue
            if not k.kw_tokens:
                continue
            rel = _cosine_overlap(page_vec, k.kw_tokens)
            if rel <= 0:
                continue
            rank = _rank_score(rel, k.score, k.vol_norm)
            cands.append((rank, k, rel))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)

        # Primary
        primary = next((k for _, k, _ in cands if k.idx not in assigned_kw_ids), None)
        if primary:
            mapping[primary.idx] = (url, "Primary")
            assigned_kw_ids.add(primary.idx)

        # Secondary
        secondary = next((k for _, k, _ in cands if k.idx not in assigned_kw_ids), None)
        if secondary:
            mapping[secondary.idx] = (url, "Secondary")
            assigned_kw_ids.add(secondary.idx)

        # AIO
        aio = next((k for _, k, _ in cands if k.idx not in assigned_kw_ids and ("AIO" in k.cats)), None)
        if aio:
            mapping[aio.idx] = (url, "AIO")
            assigned_kw_ids.add(aio.idx)

        # VEO
        veo = next((k for _, k, _ in cands if k.idx not in assigned_kw_ids and ("VEO" in k.cats)), None)
        if veo:
            mapping[veo.idx] = (url, "VEO")
            assigned_kw_ids.add(veo.idx)

    return mapping

# ---- Crawl orchestrator ----
def _collect_pages(site_url: str, include_subdomains: bool, max_pages: int) -> dict[str, str]:
    base = _normalize_site(site_url)
    if not base:
        return {}

    rp = _load_robots(base)

    # 1) Try sitemaps
    pages_from_sitemap: set[str] = set()
    if HAVE_AIOHTTP:
        try:
            pages_from_sitemap = asyncio.run(_gather_sitemap_urls_async(base, include_subdomains, max_pages))
        except RuntimeError:
            # If inside existing event loop (rare in Streamlit), fall back to sync
            pages_from_sitemap = _gather_sitemap_urls_sync(base, include_subdomains, max_pages)
    else:
        pages_from_sitemap = _gather_sitemap_urls_sync(base, include_subdomains, max_pages)

    # Filter robots + fetch
    html_map: dict[str, str] = {}
    if pages_from_sitemap:
        # Fetch content for sitemap URLs
        if HAVE_AIOHTTP:
            async def _fetch_many(urls: list[str]) -> dict[str, str]:
                out: dict[str, str] = {}
                async with aiohttp.ClientSession(headers=_DEFAULT_HEADERS) as session:
                    for u in urls:
                        if len(out) >= max_pages:
                            break
                        if not rp.can_fetch(_DEFAULT_HEADERS["User-Agent"], u):
                            continue
                        html = await _fetch_async(session, u)
                        if html:
                            out[u] = html
                return out
            try:
                html_map = asyncio.run(_fetch_many(list(pages_from_sitemap)[:max_pages]))
            except RuntimeError:
                # event loop edge-case -> sync fallback
                for u in list(pages_from_sitemap)[:max_pages]:
                    if not rp.can_fetch(_DEFAULT_HEADERS["User-Agent"], u):
                        continue
                    h = _fetch_sync(u)
                    if h:
                        html_map[u] = h
        else:
            for u in list(pages_from_sitemap)[:max_pages]:
                if not rp.can_fetch(_DEFAULT_HEADERS["User-Agent"], u):
                    continue
                h = _fetch_sync(u)
                if h:
                    html_map[u] = h

    # 2) If sitemap insufficient, BFS crawl
    if len(html_map) < max_pages:
        remaining = max_pages - len(html_map)
        if HAVE_AIOHTTP:
            try:
                bfs_map = asyncio.run(_crawl_bfs_async(base, include_subdomains, remaining, rp))
            except RuntimeError:
                bfs_map = _crawl_bfs_sync(base, include_subdomains, remaining, rp)
        else:
            bfs_map = _crawl_bfs_sync(base, include_subdomains, remaining, rp)
        # Merge (avoid duplicates)
        for k, v in bfs_map.items():
            if len(html_map) >= max_pages:
                break
            if k not in html_map:
                html_map[k] = v

    return html_map

# =========================================================
# ================= Button Action Handler =================
# =========================================================
if submit and df is not None:
    # ---------- Score/Categorize ----------
    scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

    # Build export base (same as before)
    filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
    export_df = scored[base_cols].copy()
    export_df["Strategy"] = scoring_mode

    export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes": 1, "No": 0}).fillna(0)
    export_df = export_df.sort_values(
        by=["_EligibleSort", kd_col, vol_col], ascending=[False, True, False], kind="mergesort"
    ).drop(columns=["_EligibleSort"])
    export_cols = base_cols + ["Strategy"]
    export_df = export_df[export_cols]

    # ---------- Crawl site (silent) ----------
    with st.spinner("Crawling site and mapping keywords to pages..."):
        html_map = _collect_pages(site_url, include_subdomains=include_subdomains, max_pages=int(max_pages))

    # ---------- Build page profiles ----------
    pages_profiles: list[tuple[str, dict[str, float]]] = []
    for u, html in html_map.items():
        vec = _page_profile(u, html)
        if vec:
            pages_profiles.append((u, vec))

    # If nothing crawled/profiled, we still provide the scored CSV (mapping blank)
    if not pages_profiles:
        export_df["Mapped URL"] = ""
        export_df["Mapped Role"] = "Unassigned"
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download scored + mapping CSV",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            help="Includes mapping columns; mapping may be blank if the crawl yielded no pages."
        )
    else:
        # ---------- Prepare keywords + assign ----------
        kw_objs = _prepare_keywords_for_mapping(export_df, kw_col, vol_col, kd_col)
        mapping = _assign_keywords_to_pages(pages_profiles, kw_objs, only_assign_eligible=only_assign_eligible)

        # ---------- Merge mapping back to export ----------
        # We'll use export_df's index as the key
        mapped_urls = []
        mapped_roles = []
        for idx in export_df.index.tolist():
            if idx in mapping:
                url, role = mapping[idx]
                mapped_urls.append(url)
                mapped_roles.append(role)
            else:
                mapped_urls.append("")
                mapped_roles.append("Unassigned")
        export_df["Mapped URL"] = mapped_urls
        export_df["Mapped Role"] = mapped_roles

        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download scored + mapping CSV",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            help="Sorted by eligibility (Yes first), KD ascending, Volume descending, with per-keyword URL mapping."
        )

        # Optional preview (top 10) with color on Score/Tier
        if st.checkbox("Preview first 10 rows (optional)", value=False, key="preview_mapping"):
            preview_df = export_df.copy()

            def _row_style(row):
                color = COLOR_MAP.get(int(row.get("Score", 0)) if pd.notna(row.get("Score", 0)) else 0, "#9ca3af")
                return [
                    ("background-color: " + color + "; color: black;") if c in ("Score", "Tier") else ""
                    for c in row.index
                ]

            styled = preview_df.head(10).style.apply(_row_style, axis=1)
            st.dataframe(styled, use_container_width=True)

st.markdown("---")
st.caption("© 2025 OutrankIQ • Select from three scoring strategies to target different types of keyword opportunities.")
