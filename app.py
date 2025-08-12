import io
import re
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------- Optional bs4 for future use (safe if missing) ----------
try:
    from bs4 import BeautifulSoup  # noqa: F401
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

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

LABEL_MAP = {6: "Elite", 5: "Excellent", 4: "Good", 3: "Fair", 2: "Low", 1: "Very Low", 0: "Not rated"}

# Used for card + preview styling only (NOT exported)
COLOR_MAP = {
    6: "#2ecc71", 5: "#a3e635", 4: "#facc15", 3: "#fb923c", 2: "#f87171", 1: "#ef4444", 0: "#9ca3af",
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
else:  # Competitive
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
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|step[- ]?by[- ]?step|tutorial|guide|is)\b", re.I)
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
example = pd.DataFrame({"Keyword": ["best running shoes", "seo tools", "crm software"], "Volume": [5400, 880, 12000], "KD": [38, 72, 18]})
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
    if vol_col is None: missing.append("Volume")
    if kd_col is None: missing.append("Keyword Difficulty")

    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
    else:
        # Clean numbers
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[kd_col] = df[kd_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

        scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

        # Build base export (no download yet; we add mapping after crawl)
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes": 1, "No": 0}).fillna(0)
        export_df = export_df.sort_values(
            by=["_EligibleSort", kd_col, vol_col],
            ascending=[False, True, False],
            kind="mergesort"
        ).drop(columns=["_EligibleSort"])

        export_cols = base_cols + ["Strategy"]
        export_df = export_df[export_cols]

        # Persist for mapping stage
        st.session_state["export_df_base"] = export_df
        st.session_state["filename_base"] = filename_base
        st.session_state["kw_col_name"] = kw_col

# =========================
# Site Mapping (backend crawl, minimal UI)
# =========================
st.markdown("---")
st.subheader("Site Mapping")

# ---- Internal crawler parameters (edit in code if needed) ----
CRAWL_MAX_PAGES = 300          # total HTML pages to fetch
CRAWL_TIMEOUT_SEC = 12         # per-request timeout
CRAWL_CONCURRENCY = 20         # concurrent requests (async) or threads (sync)
RESPECT_ROBOTS = True          # obey robots.txt
STRIP_QUERYSTRINGS = True      # treat ?a=b as same page
SAME_HOST_ONLY = True          # don't leave the exact host

# Imports local to this section
import asyncio
import html as _html
from urllib.parse import urljoin, urldefrag, urlparse
import urllib.robotparser as urobot
from collections import deque

# Optional async dependency
try:
    import aiohttp
    HAVE_AIOHTTP = True
except Exception:
    HAVE_AIOHTTP = False

# Always-available sync fallback
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Minimal UI: URL + button only ----
crawl_url = st.text_input("Site URL to crawl", placeholder="https://example.com", value="")
btn_crawl = st.button("🚀 Crawl Site")

BINARY_EXT = (
    ".jpg",".jpeg",".png",".gif",".webp",".svg",".pdf",".zip",".rar",".7z",".gz",".mp3",".mp4",
    ".avi",".mov",".wmv",".mkv",".doc",".docx",".xls",".xlsx",".ppt",".pptx",".ico",".dmg",".exe"
)

def _normalize_url(base: str, href: str, strip_q: bool) -> str | None:
    if not href:
        return None
    try:
        u = urljoin(base, href)
        u, _frag = urldefrag(u)
        p = urlparse(u)
        if not p.scheme.startswith("http"):
            return None
        if strip_q:
            u = f"{p.scheme}://{p.netloc}{p.path}"
        low_path = p.path.lower()
        if any(low_path.endswith(ext) for ext in BINARY_EXT):
            return None
        return u
    except Exception:
        return None

def _same_host(u: str, root: str) -> bool:
    pu, pr = urlparse(u), urlparse(root)
    return (pu.netloc == pr.netloc)

def _extract_fields(html_text: str):
    title, meta_desc, h1s, h2s, h3s = "", "", [], [], []
    text = ""
    try:
        if HAVE_BS4:
            soup = BeautifulSoup(html_text, "lxml")
            t = soup.find("title"); title = t.get_text(strip=True) if t else ""
            md = soup.find("meta", attrs={"name":"description"}); meta_desc = md.get("content","").strip() if md else ""
            h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
            h2s = [h.get_text(strip=True) for h in soup.find_all("h2")]
            h3s = [h.get_text(strip=True) for h in soup.find_all("h3")]
            for s in soup(["script","style","noscript","svg","nav","footer","form"]): s.decompose()
            text = " ".join(soup.stripped_strings)
        else:
            import re as _re
            tt = _re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=_re.I|_re.S)
            title = _html.unescape(tt.group(1).strip()) if tt else ""
            md = _re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', html_text, flags=_re.I|_re.S)
            meta_desc = _html.unescape(md.group(1).strip()) if md else ""
            h1s = [_html.unescape(x.strip()) for x in _re.findall(r"<h1[^>]*>(.*?)</h1>", html_text, flags=_re.I|_re.S)]
            h2s = [_html.unescape(x.strip()) for x in _re.findall(r"<h2[^>]*>(.*?)</h2>", html_text, flags=_re.I|_re.S)]
            h3s = [_html.unescape(x.strip()) for x in _re.findall(r"<h3[^>]*>(.*?)</h3>", html_text, flags=_re.I|_re.S)]
            text = _re.sub(r"<[^>]+>", " ", html_text); text = " ".join(text.split())
    except Exception:
        pass

    # We do NOT display these to the user; used only for backend mapping
    combined = " ".join([title, meta_desc] + h1s + h2s + h3s + [text])
    return title, meta_desc, h1s, h2s, h3s, combined[:200000]

# ---------- Async (aiohttp) path ----------
async def _fetch_async(session, url: str, timeout: int):
    try:
        async with session.get(url, timeout=timeout, allow_redirects=True) as r:
            if r.status != 200:
                return None
            ctype = r.headers.get("content-type","").lower()
            if "text/html" not in ctype:
                return None
            return await r.text(errors="ignore")
    except Exception:
        return None

async def crawl_site_async(root_url: str, max_pages: int, concurrency: int, timeout: int,
                           same_host_only: bool, strip_query: bool, respect_robots: bool):
    rp = None
    if respect_robots:
        try:
            pr = urlparse(root_url); robots_url = f"{pr.scheme}://{pr.netloc}/robots.txt"
            rp = urobot.RobotFileParser(); rp.set_url(robots_url); rp.read()
        except Exception:
            rp = None

    seen, q = set(), deque()
    start = _normalize_url(root_url, root_url, strip_query)
    if not start: return [], 0
    q.append(start); seen.add(start)

    sem = asyncio.Semaphore(concurrency)
    pages = []

    async with aiohttp.ClientSession(headers={"User-Agent": "OutrankIQ/1.2 (+https://outrankiq)"}) as session:
        async def worker():
            while q and len(pages) < max_pages:
                url = q.popleft()
                if respect_robots and rp:
                    try:
                        if not rp.can_fetch("*", url): continue
                    except Exception:
                        pass
                async with sem:
                    html_text = await _fetch_async(session, url, timeout)
                if html_text is None: continue

                title, meta_desc, h1s, h2s, h3s, combined = _extract_fields(html_text)
                pages.append({"url": url, "title": title, "meta": meta_desc, "h1": h1s, "h2": h2s, "h3": h3s, "text": combined})

                try:
                    if HAVE_BS4:
                        soup = BeautifulSoup(html_text, "lxml"); links = [a.get("href","") for a in soup.find_all("a")]
                    else:
                        import re as _re; links = _re.findall(r'href=["\'](.*?)["\']', html_text, flags=_re.I)
                except Exception:
                    links = []

                for href in links:
                    u = _normalize_url(url, href, strip_query)
                    if not u or u in seen: continue
                    if same_host_only and not _same_host(u, root_url): continue
                    seen.add(u)
                    if len(seen) <= max_pages * 3: q.append(u)

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks)

    return pages, len(seen)

def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try: loop.close()
            except Exception: pass

# ---------- Sync (requests + threads) fallback ----------
def _fetch_sync(session: requests.Session, url: str, timeout: int):
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": "OutrankIQ/1.2 (+https://outrankiq)"})
        if r.status_code != 200: return None
        ctype = r.headers.get("content-type","").lower()
        if "text/html" not in ctype: return None
        return r.text
    except Exception:
        return None

def crawl_site_sync(root_url: str, max_pages: int, workers: int, timeout: int,
                    same_host_only: bool, strip_query: bool, respect_robots: bool):
    rp = None
    if respect_robots:
        try:
            pr = urlparse(root_url); robots_url = f"{pr.scheme}://{pr.netloc}/robots.txt"
            rp = urobot.RobotFileParser(); rp.set_url(robots_url); rp.read()
        except Exception:
            rp = None

    start = _normalize_url(root_url, root_url, strip_query)
    if not start: return [], 0

    seen, q = set([start]), deque([start])
    pages = []
    session = requests.Session()

    while q and len(pages) < max_pages:
        batch = []
        while q and len(batch) < workers: batch.append(q.popleft())

        future_to_url = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for url in batch:
                if respect_robots and rp:
                    try:
                        if not rp.can_fetch("*", url): continue
                    except Exception:
                        pass
                future_to_url[ex.submit(_fetch_sync, session, url, timeout)] = url

            for fut in as_completed(future_to_url):
                url = future_to_url[fut]
                html_text = fut.result()
                if html_text is None: continue

                title, meta_desc, h1s, h2s, h3s, combined = _extract_fields(html_text)
                pages.append({"url": url, "title": title, "meta": meta_desc, "h1": h1s, "h2": h2s, "h3": h3s, "text": combined})

                try:
                    if HAVE_BS4:
                        soup = BeautifulSoup(html_text, "lxml"); links = [a.get("href","") for a in soup.find_all("a")]
                    else:
                        import re as _re; links = _re.findall(r'href=["\'](.*?)["\']', html_text, flags=_re.I)
                except Exception:
                    links = []

                for href in links:
                    u = _normalize_url(url, href, strip_query)
                    if not u or u in seen: continue
                    if same_host_only and not _same_host(u, root_url): continue
                    seen.add(u)
                    if len(seen) <= max_pages * 3: q.append(u)

    return pages, len(seen)

def _tokenize(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-_/]", " ", s)
    return set([t for t in re.split(r"[\s/_\-]+", s) if t])

def score_keyword_against_page(keyword: str, page: dict) -> tuple[float, str]:
    """
    Lightweight scoring (URL > Title > H1 > H2/H3 > Meta; body fallback)
    Returns (score, best_field)
    """
    kw_tokens = _tokenize(keyword)
    if not kw_tokens: return 0.0, ""

    from urllib.parse import urlparse as _up
    url_tokens = _tokenize(_up(page["url"]).path)
    title_tokens = _tokenize(page.get("title",""))
    meta_tokens = _tokenize(page.get("meta",""))
    h1_tokens = set().union(*[_tokenize(h) for h in page.get("h1",[]) or []])
    h2_tokens = set().union(*[_tokenize(h) for h in page.get("h2",[]) or []])
    h3_tokens = set().union(*[_tokenize(h) for h in page.get("h3",[]) or []])

    def overlap(a, b): return len(a & b)

    scores = {
        "url": 3.0 * overlap(kw_tokens, url_tokens),
        "title": 2.5 * overlap(kw_tokens, title_tokens),
        "h1": 2.2 * overlap(kw_tokens, h1_tokens),
        "h2h3": 1.5 * (overlap(kw_tokens, h2_tokens) + overlap(kw_tokens, h3_tokens)),
        "meta": 1.2 * overlap(kw_tokens, meta_tokens),
    }
    total = sum(scores.values())

    body_hit = 0.0
    if total == 0 and page.get("text"):
        present = sum(1 for t in kw_tokens if t in page["text"].lower())
        if present: body_hit = 0.8 * present
    total += body_hit

    best_field = max(scores, key=scores.get) if total > 0 else ("body" if body_hit > 0 else "")
    return float(total), best_field

def suggest_url_for_keyword(keyword: str, site_index: list[dict]) -> str:
    """Return just the best URL for the keyword."""
    if not site_index: return ""
    best_url, best_score = "", 0.0
    for p in site_index:
        score, _ = score_keyword_against_page(keyword, p)
        if score > best_score:
            best_score, best_url = score, p["url"]
    return best_url

# ---- Run crawl (backend-only signals; no page details shown) ----
if btn_crawl and crawl_url.strip():
    with st.spinner("Crawling site..."):
        try:
            if HAVE_AIOHTTP:
                pages, _ = _run_async(
                    crawl_site_async(
                        crawl_url.strip(),
                        max_pages=CRAWL_MAX_PAGES,
                        concurrency=CRAWL_CONCURRENCY,
                        timeout=CRAWL_TIMEOUT_SEC,
                        same_host_only=SAME_HOST_ONLY,
                        strip_query=STRIP_QUERYSTRINGS,
                        respect_robots=RESPECT_ROBOTS,
                    )
                )
            else:
                pages, _ = crawl_site_sync(
                    crawl_url.strip(),
                    max_pages=CRAWL_MAX_PAGES,
                    workers=CRAWL_CONCURRENCY,
                    timeout=CRAWL_TIMEOUT_SEC,
                    same_host_only=SAME_HOST_ONLY,
                    strip_query=STRIP_QUERYSTRINGS,
                    respect_robots=RESPECT_ROBOTS,
                )
            st.session_state["site_index"] = pages
            st.success("Crawl completed.")
        except Exception as e:
            st.error(f"Crawler error: {e}")

# ---- Single download: Bulk scoring + URL site mapping per keyword ----
has_export = "export_df_base" in st.session_state and isinstance(st.session_state.get("export_df_base"), pd.DataFrame)
has_site = "site_index" in st.session_state and isinstance(st.session_state.get("site_index"), list) and len(st.session_state["site_index"]) > 0

if has_export and has_site:
    export_df_base = st.session_state["export_df_base"].copy()
    filename_base = st.session_state.get("filename_base", f"outrankiq_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    kw_col_live = st.session_state.get("kw_col_name")

    if (kw_col_live is None) or (kw_col_live not in export_df_base.columns):
        kw_col_live = find_column(export_df_base, ["keyword","query","term"])

    if kw_col_live is None:
        st.error("Unable to locate the Keyword column in the scored data.")
    else:
        site_index = st.session_state["site_index"]

        mapped_urls = []
        for kw in export_df_base[kw_col_live].astype(str):
            mapped_urls.append(suggest_url_for_keyword(kw, site_index))

        export_df_final = export_df_base.copy()
        export_df_final["Suggested URL"] = mapped_urls

        csv_bytes = export_df_final.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download Scored CSV (with URL Site Mapping)",
            data=csv_bytes,
            file_name=f"{filename_base}_site_mapping.csv",
            mime="text/csv",
            help="Single file that includes keyword scoring and the Suggested URL per keyword."
        )
elif has_export and not has_site:
    st.info("Add a site URL and click “Crawl Site” to enable the mapped download.")
elif has_site and not has_export:
    st.info("Upload and score a CSV first to enable the mapped download.")

st.markdown("---")
st.caption("© 2025 OutrankIQ • Select from three scoring strategies to target different types of keyword opportunities.")
