import io
import re
import json
import math
import html
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from urllib.parse import urlparse, urljoin

# ---------- Optional deps (kept lightweight) ----------
try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

try:
    from bs4 import BeautifulSoup  # noqa: F401
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

import xml.etree.ElementTree as ET

st.set_page_config(page_title="OutrankIQ", page_icon="🔎", layout="centered")

st.title("OutrankIQ")
st.caption("Score keywords by Search Volume (A) and Keyword Difficulty (B) — with selectable scoring strategies and optional URL mapping.")

# =========================
# Helpers (existing)
# =========================
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
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|step[- ]?by[- ]?step|tutorial|guide|meaning)\b", re.I)
AEO_PAT = re.compile(r"^\s*(who|what|when|where|why|how|which|can|should)\b", re.I)
VEO_PAT = re.compile(r"\b(near me|open now|closest|call now|directions|ok google|alexa|siri|hey google|pickup)\b", re.I)
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

    # Eligibility + Reason (Option A)
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

# =========================
# URL Mapping: helpers
# =========================
def _normalize_domain_input(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = "https://" + s
    p = urlparse(s)
    return p.netloc.lower()

def _http_get(url: str, timeout=10):
    if not HAVE_REQUESTS:
        raise RuntimeError("The 'requests' package is required for site mapping.")
    ua = "OutrankIQ/1.0 (+https://example.com)"
    return requests.get(url, timeout=timeout, headers={"User-Agent": ua})

def _robots_sitemaps(base_domain: str) -> list[str]:
    """Return list of sitemap URLs from robots.txt; fall back to common paths."""
    if not base_domain:
        return []
    root = f"https://{base_domain}"
    robots_url = urljoin(root, "/robots.txt")
    sitemaps = []
    try:
        r = _http_get(robots_url, timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm = line.split(":", 1)[1].strip()
                    if sm:
                        sitemaps.append(sm)
    except Exception:
        pass
    # Fall back to common paths
    common = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml", "/wp-sitemap.xml",
              "/sitemap.xml.gz", "/sitemap_index.xml.gz", "/sitemap-index.xml.gz", "/wp-sitemap.xml.gz"]
    for path in common:
        sitemaps.append(urljoin(root, path))
    # de-dupe, preserve order
    seen = set()
    out = []
    for s in sitemaps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _parse_sitemap_xml(xml_text: str) -> tuple[list[str], list[str]]:
    """Return (child_sitemaps, loc_urls)."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return [], []
    ns = ""
    if root.tag.endswith("sitemapindex"):
        locs = [e.text.strip() for e in root.findall(".//{*}loc") if e.text]
        return locs, []
    elif root.tag.endswith("urlset"):
        urls = [e.text.strip() for e in root.findall(".//{*}loc") if e.text]
        return [], urls
    else:
        # Try namespace-agnostic find
        locs = [e.text.strip() for e in root.findall(".//{*}loc") if e.text]
        # Heuristically decide
        if "sitemap" in root.tag:
            return locs, []
        return [], locs

def _gather_urls_from_sitemaps(domain: str, max_urls: int = 250) -> list[str]:
    sitemaps = _robots_sitemaps(domain)
    urls, queue = [], sitemaps[:]
    seen_sm = set()
    host_suffix = "." + domain if not domain.startswith("www.") else domain
    while queue and len(urls) < max_urls:
        sm = queue.pop(0)
        if sm in seen_sm:
            continue
        seen_sm.add(sm)
        try:
            r = _http_get(sm, timeout=12)
            if r.status_code != 200 or "xml" not in r.headers.get("Content-Type", ""):
                continue
            children, locs = _parse_sitemap_xml(r.text)
            for c in children:
                if c not in seen_sm:
                    queue.append(c)
            for u in locs:
                try:
                    p = urlparse(u)
                    if p.scheme in ("http", "https") and (p.netloc.endswith(domain) or p.netloc.endswith(host_suffix)):
                        urls.append(u)
                        if len(urls) >= max_urls:
                            break
                except Exception:
                    continue
        except Exception:
            continue
    # de-dup while preserving order
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered[:max_urls]

# ---- Topic extraction (skim-scan)
STOPWORDS = set("""a an the and or of for to in on with without within by from at as into over under about
is are be been being it its their our your his her they them we you i this that those these home contact careers news
""".split())

def _simple_stem(w: str) -> str:
    w = w.lower()
    if w.endswith("ies"): return w[:-3] + "y"
    if w.endswith("ves"): return w[:-3] + "f"
    if w.endswith("ing") and len(w) > 5: return w[:-3]
    if w.endswith("ers") and len(w) > 4: return w[:-1]
    if w.endswith("s") and len(w) > 3: return w[:-1]
    return w

def _normalize_text_to_tokens(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [_simple_stem(t) for t in s.split() if t not in STOPWORDS and len(t) > 1]

def _token_set_from_signals(signals: list[str]) -> set[str]:
    toks = []
    for s in signals:
        toks.extend(_normalize_text_to_tokens(s or ""))
    return set(toks)

def _extract_signals_from_html(html_text: str) -> list[str]:
    """Return a list of short text signals (title, meta desc, og tags, H1, a few H2s, first ~120 words)."""
    signals = []
    if HAVE_BS4:
        soup = BeautifulSoup(html_text, "html.parser")
        # title
        if soup.title and soup.title.string:
            signals.append(soup.title.string.strip())
        # meta description
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            signals.append(md["content"].strip())
        # OG tags (allowed)
        ogt = soup.find("meta", attrs={"property": "og:title"})
        if ogt and ogt.get("content"):
            signals.append(ogt["content"].strip())
        ogd = soup.find("meta", attrs={"property": "og:description"})
        if ogd and ogd.get("content"):
            signals.append(ogd["content"].strip())
        # H1 + up to 3 H2s
        h1 = soup.find("h1")
        if h1:
            signals.append(h1.get_text(" ", strip=True))
        for h2 in soup.find_all("h2")[:3]:
            signals.append(h2.get_text(" ", strip=True))
        # first ~120 words from main/article or first paragraph
        main = soup.find("main") or soup.find("article")
        block = main.get_text(" ", strip=True) if main else ""
        if not block:
            p = soup.find("p")
            if p:
                block = p.get_text(" ", strip=True)
        if block:
            words = block.split()
            short = " ".join(words[:120])
            signals.append(short)
    else:
        # Minimal regex fallback
        title = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
        if title:
            signals.append(html.unescape(title.group(1)).strip())
        md = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', html_text, re.I | re.S)
        if md:
            signals.append(html.unescape(md.group(1)).strip())
        ogt = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']', html_text, re.I | re.S)
        if ogt:
            signals.append(html.unescape(ogt.group(1)).strip())
        ogd = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']', html_text, re.I | re.S)
        if ogd:
            signals.append(html.unescape(ogd.group(1)).strip())
        # first ~120 words (super simple)
        text = re.sub(r"<[^>]+>", " ", html_text)
        words = text.split()
        signals.append(" ".join(words[:120]))
    return [s for s in signals if s and s.strip()]

def _fetch_topic_tokens_for_urls(urls: list[str], rate_limit_per_sec: float = 2.0) -> dict[str, set[str]]:
    """Fetch each URL and return {url: token_set_from_signals}."""
    tokens_by_url = {}
    if not HAVE_REQUESTS:
        return tokens_by_url
    delay = 1.0 / max(0.1, rate_limit_per_sec)
    for u in urls:
        try:
            r = _http_get(u, timeout=12)
            if r.status_code == 200 and "html" in r.headers.get("Content-Type", ""):
                signals = _extract_signals_from_html(r.text)
                tokens_by_url[u] = _token_set_from_signals(signals)
        except Exception:
            pass
        time.sleep(delay)
    return tokens_by_url

def _is_aio(kw: str) -> bool:
    return bool(AIO_PAT.search(kw))

def _is_veo(kw: str) -> bool:
    return bool(VEO_PAT.search(kw))

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _score_keyword(kw_tokens: set[str], topic_tokens: set[str]) -> float:
    base = _jaccard(kw_tokens, topic_tokens)
    # light length boost capped
    length_boost = min(len(kw_tokens) / 6.0, 0.3)
    return base + length_boost

def _normalize_kw_df_for_mapping(df: pd.DataFrame, kw_col: str, vol_col: str, kd_col: str) -> pd.DataFrame:
    out = df.copy()
    out[kw_col] = out[kw_col].astype(str).str.strip()
    out["_kw_lower"] = out[kw_col].str.lower()
    out[vol_col] = pd.to_numeric(out[vol_col], errors="coerce")
    out[kd_col] = pd.to_numeric(out[kd_col], errors="coerce")
    # Precompute tokens & features
    out["_kw_tokens"] = out[kw_col].map(lambda s: set(_normalize_text_to_tokens(s)))
    out["_is_aio"] = out[kw_col].map(_is_aio)
    out["_is_veo"] = out[kw_col].map(_is_veo)
    return out

def _auto_map_keywords_to_urls(
    kw_pool: pd.DataFrame,
    kw_col: str,
    vol_col: str,
    kd_col: str,
    topic_tokens: dict[str, set[str]],
    honor_strategy_thresholds: bool = True,
    dedupe_across_urls: bool = True,
    min_volume: int = 0,
) -> dict[str, str]:
    """
    Return dict {keyword_lower: url} for the keywords selected as Primary, Secondary, AIO, VEO for each URL.
    """
    used_kw = set()
    mapping = {}

    # Build a mask for eligibility if honoring thresholds
    if honor_strategy_thresholds:
        eligible_mask = (pd.to_numeric(kw_pool[vol_col], errors="coerce") >= min_volume)
    else:
        eligible_mask = pd.Series([True] * len(kw_pool), index=kw_pool.index)

    # For deterministic tie-breaking
    def _sorted_candidates(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(
            by=["_match_score", vol_col, kd_col],
            ascending=[False, False, True],
            kind="mergesort",
        )

    for url, tset in topic_tokens.items():
        if not tset:
            continue

        # Compute per-URL match scores
        local = kw_pool.copy()
        local["_match_score"] = local.apply(lambda r: _score_keyword(r["_kw_tokens"], tset), axis=1)

        # Selection pipeline
        def pick(mask) -> str:
            sub = local[eligible_mask & mask].copy()
            if dedupe_across_urls and used_kw:
                sub = sub[~sub["_kw_lower"].isin(used_kw)]
            if sub.empty:
                return ""
            sub = _sorted_candidates(sub)
            return sub.iloc[0]["_kw_lower"]

        # Primary (best non-AIO/VEO)
        prim = pick(mask=(~local["_is_aio"]) & (~local["_is_veo"]))
        if prim:
            mapping[prim] = url
            used_kw.add(prim)

        # Secondary (next best non-AIO/VEO, different from Primary)
        sec = pick(mask=(~local["_is_aio"]) & (~local["_is_veo"]))
        if sec:
            mapping[sec] = url
            used_kw.add(sec)

        # AIO (prefer question/definition/meaning style)
        aio = pick(mask=local["_is_aio"])
        if not aio:
            # Fallback: any good match
            aio = pick(mask=(local["_match_score"] > 0.05))
        if aio:
            mapping[aio] = url
            used_kw.add(aio)

        # VEO (prefer near me / directions / pickup etc.)
        veo = pick(mask=local["_is_veo"])
        if not veo:
            # Fallback: any reasonable match
            veo = pick(mask=(local["_match_score"] > 0.02))
        if veo:
            mapping[veo] = url
            used_kw.add(veo)

    return mapping

# Session store for mapping artifacts
if "site_mapping" not in st.session_state:
    st.session_state.site_mapping = {
        "urls": [],
        "tokens_by_url": {},
        "keyword_url_map": {},  # {keyword_lower: url}
    }

# =========================
# Single keyword (unchanged)
# =========================
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
    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
        st.stop()

    # Clean numbers (commas, spaces, percents)
    if vol_col:
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    if kd_col:
        df[kd_col] = df[kd_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

    # Keep a normalized keyword pool for mapping downstream
    if kw_col:
        kw_pool = _normalize_kw_df_for_mapping(df, kw_col=kw_col, vol_col=vol_col, kd_col=kd_col)
        st.session_state["keyword_pool_df"] = kw_pool
        st.session_state["keyword_pool_names"] = {"kw": kw_col, "vol": vol_col, "kd": kd_col}
    else:
        kw_pool = None

    scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

    # =========================
    # SITE MAPPING (no preview)
    # =========================
    st.markdown("---")
    st.subheader("Site Mapping (optional) — associate keywords to URLs")

    colA, colB = st.columns([2, 1])
    with colA:
        domain_input = st.text_input("Domain (we’ll fetch sitemaps)", placeholder="example.com")
        pasted_urls = st.text_area("…or paste URLs (one per line)", placeholder="https://example.com/page-1\nhttps://example.com/page-2")
    with colB:
        max_pages = st.number_input("Max pages to map", min_value=1, max_value=5000, value=250, step=50)
        honor_thresholds = st.checkbox("Honor strategy thresholds when mapping", value=True,
                                       help=f"Only map keywords with Volume ≥ {MIN_VALID_VOLUME}")
        dedupe_keywords = st.checkbox("Deduplicate keywords across URLs", value=True,
                                      help="A keyword can only be assigned to one URL during mapping.")

    map_actions = st.columns([1, 1, 1, 2])
    with map_actions[0]:
        build_from_sitemap = st.button("🧭 Build from Sitemap", use_container_width=True)
    with map_actions[1]:
        use_pasted_list = st.button("📄 Use Pasted List", use_container_width=True)
    with map_actions[2]:
        auto_map = st.button("✨ Auto-map Keywords to URLs", use_container_width=True)
    with map_actions[3]:
        clear_map = st.button("🧹 Clear Mapping", use_container_width=True)

    # --- Handlers (silent; no preview table) ---
    if clear_map:
        st.session_state.site_mapping = {"urls": [], "tokens_by_url": {}, "keyword_url_map": {}}
        st.success("Cleared mapping state.")

    if build_from_sitemap:
        base = _normalize_domain_input(domain_input)
        if not base:
            st.warning("Please enter a valid domain (e.g., example.com).")
        elif not HAVE_REQUESTS:
            st.error("The 'requests' package is required to fetch sitemaps.")
        else:
            urls = _gather_urls_from_sitemaps(base, max_urls=int(max_pages))
            st.session_state.site_mapping["urls"] = urls
            st.success(f"Discovered {len(urls)} URL(s) from sitemap.")
    if use_pasted_list:
        urls = [u.strip() for u in (pasted_urls or "").splitlines() if u.strip()]
        st.session_state.site_mapping["urls"] = urls[: int(max_pages)]
        st.success(f"Accepted {len(st.session_state.site_mapping['urls'])} pasted URL(s).")

    if auto_map:
        urls = st.session_state.site_mapping.get("urls", [])
        if not urls:
            st.warning("No URLs to map. Build from sitemap or paste URLs first.")
        elif "keyword_pool_df" not in st.session_state or not kw_col:
            st.warning("Upload a keyword CSV first so we can map from your pool.")
        elif not HAVE_REQUESTS:
            st.error("The 'requests' package is required to fetch pages for topic extraction.")
        else:
            tokens_by_url = _fetch_topic_tokens_for_urls(urls)
            st.session_state.site_mapping["tokens_by_url"] = tokens_by_url
            pool = st.session_state["keyword_pool_df"]
            names = st.session_state["keyword_pool_names"]
            mapping = _auto_map_keywords_to_urls(
                kw_pool=pool,
                kw_col=names["kw"],
                vol_col=names["vol"],
                kd_col=names["kd"],
                topic_tokens=tokens_by_url,
                honor_strategy_thresholds=honor_thresholds,
                dedupe_across_urls=dedupe_keywords,
                min_volume=MIN_VALID_VOLUME if honor_thresholds else 0,
            )
            st.session_state.site_mapping["keyword_url_map"] = mapping
            st.success(f"Mapped {len(mapping)} keyword(s) across {len(tokens_by_url)} URL(s).")

    # ---------- CSV DOWNLOAD (sorted: Yes first, KD ↑ then Volume ↓) ----------
    filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    base_cols = ([kw_col] if kw_col else []) + [
        vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"
    ]
    export_df = scored[base_cols].copy()
    export_df["Strategy"] = scoring_mode

    # Inject Mapped URL (if keyword column present and we have a mapping)
    if kw_col:
        mapping = st.session_state.site_mapping.get("keyword_url_map", {}) or {}
        # Case-insensitive match
        def _map_kw_to_url(k):
            kl = str(k).strip().lower()
            return mapping.get(kl, "")
        export_df["Mapped URL"] = export_df[kw_col].map(_map_kw_to_url)

    export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes": 1, "No": 0}).fillna(0)
    export_df = export_df.sort_values(
        by=["_EligibleSort", kd_col, vol_col],
        ascending=[False, True, False],
        kind="mergesort"
    ).drop(columns=["_EligibleSort"])

    # Control final column order: place "Mapped URL" just before "Strategy" if present
    export_cols = base_cols + (["Mapped URL"] if "Mapped URL" in export_df.columns else []) + ["Strategy"]
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
st.caption("© 2025 OutrankIQ • Select from three scoring strategies and optionally map keywords to URLs via sitemaps or a pasted list.")
