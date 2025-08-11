import io
import re
import gzip
import pandas as pd
import streamlit as st
from datetime import datetime
from urllib.parse import urlparse
import requests
import xml.etree.ElementTree as ET
import concurrent.futures
import os

# Try to use BeautifulSoup if available; otherwise fallback to simple parsing
try:
    from bs4 import BeautifulSoup
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
    MIN_VALID_VOLUME = 1500
    KD_BUCKETS = [(0, 30, 6), (31, 45, 5), (46, 60, 4), (61, 70, 3), (71, 80, 2), (81, 100, 1)]
elif scoring_mode == "Competitive":
    MIN_VALID_VOLUME = 3000  # as requested
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
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|step[- ]?by[- ]?step|tutorial|guide)\b", re.I)
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

# ---------- URL mapping & scoring helpers ----------
TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(s: str) -> set[str]:
    if not isinstance(s, str):
        return set()
    return set(TOKEN_RE.findall(s.lower()))

def url_to_title(url: str) -> str:
    try:
        p = urlparse(url)
        last = p.path.strip("/").split("/")[-1] if p.path else ""
        last = last.replace("-", " ").replace("_", " ").strip()
        return last.title() if last else (p.netloc or url)
    except Exception:
        return url

def extract_page_tokens(url: str, title: str | None) -> set[str]:
    p = urlparse(url)
    path_tokens = tokenize(p.path.replace("/", " "))
    host_tokens = tokenize(p.netloc)
    title_tokens = tokenize(title or url_to_title(url))
    return path_tokens.union(host_tokens).union(title_tokens)

# Category-aware boosts for URL mapping
CATEGORY_BOOST_TERMS = {
    "AIO": {"guide", "tutorial", "how", "learn", "blog", "faq"},
    "AEO": {"faq", "questions", "what", "how", "who", "why"},
    "VEO": {"locations", "near", "store", "contact", "phone", "hours"},
    "GEO": {"guide", "how", "steps", "template", "framework"},
    "SXO": {"pricing", "compare", "comparison", "best", "review", "vs"},
    "LLM": {"ai", "docs", "developers", "api", "prompt", "gpt", "llm"},
    "SEO": set(),
}

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# --------- Faster networking & parsing ---------
USER_AGENT = "OutrankIQ/1.0"
FETCH_TIMEOUT = 2.5           # faster timeouts
CONTENT_MAX_BYTES = 120_000   # ~120KB/page cap
BODY_WORDS_LIMIT = 700        # first ~700 words

@st.cache_resource(show_spinner=False)
def get_http_session():
    """Shared requests.Session with connection pooling."""
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess

@st.cache_data(show_spinner=False)
def fetch_page_html(url: str) -> bytes | None:
    try:
        sess = get_http_session()
        with sess.get(url, timeout=FETCH_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            content = b""
            for chunk in r.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > CONTENT_MAX_BYTES:
                    break
            return content
    except Exception:
        return None

def simple_extract_signals_from_html(html_bytes: bytes) -> dict:
    """Fallback parsing without BeautifulSoup."""
    try:
        text = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return {}
    title_match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    meta_desc_match = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', text, re.I | re.S)
    body_match = re.search(r"<body[^>]*>(.*?)</body>", text, re.I | re.S)
    body_text = re.sub(r"<[^>]+>", " ", body_match.group(1)) if body_match else ""
    return {
        "title": (title_match.group(1).strip() if title_match else ""),
        "meta": (meta_desc_match.group(1).strip() if meta_desc_match else ""),
        "h1": "",
        "h2h3": "",
        "body": " ".join(body_text.split())
    }

def bs4_extract_signals_from_html(html_bytes: bytes) -> dict:
    soup = BeautifulSoup(html_bytes, "lxml" if HAVE_BS4 else "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "") or \
            (soup.find("meta", property="og:title") or {}).get("content", "") or ""
    meta =  (soup.find("meta", attrs={"name": "description"}) or {}).get("content", "") or \
            (soup.find("meta", property="og:description") or {}).get("content", "") or ""
    h1_el = soup.find("h1")
    h1 = h1_el.get_text(" ", strip=True) if h1_el else ""
    h2h3_parts = []
    for tag in soup.find_all(["h2", "h3"]):
        t = tag.get_text(" ", strip=True)
        if t:
            h2h3_parts.append(t)
    h2h3 = " | ".join(h2h3_parts[:8])
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    body_text = soup.get_text(" ", strip=True)
    words = body_text.split()
    if len(words) > BODY_WORDS_LIMIT:
        body_text = " ".join(words[:BODY_WORDS_LIMIT])
    return {"title": title, "meta": meta, "h1": h1, "h2h3": h2h3, "body": body_text}

def extract_page_signals(url: str) -> dict:
    html = fetch_page_html(url)
    if not html:
        return {}
    try:
        return bs4_extract_signals_from_html(html) if HAVE_BS4 else simple_extract_signals_from_html(html)
    except Exception:
        return simple_extract_signals_from_html(html)

def page_tokens_bundle(url: str, title: str | None, signals: dict) -> dict:
    slug_tokens = extract_page_tokens(url, title)  # host/path/title-slug
    def tok(s): return tokenize(s or "")
    sig = {
        "slug": slug_tokens,
        "title": tok(signals.get("title", "")),
        "meta": tok(signals.get("meta", "")),
        "h1": tok(signals.get("h1", "")),
        "h2h3": tok(signals.get("h2h3", "")),
        "body": tok(signals.get("body", "")),
    }
    sig["all"] = set().union(*sig.values())
    return sig

# ---------- Scoring across signals ----------
SIGNAL_WEIGHTS = {"title": 3.0, "h1": 2.0, "h2h3": 1.5, "meta": 1.5, "slug": 1.5, "body": 1.0}

def weighted_similarity(kw_tokens: set[str], token_bundle: dict) -> float:
    num, den = 0.0, 0.0
    for key, w in SIGNAL_WEIGHTS.items():
        s = jaccard(kw_tokens, token_bundle.get(key, set()))
        num += w * s
        den += w
    return num / den if den else 0.0

def score_keyword_to_page(keyword: str, categories: list[str], token_bundle: dict) -> float:
    kw_tokens = tokenize(keyword)
    base = weighted_similarity(kw_tokens, token_bundle)
    boost = 0.0
    all_tokens = token_bundle.get("all", set())
    for c in categories:
        terms = CATEGORY_BOOST_TERMS.get(c, set())
        if terms and (terms & all_tokens):
            boost += 0.05
    return min(base + boost, 1.0)

def make_page_obj(url: str, title: str | None) -> dict:
    signals = extract_page_signals(url)  # cached + fast
    tokens = page_tokens_bundle(url, title, signals)
    return {"url": url, "title": title or url_to_title(url), "tokens": tokens}

# ----- Domain-based sitemap discovery (always includes common subdomains) -----
COMMON_SUBDOMAINS = ["www", "blog", "docs", "help", "support", "learn", "resources"]

def normalize_domain(d: str) -> str:
    d = d.strip().lower().replace("http://", "").replace("https://", "").strip("/")
    return d

@st.cache_resource(show_spinner=False)
def get_plain_http_session():
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess

def fetch(url: str, timeout: float = 8.0):
    """Simple fetch for sitemaps/robots (no streaming)."""
    try:
        sess = get_plain_http_session()
        r = sess.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and r.content:
            return r
    except Exception:
        return None
    return None

def parse_sitemap(content_bytes: bytes) -> tuple[list[str], list[str]]:
    """Return (urlset_urls, child_sitemaps). Accepts raw XML bytes."""
    try:
        tree = ET.fromstring(content_bytes)
    except Exception:
        return [], []
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls, children = [], []
    # urlset
    for url in tree.findall(".//sm:url/sm:loc", ns):
        loc = (url.text or "").strip()
        if loc:
            urls.append(loc)
    # sitemapindex
    for loc in tree.findall(".//sm:sitemap/sm:loc", ns):
        child = (loc.text or "").strip()
        if child:
            children.append(child)
    return urls, children

def maybe_decompress(resp) -> bytes:
    """Handle .gz sitemaps and compressed responses."""
    try:
        content = resp.content
        if resp.headers.get("Content-Type", "").endswith("gzip") or resp.url.lower().endswith(".gz"):
            return gzip.decompress(content)
        return content
    except Exception:
        try:
            return gzip.decompress(resp.content)
        except Exception:
            return resp.content or b""

def robots_sitemaps(domain: str) -> list[str]:
    """Read robots.txt and extract all Sitemap: entries."""
    root = normalize_domain(domain)
    urls = []
    for scheme in ("https://", "http://"):
        r = fetch(f"{scheme}{root}/robots.txt", timeout=5.0)
        if not r:
            continue
        text = r.text or ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("sitemap:"):
                loc = line.split(":", 1)[1].strip()
                if loc:
                    urls.append(loc)
        if urls:
            break
    return list(dict.fromkeys(urls))  # dedupe, keep order

def alternate_sitemap_paths(domain: str) -> list[str]:
    """Try common sitemap locations on domain and common subdomains."""
    root = normalize_domain(domain)
    base_paths = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemaps.xml",
        "/sitemap/sitemap.xml",
        "/sitemap/sitemap_index.xml",
    ]
    subs = set([""] + COMMON_SUBDOMAINS)
    urls = []
    for sub in subs:
        host = f"{sub+'.' if sub else ''}{root}"
        for pth in base_paths:
            urls.append(f"https://{host}{pth}")
    return urls

def discover_urls_from_sitemaps(domain: str, extra_subs: list[str], per_sub_cap: int = 500):
    """
    Returns:
      discovered_urls: list[str]
      stats: dict with discovery diagnostics
    """
    root = normalize_domain(domain)
    if not root:
        return [], {"subdomains": [], "urls_per_sub": {}, "sitemaps_tried": 0, "sitemaps_ok": 0}

    # Start with bare + common subdomains + user extras
    initial_subs = set([""])
    initial_subs.update(COMMON_SUBDOMAINS)
    for s in extra_subs:
        s = s.strip()
        if s:
            initial_subs.add(s)

    # 1) Collect sitemap endpoints: robots + alternates
    candidate_sitemaps = []
    candidate_sitemaps += robots_sitemaps(root)          # robots.txt entries
    candidate_sitemaps += alternate_sitemap_paths(root)  # alternates on common/bare subs
    candidate_sitemaps = list(dict.fromkeys(candidate_sitemaps))  # dedupe preserve order

    # 2) Fetch & parse, auto-include subdomains discovered inside sitemaps
    discovered = []
    seen_urls = set()
    subdomains_seen = set(initial_subs)  # track names like '', 'www', 'blog', ...
    urls_per_sub = {s: 0 for s in initial_subs}
    sitemaps_tried = 0
    sitemaps_ok = 0

    def host_is_within_root(hostname: str) -> bool:
        hostname = (hostname or "").lower()
        return hostname.endswith(root)

    def sub_of(hostname: str) -> str | None:
        """Return subdomain label ('' for bare root)."""
        hostname = (hostname or "").lower()
        if not hostname.endswith(root):
            return None
        if hostname == root:
            return ""
        suffix = "." + root
        label = hostname[:-len(suffix)]
        return label  # e.g., 'www', 'blog', 'docs'

    for sm_url in candidate_sitemaps:
        sitemaps_tried += 1
        resp = fetch(sm_url, timeout=8.0)
        if not resp:
            continue
        data = maybe_decompress(resp)
        urls, children = parse_sitemap(data)
        if urls or children:
            sitemaps_ok += 1

        # If sitemapindex, follow children (one level breadth)
        child_list = children.copy()
        # also detect new subdomains mentioned and enqueue their root sitemaps
        for loc in child_list:
            try:
                h = urlparse(loc).netloc.lower()
                if host_is_within_root(h):
                    lbl = sub_of(h)
                    if lbl is not None and lbl not in subdomains_seen:
                        subdomains_seen.add(lbl)
                        urls_per_sub.setdefault(lbl, 0)
                        for pth in ("/sitemap.xml", "/sitemap_index.xml"):
                            candidate = f"https://{h}{pth}"
                            if candidate not in candidate_sitemaps:
                                candidate_sitemaps.append(candidate)
            except Exception:
                pass

        # add urls from this sitemap (cap per subdomain)
        def add_urls(url_list):
            for u in url_list:
                if u in seen_urls:
                    continue
                try:
                    p = urlparse(u)
                    host = (p.netloc or "").lower()
                    if not host_is_within_root(host):
                        continue
                    label = sub_of(host)
                    if label is None:
                        continue
                except Exception:
                    continue
                # cap per subdomain
                current = urls_per_sub.get(label, 0)
                if current >= per_sub_cap:
                    continue
                seen_urls.add(u)
                discovered.append(u)
                urls_per_sub[label] = current + 1

        add_urls(urls)

        # follow child sitemaps
        for child in child_list:
            child_resp = fetch(child, timeout=8.0)
            if not child_resp:
                continue
            child_data = maybe_decompress(child_resp)
            u2, _ = parse_sitemap(child_data)
            add_urls(u2)

    stats = {
        "subdomains": sorted([lbl if lbl else "(root)" for lbl in set(urls_per_sub.keys())]),
        "urls_per_sub": urls_per_sub,
        "sitemaps_tried": sitemaps_tried,
        "sitemaps_ok": sitemaps_ok,
        "total_urls": len(discovered),
    }
    return discovered, stats

# ---------- Suggest URLs to keywords (RESTORED) ----------
def suggest_urls_for_keywords(df: pd.DataFrame, kw_col: str | None, only_eligible: bool,
                              min_score: float, pages: list[dict]) -> pd.DataFrame:
    if not pages or kw_col is None:
        df["Suggested URL"] = ""
        df["URL Match Score"] = ""
        return df

    # Precompute category lists & keyword tokens once
    cat_lists = df["Category"].fillna("").apply(
        lambda s: [c.strip() for c in str(s).split(",") if c.strip()] if s else ["SEO"]
    )
    kw_tokens_series = (df[kw_col] if kw_col else pd.Series([""] * len(df))).astype(str).apply(tokenize)

    mask = df["Eligible"].eq("Yes") if only_eligible and "Eligible" in df.columns else pd.Series([True]*len(df), index=df.index)

    suggested, scores = [], []
    for idx, row in df.iterrows():
        if not mask.loc[idx]:
            suggested.append("")
            scores.append("")
            continue
        kw_tokens = kw_tokens_series.loc[idx]
        keyword = str(row.get(kw_col, "")).strip()
        if not keyword or not kw_tokens:
            suggested.append("")
            scores.append("")
            continue
        categories = cat_lists.loc[idx] if isinstance(cat_lists.loc[idx], list) else ["SEO"]

        best_url, best_score = "", 0.0
        for p in pages:
            s = score_keyword_to_page(keyword, categories, p["tokens"])
            if s > best_score:
                best_url, best_score = p["url"], s

        if best_score >= min_score:
            suggested.append(best_url)
            scores.append(round(float(best_score), 3))
        else:
            suggested.append("")
            scores.append("")

    df["Suggested URL"] = suggested
    df["URL Match Score"] = scores
    return df

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

    # Order columns
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

# ---------- Site mapping (sitemap discovery only) ----------
with st.expander("Site mapping (optional): map keywords to URLs discovered from sitemap(s)"):
    st.markdown(
        "<div style='background:#FEF9C3;border:1px solid #FDE68A;padding:10px;border-radius:8px;color:#000;'>"
        "<strong>Heads up:</strong> We discover URLs from sitemap(s) across common subdomains and "
        "<strong>cap at 500 pages per subdomain</strong>. Need more? Contact <em>OutrankIQ</em>."
        "</div>",
        unsafe_allow_html=True
    )
    domain = st.text_input("Domain (e.g., example.com)", "")
    extra_subs_raw = st.text_input("Extra subdomains (optional, comma-separated)", "")
    extra_subs = [s.strip() for s in extra_subs_raw.split(",")] if extra_subs_raw else []
    only_eligible = st.checkbox("Only assign eligible keywords", value=True)
    MIN_MATCH_SCORE = 0.25  # fixed threshold

    discovered_urls = []
    discovery_stats = {}
    if domain.strip():
        with st.spinner("Discovering URLs from sitemap(s)..."):
            try:
                discovered_urls, discovery_stats = discover_urls_from_sitemaps(
                    domain=domain, extra_subs=extra_subs, per_sub_cap=500
                )
            except Exception:
                discovered_urls, discovery_stats = [], {}
        st.success(f"Discovered {len(discovered_urls)} URL(s) from sitemap(s).")
    else:
        st.info("Enter a domain to begin discovery.")

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
    if vol_col is None:
        missing.append("Volume")
    if kd_col is None:
        missing.append("Keyword Difficulty")

    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
    else:
        # Clean numbers (commas, spaces, percents)
        df[vol_col] = df[vol_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[kd_col] = df[kd_col].astype(str).str.replace(r"[,\s]", "", regex=True).str.replace("%", "", regex=False)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

        scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

        # Build pages from discovery — fast parallel fetch+parse
        pages = []
        fetch_success = 0
        usable_pages = 0
        if discovered_urls:
            with st.spinner("Fetching page content & extracting signals (parallel)…"):
                max_workers = min(32, max(4, os.cpu_count() * 2 if os.cpu_count() else 8))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(make_page_obj, u, None) for u in discovered_urls]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            page = f.result()
                            pages.append(page)
                            # diagnostics
                            fetch_success += 1
                            if page.get("tokens", {}).get("all"):
                                usable_pages += 1
                        except Exception:
                            pass

        # Apply URL suggestions
        scored = suggest_urls_for_keywords(
            scored, kw_col=kw_col, only_eligible=only_eligible,
            min_score=MIN_MATCH_SCORE, pages=pages
        )

        # --------- Diagnostics summary (tiny box) ---------
        try:
            total_keywords = len(scored)
            eligible_keywords = int(scored["Eligible"].eq("Yes").sum()) if "Eligible" in scored.columns else total_keywords
            assigned_keywords = int(scored["Suggested URL"].ne("").sum())

            subdomains_list = discovery_stats.get("subdomains", [])
            urls_per_sub = discovery_stats.get("urls_per_sub", {})
            sitemaps_tried = discovery_stats.get("sitemaps_tried", 0)
            sitemaps_ok = discovery_stats.get("sitemaps_ok", 0)
            total_discovered = discovery_stats.get("total_urls", len(discovered_urls))

            subdomains_display = ", ".join(subdomains_list) if subdomains_list else "—"

            st.markdown(
                f"""
<div style='background:#F3F4F6; border:1px solid #E5E7EB; padding:12px; border-radius:8px; margin-top:8px;'>
  <strong>🔍 Crawl & Match Summary</strong><br>
  • Subdomains discovered: <strong>{len(subdomains_list)}</strong> ({subdomains_display})<br>
  • URLs discovered (capped at 500 per subdomain): <strong>{total_discovered}</strong><br>
  • Sitemaps tried: <strong>{sitemaps_tried}</strong> &nbsp;|&nbsp; OK: <strong>{sitemaps_ok}</strong><br>
  • Pages fetched successfully: <strong>{fetch_success}</strong> &nbsp;|&nbsp; Usable content: <strong>{usable_pages}</strong><br>
  • Keywords eligible for strategy: <strong>{eligible_keywords}</strong> / {total_keywords}<br>
  • Keywords assigned a Suggested URL: <strong>{assigned_keywords}</strong>
</div>
""",
                unsafe_allow_html=True
            )
        except Exception:
            pass

        # Info banners (existing)
        invalid_rows = scored["Reason"].eq("Invalid Volume/KD").sum()
        below_min_rows = scored["Reason"].str.startswith("Below min volume").sum()
        if invalid_rows or below_min_rows:
            msgs = []
            if below_min_rows:
                msgs.append(f"{below_min_rows} below minimum volume for '{scoring_mode}' ({MIN_VALID_VOLUME}).")
            if invalid_rows:
                msgs.append(f"{invalid_rows} with invalid Volume/KD.")
            st.info("Some rows were not eligible: " + " ".join(msgs))
        if pages:
            assigned = scored["Suggested URL"].ne("").sum()
            st.success(f"URL mapping done. {assigned} keyword(s) received a Suggested URL. (Min match score {MIN_MATCH_SCORE})")

        # ---------- CSV DOWNLOAD (sorted: Yes first, KD ↑ then Volume ↓) ----------
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [
            vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category",
            "Suggested URL", "URL Match Score"
        ]
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

        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download scored CSV",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            help="Sorted by eligibility (Yes first), KD ascending, Volume descending"
        )

        # Optional preview (same sorting; colorized Score/Tier cells only; NO Color column shown)
        if st.checkbox("Preview first 10 rows (optional)", value=False):
            preview_df = scored.copy()
            preview_df["Strategy"] = scoring_mode
            preview_df["_EligibleSort"] = preview_df["Eligible"].map({"Yes": 1, "No": 0}).fillna(0)
            preview_df = preview_df.sort_values(
                by=["_EligibleSort", kd_col, vol_col],
                ascending=[False, True, False],
                kind="mergesort"
            ).drop(columns=["_EligibleSort"])

            def _row_style(row):
                color = COLOR_MAP.get(int(row.get("Score", 0)) if pd.notna(row.get("Score", 0)) else 0, "#9ca3af")
                return [
                    ("background-color: " + color + "; color: black;") if c in ("Score", "Tier") else ""
                    for c in row.index
                ]

            preview_cols = export_cols  # same columns as CSV
            styled = preview_df[preview_cols].head(10).style.apply(_row_style, axis=1)
            st.dataframe(styled, use_container_width=True)

st.markdown("---")
st.caption("© 2025 OutrankIQ • Select from three scoring strategies to target different types of keyword opportunities.")
