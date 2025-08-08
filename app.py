import io
import re
import pandas as pd
import streamlit as st
from datetime import datetime
from urllib.parse import urlparse

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
    <div style='margin-bottom:6px; font-size:13px; color:#f0f9ff;'>
        Minimum Search Volume Required: <strong>{MIN_VALID_VOLUME}</strong>
    </div>
    <strong style='color:#ffffff; font-size:18px;'>{scoring_mode}</strong><br>
    <span style='color:#f8fafc; font-size:15px;'>{strategy_descriptions[scoring_mode]}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Category tagging (multi-label) ----------
CATEGORY_ORDER = ["SEO", "AIO", "VEO", "GEO", "AEO", "SXO", "LLM"]
AIO_PAT    = re.compile(r"\b(what is|what's|define|definition|how to|step[- ]?by[- ]?step|tutorial|guide)\b", re.I)
AEO_PAT    = re.compile(r"^\s*(who|what|when|where|why|how|which|can|should)\b", re.I)
VEO_PAT    = re.compile(r"\b(near me|open now|closest|call now|directions|ok google|alexa|siri|hey google)\b", re.I)
GEO_PAT    = re.compile(r"\b(how to|best way to|steps? to|examples? of|checklist|framework|template)\b", re.I)
SXO_PAT    = re.compile(r"\b(best|top|compare|comparison|vs\.?|review|pricing|cost|cheap|free download|template|examples?)\b", re.I)
LLM_PAT    = re.compile(r"\b(prompt|prompting|prompt[- ]?engineering|chatgpt|gpt[- ]?\d|llm|rag|embedding|vector|few[- ]?shot|zero[- ]?shot)\b", re.I)

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

# ---------- URL mapping helpers ----------
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
    "SEO": set(),  # baseline
}

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def score_keyword_to_page(keyword: str, categories: list[str], page_tokens: set[str]) -> tuple[float, str]:
    kw_tokens = tokenize(keyword)
    base = jaccard(kw_tokens, page_tokens)
    boost = 0.0
    applied = []
    for c in categories:
        terms = CATEGORY_BOOST_TERMS.get(c, set())
        if terms and (terms & page_tokens):
            boost += 0.05  # small, stackable boosts
            applied.append(c)
    score = min(base + boost, 1.0)
    method = f"Jaccard{f' + Boost({\"/\".join(applied)})' if applied else ''}"
    return score, method

def prepare_menu_pages(menu_df: pd.DataFrame | None, menu_text: str) -> list[dict]:
    pages = []
    if menu_df is not None and not menu_df.empty:
        url_col = find_column(menu_df, ["url", "link", "href"])
        title_col = find_column(menu_df, ["title", "name", "label", "text"])
        if url_col is None:
            return pages
        for _, r in menu_df.iterrows():
            url = str(r[url_col]).strip()
            if not url:
                continue
            title = str(r[title_col]).strip() if title_col else None
            pages.append({
                "url": url,
                "title": title or url_to_title(url),
                "tokens": extract_page_tokens(url, title)
            })
    # Fallback to textarea list (one URL per line)
    if not pages:
        for line in (menu_text or "").splitlines():
            url = line.strip()
            if not url:
                continue
            pages.append({
                "url": url,
                "title": url_to_title(url),
                "tokens": extract_page_tokens(url, None)
            })
    return pages

def suggest_urls_for_keywords(df: pd.DataFrame, kw_col: str | None, only_eligible: bool,
                              min_score: float, max_per_url: int, pages: list[dict]) -> pd.DataFrame:
    if not pages or kw_col is None:
        df["Suggested URL"] = ""
        df["URL Match Score"] = ""
        df["URL Match Method"] = ""
        return df

    # Precompute category lists per row
    cat_lists = df["Category"].fillna("").apply(lambda s: [c.strip() for c in str(s).split(",") if c.strip()] if s else ["SEO"])

    # Optional: filter to eligible rows
    mask = df["Eligible"].eq("Yes") if only_eligible and "Eligible" in df.columns else pd.Series([True]*len(df), index=df.index)

    # Greedy assignment with per-URL cap
    counts = {p["url"]: 0 for p in pages}
    suggested = []
    scores = []
    methods = []

    for idx, row in df.iterrows():
        if not mask.loc[idx]:
            suggested.append("")
            scores.append("")
            methods.append("")
            continue
        kw = str(row.get(kw_col, "")).strip()
        if not kw:
            suggested.append("")
            scores.append("")
            methods.append("")
            continue
        categories = cat_lists.loc[idx] if isinstance(cat_lists.loc[idx], list) else ["SEO"]

        # Score against all pages
        best_url, best_score, best_method = "", 0.0, ""
        for p in pages:
            s, m = score_keyword_to_page(kw, categories, p["tokens"])
            if s > best_score:
                best_url, best_score, best_method = p["url"], s, m

        # Apply threshold and per-URL cap
        if best_score >= min_score and counts.get(best_url, 0) < max_per_url:
            counts[best_url] = counts.get(best_url, 0) + 1
            suggested.append(best_url)
            scores.append(round(float(best_score), 3))
            methods.append(best_method)
        else:
            suggested.append("")
            scores.append("")
            methods.append("")

    df["Suggested URL"] = suggested
    df["URL Match Score"] = scores
    df["URL Match Method"] = methods
    return df

# ---------- Scoring ----------
def calculate_score(volume: float, kd: float) -> int:
    """Return score 0-6, but ONLY if eligible (volume >= min)."""
    if pd.isna(volume) or pd.isna(kd):
        return 0
    if volume < MIN_VALID_VOLUME:
        return 0
    kd = max(0.0, min(100.0, float(kd)))  # clamp
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

# ---------- Site mapping (Option A) ----------
with st.expander("Site mapping (optional): map keywords to main menu URLs"):
    st.markdown("Provide your main menu URLs to associate each keyword with the best page.")
    colm1, colm2 = st.columns(2)
    with colm1:
        menu_csv = st.file_uploader("Upload menu CSV (columns: URL, Title optional)", type=["csv"], key="menu_csv")
    with colm2:
        max_per_url = st.number_input("Max keywords per URL", min_value=1, max_value=1000, value=50, step=1)
    menu_text = st.text_area("Or paste URLs (one per line)", height=120, placeholder="https://example.com/\nhttps://example.com/pricing\nhttps://example.com/blog\n...")
    only_eligible = st.checkbox("Only assign eligible keywords", value=True)
    min_match_score = st.slider("Minimum match score to assign", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

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

        # Prepare menu pages (CSV takes precedence; otherwise textarea)
        menu_df = None
        if menu_csv is not None:
            try:
                menu_df = try_read(menu_csv.getvalue())
            except Exception:
                st.warning("Could not read menu CSV; falling back to pasted URLs.")
                menu_df = None
        pages = prepare_menu_pages(menu_df, menu_text)

        # Apply URL suggestions
        scored = suggest_urls_for_keywords(
            scored, kw_col=kw_col, only_eligible=only_eligible,
            min_score=min_match_score, max_per_url=max_per_url, pages=pages
        )

        # Info banners
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
            st.success(f"URL mapping done. {assigned} keyword(s) received a Suggested URL.")

        # ---------- CSV DOWNLOAD (sorted: Yes first, KD ↑ then Volume ↓) ----------
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [
            vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category",
            "Suggested URL", "URL Match Score", "URL Match Method"
        ]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        # Sort: Eligible (Yes first) → KD ↑ → Volume ↓
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
