### START app.py — PART 1 / 6

import io
import re
import hashlib
from typing import Optional, List
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from datetime import datetime

from mapping import run_mapping
from crawler import fetch_profiles

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

.stApp {{ background-color: var(--bg); }}
html, body, [class^="css"], [class*=" css"] {{ color: var(--ink) !important; }}
h1, h2, h3, h4, h5, h6 {{ color: var(--ink) !important; }}

.oiq-header {{
  background: var(--accent);
  color: #ffffff;
  padding: 22px 20px;
  margin-bottom: 16px;
}}
.oiq-bleed {{
  margin-left: calc(50% - 50vw);
  margin-right: calc(50% - 50vw);
  width: 100vw;
}}
.oiq-header-inner {{
  max-width: 1000px;
  margin: 0 auto;
  padding-left: 16px;
}}
.oiq-title {{
  font-size: 40px;
  font-weight: 800;
}}
.oiq-sub {{
  margin-top: 6px;
  font-size: 16px;
}}

.stButton > button, .stDownloadButton > button {{
  background-color: var(--accent) !important;
  color: var(--ink) !important;
  font-weight: 700 !important;
}}

.oiq-footer {{
  color: var(--ink);
  font-size: 13px;
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
    <div class="oiq-sub">
      Score keywords by Search Volume (A) and Keyword Difficulty (B)
    </div>
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

LABEL_MAP = {
    6: "Elite",
    5: "Excellent",
    4: "Good",
    3: "Fair",
    2: "Low",
    1: "Very Low",
    0: "Not rated",
}

### END app.py — PART 1 / 6

### START app.py — PART 2 / 6

strategy_descriptions = {
    "Low Hanging Fruit": (
        "Keywords that can be used to rank quickly with minimal effort. "
        "Ideal for new content or low-authority sites."
    ),
    "In The Game": (
        "Moderate difficulty keywords that are within reach for growing sites."
    ),
    "Competitive": (
        "High-volume, high-difficulty keywords dominated by authoritative domains."
    ),
}

# ---------- Strategy ----------
scoring_mode = st.selectbox(
    "Choose Scoring Strategy",
    ["Low Hanging Fruit", "In The Game", "Competitive"],
)

# Reset mapping state on strategy switch
if "last_strategy" not in st.session_state:
    st.session_state["last_strategy"] = scoring_mode

if st.session_state.get("last_strategy") != scoring_mode:
    st.session_state["last_strategy"] = scoring_mode
    # Reset ONLY the mapping lifecycle keys
    for k in [
        "map_signature",
        "map_ready",
        "mapping_running",
        "map_result_df",
        "crawl_signals",
    ]:
        st.session_state.pop(k, None)

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
  <div style="font-size:13px;">Minimum Search Volume Required:
    <strong>{MIN_VALID_VOLUME}</strong>
  </div>
  <strong>{scoring_mode}</strong><br>
  <span>{strategy_descriptions[scoring_mode]}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Category Tagging ----------
AIO_PAT = re.compile(
    r"\b(what is|define|definition|explain|overview|guide to|examples of|types of)\b",
    re.I,
)
AEO_PAT = re.compile(
    r"^\s*(who|what|when|where|why|how|can|is|are|does|do)\b",
    re.I,
)
VEO_PAT = re.compile(
    r"\b(near me|nearby|local|open now|directions|call now)\b",
    re.I,
)
GEO_PAT = re.compile(
    r"\b(how to|steps to|checklist|framework|template|workflow|process)\b",
    re.I,
)
SXO_PAT = re.compile(
    r"\b(best|compare|vs\.?|pricing|reviews|cheap|affordable)\b",
    re.I,
)
LLM_PAT = re.compile(
    r"\b(chatgpt|gpt|llm|prompt engineering|ai search|answer engine)\b",
    re.I,
)

CATEGORY_ORDER = ["SEO", "AIO", "VEO", "GEO", "AEO", "SXO", "LLM"]

def categorize_keyword(kw: str) -> List[str]:
    if not isinstance(kw, str) or not kw.strip():
        return ["SEO"]

    text = kw.lower()
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

def add_scoring_columns(
    df: pd.DataFrame,
    volume_col: str,
    kd_col: str,
    kw_col: Optional[str],
) -> pd.DataFrame:
    out = df.copy()

    out["Score"] = [
        calculate_score(v, k)
        for v, k in zip(out[volume_col], out[kd_col])
    ]

    out["Eligible"] = out["Score"].apply(lambda s: "Yes" if s > 0 else "No")
    out["Tier"] = out["Score"].map(LABEL_MAP).fillna("Not rated")

    kw_series = out[kw_col] if kw_col else pd.Series([""] * len(out))
    out["Category"] = [
        ", ".join(categorize_keyword(str(k)))
        for k in kw_series
    ]

    return out

### END app.py — PART 2 / 6
### START app.py — PART 3 / 6

# ---------- Single Keyword ----------
st.subheader("Single Keyword Score")

with st.form("single_keyword_form"):
    c1, c2 = st.columns(2)
    with c1:
        vol_val = st.number_input(
            "Search Volume (A)",
            min_value=0,
            step=10,
            value=0,
        )
    with c2:
        kd_val = st.number_input(
            "Keyword Difficulty (B)",
            min_value=0,
            step=1,
            value=0,
        )

    if st.form_submit_button("Calculate Score"):
        score = calculate_score(vol_val, kd_val)
        tier = LABEL_MAP.get(score, "Not rated")
        st.markdown(
            f"""
            <div style="padding:16px;border-radius:12px;
                        background:#f1f5f9;text-align:center;">
              <div style="font-size:22px;font-weight:700;">
                Score: {score}
              </div>
              <div style="font-size:16px;">
                Tier: {tier}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if vol_val < MIN_VALID_VOLUME:
            st.warning(
                f"This strategy requires a minimum volume of {MIN_VALID_VOLUME}."
            )

st.markdown("---")
st.subheader("Bulk Scoring (CSV Upload)")

# ---------- User URLs ----------
url_text = st.text_area(
    "Key URLs for mapping (one per line, up to 10)",
    placeholder="https://example.com/page-1\nhttps://example.com/page-2",
    help="Only these URLs will be crawled and eligible for keyword mapping.",
)

urls: List[str] = []
if url_text.strip():
    urls = [u.strip() for u in url_text.splitlines() if u.strip()]
    urls = urls[:10]

st.session_state["user_mapping_urls"] = tuple(urls)

# ---------- CSV Upload ----------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    raw = uploaded.getvalue()

    def try_read_csv(data: bytes) -> pd.DataFrame:
        trials = [
            {"encoding": None},
            {"encoding": "utf-8"},
            {"encoding": "utf-8-sig"},
            {"encoding": "ISO-8859-1"},
            {"encoding": "cp1252"},
            {"encoding": "utf-16"},
        ]
        last_err = None
        for t in trials:
            try:
                return pd.read_csv(io.BytesIO(data), **t)
            except Exception as e:
                last_err = e
        raise last_err

    try:
        df = try_read_csv(raw)
    except Exception:
        st.error("Could not read the CSV file.")
        st.stop()

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
        st.stop()

    # Clean numeric columns
    df[vol_col] = (
        df[vol_col]
        .astype(str)
        .str.replace(r"[,\s]", "", regex=True)
        .str.replace("%", "", regex=False)
    )
    df[kd_col] = (
        df[kd_col]
        .astype(str)
        .str.replace(r"[,\s]", "", regex=True)
        .str.replace("%", "", regex=False)
    )

    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce").clip(0, 100)

    # Apply scoring
    scored = add_scoring_columns(df, vol_col, kd_col, kw_col)

    base_cols = [
        kw_col,
        vol_col,
        kd_col,
        "Score",
        "Tier",
        "Eligible",
        "Category",
    ]

    export_df = scored[base_cols].copy()
    export_df["Strategy"] = scoring_mode

    export_df["_EligibleSort"] = (
        export_df["Eligible"].map({"Yes": 1, "No": 0}).fillna(0)
    )
    export_df = (
        export_df.sort_values(
            by=["_EligibleSort", kd_col, vol_col],
            ascending=[False, True, False],
            kind="mergesort",
        )
        .drop(columns=["_EligibleSort"])
    )

### END app.py — PART 3 / 6

### START app.py — PART 4 / 6

    # ---------- Mapping signature ----------
    sig_df = export_df[[kw_col, vol_col, kd_col]].copy()
    sig_csv = sig_df.fillna("").astype(str).to_csv(index=False)

    user_urls_for_sig = st.session_state.get("user_mapping_urls") or ()
    first_for_sig = user_urls_for_sig[0] if user_urls_for_sig else ""

    base_norm = ""
    if first_for_sig:
        try:
            p = urlparse(first_for_sig.strip())
            if p.scheme and p.netloc:
                base_norm = f"{p.scheme}://{p.netloc}".lower()
            else:
                base_norm = first_for_sig.strip().lower()
        except Exception:
            base_norm = first_for_sig.strip().lower()

    sig_base = (
        f"site-map-v14-run_mapping_only|"
        f"{base_norm}|{scoring_mode}|{kw_col}|{vol_col}|{kd_col}|{len(export_df)}"
    )

    curr_signature = hashlib.md5(
        (sig_base + "\n" + sig_csv).encode("utf-8")
    ).hexdigest()

    # Invalidate previous mapping if inputs changed
    if st.session_state.get("map_signature") != curr_signature:
        st.session_state["map_ready"] = False
        st.session_state.pop("map_result_df", None)
        st.session_state.pop("crawl_signals", None)

    # ---------- DEBUG (temporary) ----------
    debug_mapping = st.checkbox("Debug mapping (temporary)", value=False)

    # ---------- Mapping button ----------
    user_urls_for_btn = st.session_state.get("user_mapping_urls") or ()
    can_map = len(user_urls_for_btn) > 0

    map_btn = st.button(
        "Map keywords to site",
        type="primary",
        disabled=not can_map,
        help="Crawls the supplied URLs and maps keywords using crawl signals.",
    )

    if map_btn and not st.session_state.get("mapping_running", False):
        st.session_state["mapping_running"] = True

        loader = st.empty()
        loader.markdown(
            """
            <div class="oiq-loader">
              <div class="oiq-spinner"></div>
              <div class="oiq-loader-text">
                Crawling URLs & mapping keywords…
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Running keyword mapping…"):
            url_list = list(user_urls_for_btn)[:10]

            # 1️⃣ Crawl user URLs (single authority)
            profiles = fetch_profiles(url_list)

            st.session_state["crawl_signals"] = {
                p["url"]: p
                for p in profiles
                if isinstance(p, dict) and p.get("url")
            }

            page_signals_by_url = st.session_state["crawl_signals"]

            # ---- DEBUG prints (prove what we have) ----
            if debug_mapping:
                st.write("profiles:", len(profiles))
                st.write("crawl_signals:", len(page_signals_by_url))
                st.write("signal sample keys:", list(page_signals_by_url.keys())[:3])
                if page_signals_by_url:
                    first_sig = next(iter(page_signals_by_url.values()))
                    st.write(
                        "token counts (first page):",
                        {
                            k: len(first_sig.get(k, []))
                            for k in ["slug_tokens", "title_tokens", "h1_tokens", "h2h3_tokens", "meta_tokens"]
                        },
                    )

            if not page_signals_by_url:
                st.error("No crawl data found. Please click Map keywords to site again.")
                st.session_state["mapping_running"] = False
                st.stop()

            # 2️⃣ Ensure Map URL column exists
            if "Map URL" not in export_df.columns:
                export_df["Map URL"] = ""

            # 3️⃣ Run mapping (single authority)
            mapped_df = run_mapping(
                df=export_df,
                page_signals_by_url=page_signals_by_url,
            )

            if debug_mapping and isinstance(mapped_df, pd.DataFrame) and "Map URL" in mapped_df.columns:
                st.write("mapped count:", int((mapped_df["Map URL"].astype(str).str.len() > 0).sum()))

            st.session_state["map_result_df"] = mapped_df
            st.session_state["map_signature"] = curr_signature
            st.session_state["map_ready"] = True

        loader.empty()
        st.session_state["mapping_running"] = False

### END app.py — PART 4 / 6


### START app.py — PART 5 / 6

    # ---------- Apply mapping results ----------
    if (
        st.session_state.get("map_ready")
        and st.session_state.get("map_signature") == curr_signature
    ):
        final_df = st.session_state.get("map_result_df")

        # Use mapped_df as the single download source
        if isinstance(final_df, pd.DataFrame) and len(final_df) == len(export_df):
            export_df = final_df.copy()
        else:
            # If mapping isn't valid, force download to stay locked
            can_download = False

        # Ensure Map URL exists (mapping should create it, but keep safe)
        if "Map URL" not in export_df.columns:
            export_df["Map URL"] = ""

        # Clear mapped URL if keyword is not eligible
        export_df.loc[export_df["Eligible"] != "Yes", "Map URL"] = ""

        # Only allow download if mapped_df was valid
        can_download = isinstance(final_df, pd.DataFrame) and len(final_df) == len(export_df)

    else:
        export_df["Map URL"] = ""
        can_download = False

    # ---------- Final export ----------
    export_cols = [
        kw_col,
        vol_col,
        kd_col,
        "Score",
        "Tier",
        "Eligible",
        "Category",
        "Strategy",
        "Map URL",
    ]

    export_df = export_df[export_cols]

    filename_base = (
        f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    )

    csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download scored CSV",
        data=csv_bytes,
        file_name=f"{filename_base}.csv",
        mime="text/csv",
        disabled=not can_download,
        help="Mapping runs only after clicking 'Map keywords to site'",
    )

### END app.py — PART 5 / 6

### START app.py — PART 6 / 6

# ---------- Footer ----------
st.markdown(
    "<div class='oiq-footer'>© 2026 OutrankIQ</div>",
    unsafe_allow_html=True,
)

### END app.py — PART 6 / 6
