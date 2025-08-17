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

# ---------- Brand / Theme ----------
BRAND_BG = "#747474"     # background
BRAND_INK = "#242F40"    # blue/ink
BRAND_ACCENT = "#E1B000" # yellow
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
  --accent-rgb: 225,176,0;
  --light: {BRAND_LIGHT};
}}
/* App background */
.stApp {{ background-color: var(--bg); }}

/* Base text on dark bg */
html, body, [class^="css"], [class*=" css"] {{ color: var(--light) !important; }}

/* Headings */
h1, h2, h3, h4, h5, h6 {{ color: var(--light) !important; }}

/* Inputs / selects / numbers: white surface with ink text */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {{
  background-color: var(--light) !important;
  color: var(--ink) !important;
  border-radius: 8px !important;
}}

/* Hand cursor for select + number inputs (including +/-) */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stNumberInput button {{ cursor: pointer !important; }}

/* Selectbox: caret inside + yellow focus */
.stSelectbox div[data-baseweb="select"] > div {{
  border: 2px solid var(--light) !important;
  position: relative;
}}
.stSelectbox div[data-baseweb="select"]:focus-within > div {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
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

/* Number inputs: ensure YELLOW focus (never red) + blue steppers */
.stNumberInput input {{
  border: 2px solid var(--light) !important;
  outline: none !important;
}}
.stNumberInput input:focus,
.stNumberInput input:focus-visible {{
  outline: none !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
}}
.stNumberInput:focus-within input {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(var(--accent-rgb), .35) !important;
}}
.stNumberInput button {{
  background: var(--ink) !important;        /* blue default */
  color: #ffffff !important;
  border: 1px solid var(--ink) !important;
}}
.stNumberInput button:hover,
.stNumberInput button:active,
.stNumberInput button:focus-visible {{
  background: var(--accent) !important;     /* yellow on interaction */
  color: #000 !important;
  border-color: var(--accent) !important;
}}

/* File uploader dropzone */
[data-testid="stFileUploaderDropzone"] {{
  background: rgba(255,255,255,0.98);
  border: 2px dashed var(--accent);
}}
/* Text in uploader area is dark for readability */
[data-testid="stFileUploader"] * {{ color: var(--ink) !important; }}

/* “Browse files” button: blue default, transparent on hover (like Calculate) */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] label,
[data-testid="stFileUploaderDropzone"] [role="button"] {{
  background-color: var(--ink) !important;  /* #242F40 */
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
  background-color: transparent !important; /* transparent hover */
  color: var(--ink) !important;
  border-color: var(--ink) !important;
}}

/* Tables/readability */
.stDataFrame, .stDataFrame * , .stTable, .stTable * {{ color: var(--ink) !important; }}

/* Action buttons (download & calculate) — transparent on hover */
.stButton > button, .stDownloadButton > button {{
  background-color: var(--accent) !important;
  color: var(--ink) !important;
  border: 2px solid var(--light) !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 0 rgba(0,0,0,.15);
  transition: background-color .15s ease, color .15s ease, border-color .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background-color: transparent !important; /* transparent on hover */
  color: var(--light) !important;
  border-color: var(--accent) !important;
}}

/* Strategy banner helper */
.info-banner {{
  background: linear-gradient(90deg, var(--ink) 0%, var(--accent) 100%);
  padding: 16px; border-radius: 12px; color: var(--light);
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Title + tagline ----------
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

# Used for the single-word color card
COLOR_MAP = {
    6: "#2ecc71",
    5: "#a3e635",
    4: "#facc15",
    3: "#fb923c",
    2: "#f87171",
    1: "#ef4444",
    0: "#9ca3af",
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
else:
    MIN_VALID_VOLUME = 3000
    KD_BUCKETS = [(0, 40, 6), (41, 60, 5), (61, 75, 4), (76, 85, 3), (86, 95, 2), (96, 100, 1)]

st.markdown(
    f"""
<div class="info-banner" style="margin-bottom:16px;">
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
AIO_PAT = re.compile(r"\b(what is|what's|define|definition|how to|step[- ]?by[- ]?step|tutorial|guide)\b", re.I)
AEO_PAT = re.compile(r"^\s*(who|what|when|where|why|how|which|can|should)\b", re.I)
VEO_PAT = re.compile(r"\b(near me|open now|closest|call now|directions|ok google|alexa|siri|hey google)\b", re.I)
GEO_PAT = re.compile(r"\b(how to|best way to|steps? to|examples? of|checklist|framework|template)\b", re.I)
SXO_PAT = re.compile(r"\b(best|top|compare|comparison|vs\.?|review|pricing|cost|cheap|free download|template|examples?)\b", re.I)
LLM_PAT = re.compile(r"\b(prompt|prompting|prompt[- ]?engineering|chatgpt|gpt[- ]?\d|llm|rag|embedding|vector|few[- ]?shot|zero[- ]?shot)\b", re.I)
CATEGORY_ORDER = ["SEO", "AIO", "VEO", "GEO", "AEO", "SXO", "LLM"]

def categorize_keyword(kw: str) -> list[str]:
    if not isinstance(kw: = str) or not kw.strip():
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
        if pd.isna(vol) or pd.isna(kd): return "No","Invalid Volume/KD"
        if vol < MIN_VALID_VOLUME: return "No", f"Below min volume for {scoring_mode} ({MIN_VALID_VOLUME})"
        return "Yes",""
    eligible, reason = zip(*(_eligibility_reason(v,k) for v,k in zip(out[volume_col], out[kd_col])))
    out["Eligible"] = list(eligible); out["Reason"] = list(reason)
    out["Score"] = [calculate_score(v,k) for v,k in zip(out[volume_col], out[kd_col])]
    out["Tier"]  = out["Score"].map(LABEL_MAP).fillna("Not rated")
    kw_series = out[kw_col] if kw_col else pd.Series([""]*len(out))
    out["Category"] = [", ".join(categorize_keyword(str(k))) for k in kw_series]
    ordered = ([kw_col] if kw_col else []) + [volume_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
    remaining = [c for c in out.columns if c not in ordered]
    return out[ordered + remaining]

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

uploaded = st.file_uploader("Upload CSV", type=["csv"])
example = pd.DataFrame({"Keyword":["best running shoes","seo tools","crm software"], "Volume":[5400,880,12000], "KD":[38,72,18]})
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
                return pd.read_csv(io.BytesIO(bytes_data), **{k:v for k,v in t.items() if v is not None})
            except Exception as e:
                last_err = e
        raise last_err

    try:
        df = try_read(raw)
    except Exception:
        st.error("Could not read the file. Please ensure it's a CSV (or TSV) exported from Excel/Sheets and try again.")
        st.stop()

    # Find relevant columns
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

        # ---------- CSV DOWNLOAD ----------
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        base_cols = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier", "Eligible", "Reason", "Category"]
        export_df = scored[base_cols].copy()
        export_df["Strategy"] = scoring_mode

        export_df["_EligibleSort"] = export_df["Eligible"].map({"Yes":1,"No":0}).fillna(0)
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

st.markdown("---")
st.caption(f"© {datetime.now().year} OutrankIQ")
