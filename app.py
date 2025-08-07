import io
import pandas as pd
import streamlit as st
from datetime import datetime

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
    MIN_VALID_VOLUME = 1501
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

# ---------- Scoring ----------
def calculate_score(volume: float, kd: float) -> int:
    # Both values are expected to be numeric already (cleaned below)
    if pd.isna(volume) or pd.isna(kd):
        return 0

    # Enforce thresholds / bounds
    if volume < MIN_VALID_VOLUME:
        return 0

    # Clamp KD to 0–100 to avoid out-of-range values breaking buckets
    kd = max(0.0, min(100.0, float(kd)))

    for low, high, score in KD_BUCKETS:
        if low <= kd <= high:
            return score
    return 0

def add_score_columns(df: pd.DataFrame, volume_col: str, kd_col: str) -> pd.DataFrame:
    out = df.copy()
    out["Score"] = [calculate_score(v, k) for v, k in zip(out[volume_col], out[kd_col])]
    out["Tier"] = out["Score"].map(LABEL_MAP).fillna("Not rated")
    out["Color"] = out["Score"].map(COLOR_MAP).fillna("#9ca3af")
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
        if vol_val < MIN_VALID_VOLUME:
            st.warning(
                f"The selected strategy requires a minimum search volume of {MIN_VALID_VOLUME}. "
                f"Please enter a volume that meets the threshold."
            )
        label = LABEL_MAP.get(sc, "Not rated")
        color = COLOR_MAP.get(sc, "#9ca3af")
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
    {
        "Keyword": ["best running shoes", "seo tools", "crm software"],
        "Volume": [5400, 880, 12000],
        "KD": [38, 72, 18],
    }
)
with st.expander("See example CSV format"):
    st.dataframe(example, use_container_width=True)

# ---------- Robust CSV reader + numeric cleaning ----------
if uploaded is not None:
    raw = uploaded.getvalue()

    def try_read(bytes_data: bytes) -> pd.DataFrame:
        trials = [
            {"encoding": None, "sep": None, "engine": "python"},     # let pandas infer
            {"encoding": "utf-8", "sep": None, "engine": "python"},
            {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
            {"encoding": "ISO-8859-1", "sep": None, "engine": "python"},
            {"encoding": "cp1252", "sep": None, "engine": "python"},
            {"encoding": "utf-16", "sep": None, "engine": "python"},
            {"encoding": None, "sep": ",", "engine": "python"},      # force comma
            {"encoding": None, "sep": "\t", "engine": "python"},     # TSV fallback
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

    # Try to find relevant columns with flexible names
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
        # ---- CLEAN NUMBERS so scoring is reliable ----
        # Handle commas, spaces, and percent signs (e.g., "1,500", "38%")
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

        # Convert to numeric; invalid parse -> NaN (becomes Not rated)
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce").clip(lower=0, upper=100)

        # Show a small info if some rows couldn't be parsed
        invalid_count = df[vol_col].isna().sum() + df[kd_col].isna().sum()
        if invalid_count:
            st.info("Some rows had non-numeric Volume/KD (commas/percents are okay). Unreadable values will be 'Not rated'.")

        scored = add_score_columns(df, vol_col, kd_col)

        # Reorder: Keyword (if present), Volume, KD, Score, Tier, then any remaining columns
        ordered = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier"]
        remaining = [c for c in scored.columns if c not in ordered + ["Color"]]
        scored = scored[ordered + remaining + ["Color"]]  # keep Color at end for preview styling

        st.success("Scoring complete")

        # ---------- CSV DOWNLOAD ONLY (no table shown by default) ----------
        filename_base = f"outrankiq_{scoring_mode.lower().replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        export_cols = [c for c in scored.columns if c != "Color"]
        export_df = scored[export_cols]

        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")  # BOM helps Excel open UTF-8 cleanly
        st.download_button(
            label="⬇️ Download scored CSV",
            data=csv_bytes,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            help="CSV with Score and Tier added"
        )

        # Optional tiny preview (off by default) so you don’t leak full data on screen
        if st.checkbox("Preview first 10 rows (optional)", value=False):
            def _row_style(row):
                # color Score/Tier cells based on computed Color
                style = []
                for col in row.index:
                    if col in ("Score", "Tier"):
                        style.append(f"background-color: {row.get('Color', '#9ca3af')}; color: black;")
                    else:
                        style.append("")
                return style

            st.dataframe(
                scored.head(10).style.apply(_row_style, axis=1).hide(axis="columns", subset=["Color"]),
                use_container_width=True
            )

st.markdown("---")
st.caption("© 2025 OutrankIQ • Select from three scoring strategies to target different types of keyword opportunities.")
