import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OutrankIQ", page_icon="🔎", layout="centered")

st.title("OutrankIQ")
st.caption("Score keywords by Search Volume (A) and Keyword Difficulty (B) — now with 6 equal scoring levels.")

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
    6: "#2ecc71",   # Bright Green
    5: "#a3e635",   # Lime
    4: "#facc15",   # Yellow
    3: "#fb923c",   # Orange
    2: "#f87171",   # Tomato
    1: "#ef4444",   # Red
    0: "#9ca3af",   # Gray
}

with st.expander("⚙️ Settings", expanded=False):
    MIN_VALID_VOLUME = st.number_input("Minimum valid Volume (A must be ≥ this to score)", min_value=0, value=100, step=50)
    KD_MIN = st.number_input("KD minimum", min_value=0, value=0, step=1)
    KD_MAX = st.number_input("KD maximum", min_value=1, value=100, step=1)
    LEVELS = 6
    total_span = max(1, KD_MAX - KD_MIN + 1)
    base = total_span // LEVELS
    remainder = total_span % LEVELS
    KD_BUCKETS = []
    start = KD_MIN
    score_val = LEVELS
    for i in range(LEVELS):
        width = base + (1 if i < remainder else 0)
        end = start + width - 1
        KD_BUCKETS.append((start, end, score_val))
        score_val -= 1
        start = end + 1
    st.write("KD buckets:")
    st.table(pd.DataFrame(KD_BUCKETS, columns=["KD From", "KD To", "Score"]))

def calculate_score(volume: float, kd: float) -> int:
    try:
        if pd.isna(volume) or pd.isna(kd):
            return 0
        volume = float(volume)
        kd = float(kd)
    except Exception:
        return 0
    if volume < MIN_VALID_VOLUME:
        return 0
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

st.subheader("Single Keyword Score")
with st.form("single"):
    col1, col2 = st.columns(2)
    with col1:
        vol_val = st.number_input("Search Volume (A)", min_value=0, step=10, value=0)
    with col2:
        kd_val = st.number_input("Keyword Difficulty (B)", min_value=0, step=1, value=0)
    if st.form_submit_button("Calculate Score"):
        sc = calculate_score(vol_val, kd_val)
        st.success(f"Score: **{sc}** • Tier: **{LABEL_MAP.get(sc, 'Not rated')}**")

st.markdown("---")
st.subheader("Bulk Scoring (CSV Upload)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
example = pd.DataFrame({"Keyword": ["best running shoes", "seo tools", "crm software"], "Volume": [5400, 880, 12000], "KD": [38, 72, 18]})
with st.expander("See example CSV format"):
    st.dataframe(example, use_container_width=True)
    st.download_button("Download example.csv", data=example.to_csv(index=False).encode("utf-8"), file_name="example.csv", mime="text/csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not read the file. Please upload a valid CSV.")
        st.stop()
    vol_col = find_column(df, ["volume", "search volume", "sv"])
    kd_col = find_column(df, ["kd", "difficulty", "keyword difficulty"])
    kw_col = find_column(df, ["keyword", "query", "term"])
    missing = []
    if vol_col is None: missing.append("Volume")
    if kd_col is None: missing.append("Keyword Difficulty")
    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
    else:
        scored = add_score_columns(df, vol_col, kd_col)
        ordered = ([kw_col] if kw_col else []) + [vol_col, kd_col, "Score", "Tier"]
        remaining = [c for c in scored.columns if c not in ordered]
        scored = scored[ordered + remaining]
        st.success("Scoring complete.")

        def style_row(row):
            return [f"background-color: {row['Color']}" if col in ["Score", "Tier"] else "" for col in row.index]

        st.dataframe(
            scored.style.apply(style_row, axis=1),
            use_container_width=True
        )

        buff = io.BytesIO()
        scored.to_csv(buff, index=False)
        st.download_button("Download scored CSV", data=buff.getvalue(), file_name="scored_keywords.csv", mime="text/csv")

st.markdown("---")
st.caption("© 2025 OutrankIQ • 6-level KD scoring with color-coded visual tiers. Adjust KD min/max and volume threshold in Settings.")
