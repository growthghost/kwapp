""import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OutrankIQ", page_icon="🔎", layout="centered")

st.title("OutrankIQ")
st.caption("Score keywords by Search Volume (A) and Keyword Difficulty (B) — with selectable scoring strategies.")

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

strategy_descriptions = {
    "Low Hanging Fruit": "Focus on easy-to-rank keywords with low difficulty and moderate volume.",
    "In The Game": "Target mid-range keywords that require some authority and effort.",
    "Competitive": "Aim for high-volume, high-difficulty terms used by top-tier competitors."
}

scoring_mode = st.selectbox("Choose Scoring Strategy", ["Low Hanging Fruit", "In The Game", "Competitive"])
st.markdown(f"<div style='background-color:#f1f5f9; padding:10px; border-left:6px solid #1e3a8a; margin-bottom:10px;'>• <strong>{scoring_mode}</strong>: {strategy_descriptions[scoring_mode]}</div>", unsafe_allow_html=True)

with st.expander("⚙️ Settings", expanded=False):
    if scoring_mode == "Low Hanging Fruit":
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
    elif scoring_mode == "Competitive":
        MIN_VALID_VOLUME = 10000
        KD_BUCKETS = [
            (83, 100, 6),
            (67, 82, 5),
            (51, 66, 4),
            (35, 50, 3),
            (18, 34, 2),
            (0, 17, 1),
        ]
    elif scoring_mode == "In The Game":
        MIN_VALID_VOLUME = 1000
        KD_BUCKETS = [
            (70, 100, 6),
            (55, 69, 5),
            (40, 54, 4),
            (25, 39, 3),
            (10, 24, 2),
            (0, 9, 1),
        ]

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
        label = LABEL_MAP.get(sc, "Not rated")
        color = COLOR_MAP.get(sc, "#9ca3af")
        st.markdown(f"""
            <div style='background-color:{color}; padding:16px; border-radius:8px; text-align:center;'>
                <span style='font-size:22px; font-weight:bold; color:#000;'>Score: {sc} • Tier: {label}</span>
            </div>
        """, unsafe_allow_html=True)

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

        def highlight_scores(row):
            style = []
            for col in row.index:
                if col == "Score" or col == "Tier":
                    style.append(f"background-color: {row['Color']}; color: black;")
                else:
                    style.append("")
            return style

        styled_df = scored.style.apply(highlight_scores, axis=1)
        st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        buff = io.BytesIO()
        scored.to_csv(buff, index=False)
        st.download_button("Download scored CSV", data=buff.getvalue(), file_name="scored_keywords.csv", mime="text/csv")

st.markdown("---")
st.caption("© 2025 OutrankIQ • Select from three scoring strategies to target different types of keyword opportunities.")
