import io
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
    "Low Hanging Fruit": "Keywords that can be used to rank quickly with minimal effort. Ideal for new content or low-authority sites. Try targeting long-tail keywords, create quick-win content, and build a few internal links.",
    "In The Game": "Moderate difficulty keywords that are within reach for growing sites. Focus on optimizing content, earning backlinks, and matching search intent to climb the ranks.",
    "Competitive": "High-volume, high-difficulty keywords dominated by authoritative domains. Requires strong content, domain authority, and strategic SEO to compete. Great for long-term growth."
}

scoring_mode = st.selectbox("Choose Scoring Strategy", ["Low Hanging Fruit", "In The Game", "Competitive"])

if scoring_mode == "Low Hanging Fruit":
    MIN_VALID_VOLUME = 10
    KD_BUCKETS = [
        (0, 15, 6),
        (16, 20, 5),
        (21, 25, 4),
        (26, 50, 3),
        (51, 75, 2),
        (76, 100, 1),
    ]
elif scoring_mode == "In The Game":
    MIN_VALID_VOLUME = 1500
    KD_BUCKETS = [
        (0, 30, 6),
        (31, 45, 5),
        (46, 60, 4),
        (61, 70, 3),
        (71, 80, 2),
        (81, 100, 1),
    ]
elif scoring_mode == "Competitive":
    MIN_VALID_VOLUME = 1501
    KD_BUCKETS = [
        (0, 40, 6),
        (41, 60, 5),
        (61, 75, 4),
        (76, 85, 3),
        (86, 95, 2),
        (96, 100, 1),
    ]
st.markdown(f"""
<div style='background: linear-gradient(to right, #3b82f6, #60a5fa); padding:16px; border-radius:8px; margin-bottom:16px;'>
    <div style='margin-bottom:6px; font-size:13px; color:#f0f9ff;'>Minimum Search Volume Required: <strong>{MIN_VALID_VOLUME}</strong></div>
    <strong style='color:#ffffff; font-size:18px;'>{scoring_mode}</strong><br>
    <span style='color:#f8fafc; font-size:15px;'>{strategy_descriptions[scoring_mode]}</span>
</div>
""", unsafe_allow_html=True)



    
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
        if vol_val < MIN_VALID_VOLUME:
            st.warning(f"The selected strategy requires a minimum search volume of {MIN_VALID_VOLUME}. Please enter a volume that meets the threshold.")
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
        try:
        df = pd.read_csv(uploaded)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded, encoding="ISO-8859-1")
        except Exception:
            st.error("Could not read the file. Please upload a valid CSV.")
            st.stop()
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
        st.success("Scoring complete 🎉")
st.balloons()

        with st.expander("Email the scored results"):
            email = st.text_input("Enter your email address")
            if st.button("Send CSV to Email"):
                st.info("This feature is coming soon! For now, please download using the button below.")

        def highlight_scores(row):
            style = []
            for col in row.index:
                if col == "Score" or col == "Tier":
                    style.append(f"background-color: {row['Color']}; color: black;")
                else:
                    style.append("")
            return style

        styled_df = scored.style.apply(highlight_scores, axis=1)
        # st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        buff = io.BytesIO()
        scored.to_csv(buff, index=False)
        st.download_button("Download scored CSV", data=buff.getvalue(), file_name="scored_keywords.csv", mime="text/csv")

st.markdown("---")
st.caption("© 2025 OutrankIQ • Select from three scoring strategies to target different types of keyword opportunities.")
