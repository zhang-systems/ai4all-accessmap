# AccessMap - Streamlit app
# Loads the trained models and FAISS indexes from notebook 03.
# Run from the project root: streamlit run app.py
#
# Folder layout this app expects:
#   accessmap-project/
#     app.py            <- this file
#     data/models/      <- rf_pmr.pkl, encoders, scaler
#     data/processed/   <- housing_clean.csv, embeddings, faiss indexes

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import faiss

st.set_page_config(page_title="AccessMap", layout="centered")

MODELS_DIR = "data/models"
PROCESSED_DIR = "data/processed"

# Thresholds behind our rule-based label (same as notebook 03)
WIDTH_MIN = 0.9        # meters - minimum obstacle-free width for a wheelchair
CURB_MAX = 0.06        # meters - maximum curb height a wheelchair can pass
CURB_MEDIAN_FILL = 0.06  # fill value notebook 03 used for missing curb heights

# Readable names for the model's input features (for the explanation box)
FRIENDLY_NAMES = {
    "obstacle_free_width_float": "Obstacle-free width",
    "curb_height_max": "Max curb height",
    "curb_height_missing": "Curb height missing? (flag)",
    "width_fill": "Width imputed? (flag)",
    "crossing": "Crossing (yes/no)",
    "length": "Segment length",
}

# Static satellite image of Amsterdam (PMR dataset region), served by Esri's
# public World Imagery REST endpoint. No API key required.
SATELLITE_BG_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/export?bbox=4.82,52.33,4.98,52.41&bboxSR=4326&size=1920,1200"
    "&format=png&f=image"
)


# ---------------------------------------------------------
# Theme: satellite background + professional dashboard styling
# ---------------------------------------------------------
def apply_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        /* Satellite map fills the page, fully visible - no dark scrim */
        .stApp {{
            background: url('{SATELLITE_BG_URL}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            padding-top: 2rem;
        }}

        /* Header card floats on top of the map */
        .am-hero {{
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.2rem;
            backdrop-filter: blur(6px);
            box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        }}

        .am-hero h1 {{
            color: #FFFFFF;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin-bottom: 0.25rem;
        }}

        .am-hero p {{
            color: #CBD5E1;
            font-size: 0.98rem;
            margin: 0;
        }}

        .am-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #94A3B8;
            padding: 0.15rem 0.65rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: rgba(15, 23, 42, 0.75);
            padding: 6px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(6px);
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 44px;
            border-radius: 6px;
            color: #CBD5E1;
            font-weight: 500;
            font-size: 0.9rem;
        }}

        .stTabs [aria-selected="true"] {{
            background: #1D4ED8 !important;
            color: white !important;
        }}

        /* Content area sits on top of the map like a dashboard panel */
        section[data-testid="stMain"] > div.block-container {{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 18px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.20);
        }}

        .am-panel h2, .am-panel h3 {{
            color: #0F172A;
            font-weight: 700;
        }}

        .am-panel p, .am-panel li {{
            color: #334155;
        }}

        .am-panel strong {{
            color: #1D4ED8;
        }}

        /* Inputs */
        .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
            border-radius: 6px !important;
        }}

        /* Buttons */
        .stButton > button {{
            background: #1D4ED8;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.3rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(29, 78, 216, 0.3);
            transition: background 0.15s ease;
        }}

        .stButton > button:hover {{
            background: #1E40AF;
        }}

        /* Alerts */
        div[data-testid="stAlert"] {{
            border-radius: 8px;
        }}

        /* Dataframe */
        div[data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow: hidden;
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="am-hero">
            <h1>AccessMap</h1>
            <p>Sidewalk accessibility: check a segment, search similar addresses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Loaders. Cached so files load once, not on every click.
# ---------------------------------------------------------
@st.cache_resource
def load_pmr_model():
    rf_pmr = joblib.load(f"{MODELS_DIR}/rf_pmr.pkl")
    crossing_encoder = joblib.load(f"{MODELS_DIR}/crossing_encoder.pkl")
    return rf_pmr, crossing_encoder


@st.cache_resource
def load_housing_search():
    housing = pd.read_csv(f"{PROCESSED_DIR}/housing_clean.csv")
    housing = housing[housing["sidewalk_ok"].isin(["yes", "no"])].reset_index(drop=True)
    index = faiss.read_index(f"{PROCESSED_DIR}/housing_faiss.index")
    embeddings = np.load(f"{PROCESSED_DIR}/housing_address_embeddings.npy").astype("float32")
    return housing, index, embeddings


def explain_factors(width, curb_val, curb_known, crossing):
    """Plain-language readout of how each input pushes the prediction.

    The two decisive features are width and curb height, because our label
    rule is built from them. The rest add context.
    """
    lines = []
    if width < WIDTH_MIN:
        lines.append(
            f"- **Width {width:.2f} m** is below the {WIDTH_MIN} m wheelchair "
            "minimum → pushes toward **not accessible**."
        )
    else:
        lines.append(
            f"- **Width {width:.2f} m** clears the {WIDTH_MIN} m wheelchair "
            "minimum → pushes toward **accessible**."
        )
    if curb_val > CURB_MAX:
        lines.append(
            f"- **Curb height {curb_val:.2f} m** is above the {CURB_MAX} m "
            "maximum a wheelchair can pass → pushes toward **not accessible**."
        )
    else:
        lines.append(
            f"- **Curb height {curb_val:.2f} m** is at or below the {CURB_MAX} m "
            "maximum → pushes toward **accessible**."
        )
    if not curb_known:
        lines.append(
            f"- Curb height wasn't measured, so the model used the dataset "
            f"median ({CURB_MEDIAN_FILL} m) and was told it's an estimate — "
            "treat this prediction with extra caution."
        )
    if crossing == "Yes":
        lines.append(
            "- This segment is a **crossing**; crossings in the data are "
            "slightly more likely to have curb problems."
        )
    return "\n".join(lines)


# ---------------------------------------------------------
# App
# ---------------------------------------------------------
apply_theme()
hero()

tab_pmr, tab_housing, tab_about = st.tabs(
    ["Sidewalk Checker (PMR)", "Address Search (Housing)", "About & Limits"]
)

# ---------------------------------------------------------
# Tab 1: PMR sidewalk checker
# ---------------------------------------------------------
with tab_pmr:
    st.markdown('<div class="am-panel">', unsafe_allow_html=True)
    st.header("Check a sidewalk segment")
    st.write(
        "Enter the segment's measurements. The model says if it's "
        "accessible for a wheelchair user."
    )

    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input(
            "Obstacle-free width (m)", min_value=0.0, max_value=10.0,
            value=1.2, step=0.1,
            help="Rule of thumb: 0.9m is the minimum for a wheelchair."
        )
        length = st.number_input(
            "Segment length (m)", min_value=0.0, max_value=500.0,
            value=2.0, step=0.5,
        )
        crossing = st.selectbox("Is it a crossing?", ["No", "Yes"])
    with col2:
        curb_known = st.checkbox("Curb height was measured", value=True)
        curb_height = st.number_input(
            "Max curb height (m)", min_value=0.0, max_value=0.5,
            value=0.02, step=0.01, disabled=not curb_known,
            help="Rule of thumb: 0.06m (6cm) is the max a wheelchair can pass."
        )
        width_fill = st.number_input(
            "Width fill", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
            help="0.0 if the width value was measured, not imputed."
        )

    if st.button("Check accessibility", type="primary"):
        try:
            rf_pmr, crossing_encoder = load_pmr_model()

            curb_val = curb_height if curb_known else CURB_MEDIAN_FILL
            row = pd.DataFrame([{
                "length": length,
                "obstacle_free_width_float": width,
                "crossing": crossing_encoder.transform([crossing])[0],
                "width_fill": width_fill,
                "curb_height_max": curb_val,
                "curb_height_missing": 0 if curb_known else 1,
            }])

            pred = rf_pmr.predict(row)[0]
            proba = rf_pmr.predict_proba(row)[0]
            conf = proba[1] if pred == 1 else proba[0]

            if pred == 1:
                st.success(f"Accessible (confidence {conf:.0%})")
            else:
                st.error(f"Not accessible (confidence {conf:.0%})")

            # ---- Why the model decided this ----
            st.markdown("**Why the model says this:**")
            st.markdown(explain_factors(width, curb_val, curb_known, crossing))

            with st.expander("Where does the confidence number come from?"):
                n_trees = getattr(rf_pmr, "n_estimators", None)
                trees_txt = f"{n_trees} decision trees" if n_trees else "many decision trees"
                st.markdown(
                    f"""
**The confidence is a vote count, not a guarantee.** Our Random Forest
has {trees_txt} that each cast a vote — {conf:.0%} means about
{conf:.0%} of the trees agreed on this label. It measures how unanimous
the forest is, not how certain we are about the real sidewalk.

**What the model weighs**, ranked by influence (feature importance):
                    """
                )
                names = list(getattr(rf_pmr, "feature_names_in_", row.columns))
                imp = pd.DataFrame({
                    "Factor": [FRIENDLY_NAMES.get(n, n) for n in names],
                    "Influence": rf_pmr.feature_importances_,
                }).sort_values("Influence", ascending=False)
                imp["Influence"] = (imp["Influence"] * 100).round(1).astype(str) + "%"
                st.dataframe(imp, use_container_width=True, hide_index=True)
                st.markdown(
                    """
**How accurate is this?** 100% on a held-out test of 14,455 segments —
but the label is our own rule (width ≥ 0.9 m, curb ≤ 0.06 m), so perfect
accuracy means the model consistently recovers that rule, **not** that
it has been verified against real wheelchair users (only 50 verified
labels exist). Treat the result as guidance, not ground truth.
                    """
                )
        except FileNotFoundError:
            st.error(
                "Model files not found. Run notebook 03 first so "
                "data/models/ has rf_pmr.pkl and crossing_encoder.pkl."
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Housing address similarity search
# ---------------------------------------------------------
with tab_housing:
    st.markdown('<div class="am-panel">', unsafe_allow_html=True)
    st.header("Find similar addresses")
    st.write(
        "Pick an address from the dataset. FAISS returns the most "
        "similar addresses by ModernBERT embedding distance."
    )

    try:
        housing, index, embeddings = load_housing_search()

        query_addr = st.selectbox(
            "Pick an address",
            options=housing.index,
            format_func=lambda i: housing["aadress"].iloc[i],
        )
        k = st.slider("How many results", min_value=3, max_value=10, value=5)

        if st.button("Search", type="primary"):
            distances, neighbor_ids = index.search(embeddings[query_addr:query_addr + 1], k + 1)

            st.subheader("Results")
            results = []
            for nid, dist in zip(neighbor_ids[0], distances[0]):
                if nid == query_addr:
                    continue  # skip the query itself
                results.append({
                    "Address": housing["aadress"].iloc[nid],
                    "Sidewalk OK?": housing["sidewalk_ok"].iloc[nid],
                    "Distance": round(float(dist), 3),
                })
            st.dataframe(pd.DataFrame(results[:k]), use_container_width=True)

            st.caption(
                "Lower distance = more similar address text. "
                "Similar addresses are usually in the same area. "
                "The 'Sidewalk OK?' column is the real crowd-sourced label, "
                "not a prediction — we tested a prediction model for US "
                "addresses and it had no signal, so we report the data instead."
            )
    except FileNotFoundError:
        st.error(
            "Search files not found. Run notebook 03 first so "
            "data/processed/ has housing_faiss.index and the embeddings."
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 3: About & limitations
# ---------------------------------------------------------
with tab_about:
    st.markdown('<div class="am-panel">', unsafe_allow_html=True)
    st.header("About this project")
    st.markdown(
        """
**What this app does**
- Sidewalk Checker: a Random Forest trained on 72,274 Amsterdam
  sidewalk segments (PMR dataset). Input a segment's measurements,
  get an accessible / not accessible call.
- Address Search: FAISS similarity search over 6,425 US addresses
  (Housing dataset), embedded with ModernBERT.

**What this app does NOT do (on purpose)**
- No sidewalk *prediction* for US addresses. We tested it. After
  removing the biased `state` feature, the remaining house-level
  features carry no real signal (accuracy 85.0% vs. an 85.3%
  majority baseline, recall on bad sidewalks 0.01). We report the
  crowd label instead of pretending to predict.

**Known limits**
- The PMR label is our own rule (width >= 0.9m, curb <= 0.06m),
  not verified ground truth. Only 50 verified labels exist in the
  data — too few to train on.
- 74% of PMR segments have no measured curb height.
- Housing labels are crowd judgments from street photos, not
  physical measurements.
- Neither model transfers to the other region.
        """
    )
    st.caption("Group 13A — AI4ALL Ignite Summer 2026")
    st.markdown('</div>', unsafe_allow_html=True)