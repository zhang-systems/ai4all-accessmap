# AccessMap - Streamlit app
# Loads the trained models and FAISS indexes from notebook 03.
# Run from the project root: streamlit run app.py
#
# Folder layout this app expects:
#   accessmap-project/
#     app.py            <- this file
#     data/models/      <- rf_pmr.pkl, encoders, scaler
#     data/processed/   <- housing_clean.csv, embeddings, faiss indexes

import base64
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import faiss
from PIL import Image

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

# Wheelchair-accessibility icon, embedded inline so the app stays a single
# file (no separate assets folder to keep in sync).
LOGO_B64 = "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"


def _load_icon():
    return Image.open(BytesIO(base64.b64decode(LOGO_B64)))


st.set_page_config(page_title="AccessMap", page_icon=_load_icon(), layout="centered")

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

        .am-hero-row {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 1rem;
        }}

        .am-hero-icon {{
            height: 68px;
            width: auto;
            flex-shrink: 0;
        }}

        .am-hero h1 {{
            color: #FFFFFF;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin: 0;
        }}

        .am-hero p {{
            color: #CBD5E1;
            font-size: 0.98rem;
            margin: 0.7rem 0 0 0;
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
        f"""
        <div class="am-hero">
            <div class="am-hero-row">
                <h1>AccessMap</h1>
                <img class="am-hero-icon" src="data:image/png;base64,{LOGO_B64}" alt="Accessibility icon" />
            </div>
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