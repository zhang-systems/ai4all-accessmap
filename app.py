# AccessMap - Streamlit app
# Loads the trained models and FAISS indexes from notebook 03.
# Run from the project root: streamlit run app.py
#
# Folder layout this app expects:
#   accessmap-project/
#     app.py            <- this file
#     src/logo.png      <- app logo
#     data/models/      <- rf_pmr.pkl, encoders, scaler
#     data/processed/   <- housing_clean.csv, embeddings, faiss indexes

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import faiss

st.set_page_config(page_title="AccessMap", page_icon="src/logo.png", layout="centered")

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


# ---------------------------------------------------------
# Custom styling — matches the presentation palette
# Deep teal 0A3A42, accent 02C39A, coral C0392B, off-white F4FAF9
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@700&display=swap');

    /* Body font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main title — serif to match the presentation deck */
    h1 {
        font-family: 'Playfair Display', Cambria, serif !important;
        color: #0A3A42 !important;
        letter-spacing: -1px;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: #0A3A42 !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
    }

    /* Round the logo image */
    div[data-testid="stImage"] img {
        border-radius: 50%;
        box-shadow: 0 4px 14px rgba(10, 58, 66, 0.15);
    }

    /* Tab styling — brand color for the active tab */
    button[data-baseweb="tab"] {
        font-weight: 500;
        color: #5B6770;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #02C39A !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #02C39A !important;
    }

    /* Primary button — brand teal */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #02C39A;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #00A896;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(2, 195, 154, 0.3);
    }

    /* Result banner styling — bigger, cleaner cards */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: none;
        box-shadow: 0 2px 8px rgba(10, 58, 66, 0.06);
    }

    /* Section divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #E6F2F1;
    }

    /* Captions — softer */
    .stCaption, div[data-testid="stCaptionContainer"] {
        color: #5B6770 !important;
    }
    </style>
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
    """Plain-language readout of how each input pushes the prediction."""
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
# App header
# ---------------------------------------------------------
col_logo, col_title = st.columns([1, 6], vertical_alignment="center")
with col_logo:
    #st.image("src/logo.png", width=90)
    st.markdown("<div style='font-size: 70px; text-align: center;'>♿</div>", unsafe_allow_html=True)
with col_title:
    st.title("AccessMap")
    st.caption("Sidewalk accessibility: check a segment, search similar addresses.")

st.markdown("<hr/>", unsafe_allow_html=True)

tab_pmr, tab_housing, tab_about = st.tabs(
    ["Sidewalk Checker (PMR)", "Address Search (Housing)", "About & Limits"]
)

# ---------------------------------------------------------
# Tab 1: PMR sidewalk checker
# ---------------------------------------------------------
with tab_pmr:
    st.header("Check a sidewalk segment")
    st.write(
        "Enter the segment's measurements. The model says if it's "
        "accessible for a wheelchair user."
    )

    st.markdown("##### Segment measurements")
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

    st.markdown("")  # small vertical spacer
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

            # ---- Big result card ----
            if pred == 1:
                bg, border, text_color, icon, label = (
                    "#E8F7F1", "#02C39A", "#0A3A42", "✓", "Accessible"
                )
            else:
                bg, border, text_color, icon, label = (
                    "#FBE9E7", "#C0392B", "#7A1F14", "✕", "Not accessible"
                )

            st.markdown(
                f"""
                <div style="
                    background: {bg};
                    border-left: 6px solid {border};
                    border-radius: 12px;
                    padding: 1.5rem 1.75rem;
                    margin: 1rem 0;
                    box-shadow: 0 2px 10px rgba(10, 58, 66, 0.06);
                ">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <div style="font-size: 0.85rem; font-weight: 600;
                                        text-transform: uppercase; letter-spacing: 1px;
                                        color: {border}; margin-bottom: 0.3rem;">
                                Prediction
                            </div>
                            <div style="font-size: 1.6rem; font-weight: 700;
                                        color: {text_color};">
                                {icon}  {label}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.85rem; font-weight: 600;
                                        text-transform: uppercase; letter-spacing: 1px;
                                        color: {border}; margin-bottom: 0.3rem;">
                                Confidence
                            </div>
                            <div style="font-size: 2.2rem; font-weight: 700;
                                        color: {text_color}; line-height: 1;">
                                {conf:.0%}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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

# ---------------------------------------------------------
# Tab 2: Housing address similarity search
# ---------------------------------------------------------
with tab_housing:
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

        st.markdown("")
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
            st.dataframe(pd.DataFrame(results[:k]), use_container_width=True, hide_index=True)

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

# ---------------------------------------------------------
# Tab 3: About & limitations
# ---------------------------------------------------------
with tab_about:
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


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #5B6770; font-size: 0.85rem; padding: 0.5rem 0 1rem 0;">
        AccessMap · Group 13A · AI4ALL Ignite Summer 2026
    </div>
    """,
    unsafe_allow_html=True,
)
