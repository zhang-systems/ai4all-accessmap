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

st.set_page_config(page_title="AccessMap", page_icon="wheelchair", layout="centered")

MODELS_DIR = "data/models"
PROCESSED_DIR = "data/processed"


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


# ---------------------------------------------------------
# App
# ---------------------------------------------------------
st.title("AccessMap")
st.caption("Sidewalk accessibility: check a segment, search similar addresses.")

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

            # Same fill value notebook 03 used for missing curb heights
            CURB_MEDIAN_FILL = 0.06
            row = pd.DataFrame([{
                "length": length,
                "obstacle_free_width_float": width,
                "crossing": crossing_encoder.transform([crossing])[0],
                "width_fill": width_fill,
                "curb_height_max": curb_height if curb_known else CURB_MEDIAN_FILL,
                "curb_height_missing": 0 if curb_known else 1,
            }])

            pred = rf_pmr.predict(row)[0]
            proba = rf_pmr.predict_proba(row)[0]

            if pred == 1:
                st.success(f"Accessible (confidence {proba[1]:.0%})")
            else:
                st.error(f"Not accessible (confidence {proba[0]:.0%})")

            st.caption(
                "Reminder: this label comes from our own rule "
                "(width >= 0.9m and curb <= 0.06m), not verified ground truth."
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
                "Similar addresses are usually in the same area."
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
