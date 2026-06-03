# filename: streamlit_app.py
# purpose:  PatrolIQ home page — overview dashboard with KPI cards
# version:  1.0

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FAST_MODE,
    MLFLOW_EXPORTS_DIR,
    PROCESSED_CSV,
    SAMPLE_METADATA,
)

st.set_page_config(
    page_title="PatrolIQ — Urban Safety Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_metadata() -> dict:
    if SAMPLE_METADATA.exists():
        with open(SAMPLE_METADATA) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_processed() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_CSV, usecols=["arrest", "domestic", "primary_type"])
    # Normalise bool columns — CSV round-trip produces "True"/"False" strings
    for col in ("arrest", "domestic"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
    return df

@st.cache_data
def load_best_models() -> dict:
    path = MLFLOW_EXPORTS_DIR / "best_models.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

meta        = load_metadata()
best_models = load_best_models()

# ── FAST_MODE banner ─────────────────────────────────────────
if FAST_MODE:
    st.warning(
        "**FAST_MODE is ON** — displaying results from a 50K record sample "
        "(Feb–Apr 2026). Switch `FAST_MODE = False` in config.py and re-run "
        "the pipeline for full 500K production results."
    )

# ── Header ────────────────────────────────────────────────────
st.title("🔍 PatrolIQ — Urban Safety Intelligence Platform")
st.caption(
    "Unsupervised machine learning on Chicago crime data (2001–2026) | "
    "Urban Safety Intelligence Platform"
)
st.divider()

# ── KPI cards ─────────────────────────────────────────────────
n_records    = meta.get("n_records", "N/A")
date_min     = meta.get("date_min", "N/A")
date_max     = meta.get("date_max", "N/A")
crime_types  = meta.get("crime_types", "N/A")

try:
    df_kpi      = load_processed()
    arrest_rate  = f"{df_kpi['arrest'].astype(bool).mean() * 100:.1f}%"
    domestic_pct = f"{df_kpi['domestic'].astype(bool).mean() * 100:.1f}%"
except Exception:
    arrest_rate  = "N/A"
    domestic_pct = "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Crimes",  f"{n_records:,}" if isinstance(n_records, int) else n_records)
# Split date range into two lines to avoid truncation
col2.metric("Sample Start",  date_min)
col3.metric("Sample End",    date_max)
crime_label = str(crime_types) if crime_types != "N/A" else "N/A"
col4.metric("Crime Types",   crime_label)
col5.metric("Arrest Rate",   arrest_rate)

st.divider()

# ── Model performance summary ─────────────────────────────────
st.subheader("Model Performance Summary")

if best_models:
    bcol1, bcol2, bcol3 = st.columns(3)

    with bcol1:
        geo = best_models.get("geo_kmeans", {})
        st.metric("Geographic K-Means", f"Silhouette: {geo.get('silhouette', 'N/A'):.3f}",
                  f"K = {geo.get('n_clusters', 'N/A')} clusters")

    with bcol2:
        temp = best_models.get("temporal_kmeans", {})
        st.metric("Temporal K-Means", f"Silhouette: {temp.get('silhouette', 'N/A'):.3f}",
                  f"K = {temp.get('n_clusters', 'N/A')} clusters")

    with bcol3:
        pca = best_models.get("pca", {})
        st.metric("PCA Variance", f"{pca.get('cumulative_variance', 0)*100:.1f}%",
                  f"{pca.get('n_components', 'N/A')} components")

st.divider()

# ── Section navigation cards ──────────────────────────────────
st.subheader("Explore the Analysis")

nav1, nav2, nav3 = st.columns(3)
nav4, nav5, _    = st.columns(3)

with nav1:
    st.info("**🗺️ Page 1 — Geographic Clustering**\n\n"
            "K-Means, DBSCAN, and Hierarchical clustering on crime lat/lon. "
            "Interactive Folium map with cluster-colored crime locations.")

with nav2:
    st.info("**⏰ Page 2 — Temporal Clustering**\n\n"
            "K-Means on time-of-day features. Heatmap shows when each crime "
            "cluster peaks across hours of the day.")

with nav3:
    st.info("**📊 Page 3 — Dimensionality Reduction**\n\n"
            "PCA 2D/3D projection and t-SNE embedding of the 14-feature "
            "crime space, colored by cluster labels.")

with nav4:
    st.info("**📈 Page 4 — MLflow Experiments**\n\n"
            "Experiment tracking across all 3 models. Metrics comparison "
            "and best model registry.")

with nav5:
    st.info("**ℹ️ Page 5 — About**\n\n"
            "Project overview, dataset facts, tech stack, and methodology.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "PatrolIQ | Chicago Crime Dataset (data.cityofchicago.org) | "
    "Built with scikit-learn, MLflow, Streamlit, Folium, Plotly"
)
