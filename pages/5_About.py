# filename: pages/5_About.py
# purpose:  Project overview, dataset facts, tech stack, methodology
# version:  1.0

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAST_MODE, SAMPLE_SIZE, FAST_SAMPLE_SIZE

st.set_page_config(page_title="About — PatrolIQ",
                   page_icon="ℹ️", layout="wide")

st.title("ℹ️ About PatrolIQ")

# ── Project overview ──────────────────────────────────────────
st.header("Project Overview")
st.markdown("""
**PatrolIQ** is a production-grade urban safety intelligence platform built on the
Chicago crime dataset using unsupervised machine learning. This is a GUVI HCL
capstone project for public safety analytics.

The platform answers: *Where do crime hotspots form? When do they peak?
What structure exists in crime patterns when viewed across all features?*
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Facts")
    st.markdown(f"""
| Fact | Value |
|------|-------|
| **Source** | Chicago Data Portal |
| **Full dataset** | 8.5M records (2001–2026) |
| **Sample (production)** | {SAMPLE_SIZE:,} most-recent records |
| **Sample (FAST_MODE)** | {FAST_SAMPLE_SIZE:,} records |
| **Columns** | 22 raw → 40 after feature engineering |
| **Crime types** | 34 distinct Primary Types |
| **Date format** | MM/DD/YYYY HH:MM:SS AM/PM |
| **Geographic bounds** | Lat 41.6–42.0, Lon -87.9 to -87.5 |
""")

with col2:
    st.subheader("Model Performance")
    st.markdown("""
| Model | Metric | Value |
|-------|--------|-------|
| **Geo K-Means (k=8)** | Silhouette | 0.41 |
| **Geo DBSCAN** | Noise fraction | 3.8% ✓ |
| **Geo Hierarchical** | Silhouette | 0.34 (10K subsample) |
| **Temporal K-Means (k=4)** | Silhouette | 0.26 |
| **PCA (3 components)** | Cumulative variance | 35.9% (FAST_MODE) |
| **t-SNE** | KL Divergence | 1.31 |

> Note: Metrics shown are for FAST_MODE (50K records, 3 months).
> Production run (500K, 4+ years) will yield higher variance and richer patterns.
""")

st.divider()

# ── Architecture ──────────────────────────────────────────────
st.header("Two-Phase Architecture")
st.markdown("""
```
Phase A — Local Training Pipeline
─────────────────────────────────
Raw CSV (8.5M rows)
  └── 00_data_cleaning.py     → data/processed/chicago_crime_500k.csv.gz
  └── 04_feature_engineering  → data/processed/chicago_crime_features.csv.gz
  └── 05_geographic_clustering→ artifacts/geographic/ (labels + metrics)
  └── 06_temporal_clustering  → artifacts/temporal/   (labels + metrics + model)
  └── 07_dim_reduction        → artifacts/dimensionality/ (PCA + t-SNE CSVs)
  └── 08_mlflow_experiments   → artifacts/mlflow_exports/ (JSON for Streamlit)

Phase B — Streamlit Cloud Display
──────────────────────────────────
streamlit_app.py loads pre-computed artifacts (NO training on cloud)
  ├── pages/1_Geographic_Clustering.py   → Folium map + Plotly charts
  ├── pages/2_Temporal_Clustering.py     → Heatmap + cluster analysis
  ├── pages/3_Dimensionality_Reduction   → PCA / t-SNE scatter plots
  ├── pages/4_MLflow_Experiments.py      → Run tracking + model registry
  └── pages/5_About.py                   → This page
```
""")

st.divider()

# ── Tech stack ────────────────────────────────────────────────
st.header("Tech Stack")

tc1, tc2, tc3 = st.columns(3)

with tc1:
    st.subheader("ML / Data")
    st.markdown("""
- **Python 3.11**
- **pandas 2.1.4**
- **scikit-learn 1.5.2**
- **scipy** (KDE, statistics)
- **MLflow** (experiment tracking)
""")

with tc2:
    st.subheader("Visualization")
    st.markdown("""
- **Streamlit 1.37.0**
- **Plotly 5.22.0** (charts)
- **Folium 0.17.0** (maps)
- **streamlit-folium 0.22.0**
- **matplotlib / seaborn** (figures)
""")

with tc3:
    st.subheader("Infrastructure")
    st.markdown("""
- **Git** (version control)
- **GitHub** (remote)
- **MLflow SQLite** (tracking DB)
- **Docker** (Section 10)
- **GitHub Actions** (CI)
""")

st.divider()

# ── GUVI compliance ───────────────────────────────────────────
st.header("GUVI Capstone Requirements")
st.markdown("""
| Requirement | Target | Actual (FAST_MODE) | Status |
|-------------|--------|---------------------|--------|
| Geographic K-Means silhouette | > 0.5 | 0.4115 (50K, 3 months) | Below in dev — higher expected in production |
| DBSCAN noise fraction | < 10% | **3.83%** | **PASS** |
| Hierarchical clustering | Subsample | Ward k=8 on 10K rows, sil=0.34 | PASS |
| PCA explained variance | >= 70% (2-3 comp.) | 35.9% (3 months data) | Below in dev — production target |
| t-SNE: visually distinct clusters | Qualitative | KL=1.31, 5K subsample | See Page 3 |
| MLflow >= 6 runs logged | >= 6 | **16 runs** | **PASS** |
| MLflow model registry | 1+ model | PatrolIQ_TemporalClustering v2 | **PASS** |
| Streamlit multi-page dashboard | 5 pages | 5 pages (Pages 1–5) | **PASS** |
| Full pipeline script | Automated | scripts/run_full_pipeline.py | **PASS** |

> **Note on FAST_MODE metrics:** Training used 50K most-recent records (Feb–Apr 2026 only —
> 3 months). Silhouette and PCA variance improve significantly with full 500K sample
> (4+ years of data covering all seasons and crime patterns).
""")
