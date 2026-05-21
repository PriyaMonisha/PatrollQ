# PatrolIQ — Urban Safety Intelligence Platform

> Unsupervised machine learning on 8.5M Chicago crime records to identify geographic hotspots,
> temporal patterns, and structural crime clusters for data-driven patrol resource allocation.

[![CI](https://github.com/PriyaMonisha/PatrollQ/actions/workflows/ci.yml/badge.svg)](https://github.com/PriyaMonisha/PatrollQ/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.37.0-red.svg)](https://streamlit.io)

---

## Demo

> **Record a 20-second GIF using [ScreenToGif](https://www.screentogif.com/) (free, Windows):**
> Map loads → click a cluster → tooltip appears → temporal heatmap → About page metrics
>
> Save as `docs/demo.gif` and replace this line with: `![Demo](docs/demo.gif)`

---

## Architecture

```
Phase A — Local Training Pipeline
──────────────────────────────────
Raw CSV (8.5M rows, Chicago Data Portal)
  └── 00_data_cleaning.py      → data/processed/chicago_crime_500k.csv.gz
  └── 04_feature_engineering   → data/processed/chicago_crime_features.csv.gz
  └── 05_geographic_clustering → artifacts/geographic/   (labels + metrics)
  └── 06_temporal_clustering   → artifacts/temporal/     (labels + metrics + model)
  └── 07_dimensionality_reduction → artifacts/dimensionality/ (PCA + t-SNE CSVs)
  └── 08_mlflow_experiments    → artifacts/mlflow_exports/ (JSON summaries)

Phase B — Streamlit Cloud Display
──────────────────────────────────
Loads pre-computed artifacts — NO training on cloud
  ├── Page 1: Geographic Clustering  (Folium map + cluster insights)
  ├── Page 2: Temporal Clustering    (hourly heatmap + weekend analysis)
  ├── Page 3: Dimensionality Reduction (PCA scree + t-SNE scatter)
  ├── Page 4: MLflow Experiments     (run comparison + model registry)
  └── Page 5: About                  (metrics, architecture, compliance)
```

---

## Key Results

| Model | Algorithm | Metric | Value |
|-------|-----------|--------|-------|
| Geographic clustering | K-Means (k=8) | Silhouette Score | 0.41 |
| Geographic clustering | K-Means (k=8) | Davies-Bouldin | 0.78 |
| Geographic clustering | DBSCAN | Noise fraction | **3.83%** ✓ |
| Temporal clustering | K-Means (k=4) | Silhouette Score | 0.26 |
| Dimensionality reduction | PCA (3 components) | Cumulative variance | 35.9% |
| Dimensionality reduction | t-SNE | KL Divergence | 1.31 |
| Experiment tracking | MLflow | Total runs logged | **16** ✓ |
| Dataset | Chicago Data Portal | Records sampled | 500K from 8.5M |

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/PriyaMonisha/PatrollQ.git
cd PatrollQ

# 2. Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. Launch the Streamlit app (artifacts already committed)
streamlit run streamlit_app.py
```

App opens at **http://localhost:8501**

---

## Reproduce Results

```bash
# Run the full training pipeline from scratch
python scripts/run_full_pipeline.py

# Verify results match reference (seed=42)
python scripts/reproduce_results.py --seed 42
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **ML** | scikit-learn 1.5.2 (K-Means, DBSCAN, Hierarchical, PCA, t-SNE) |
| **Experiment Tracking** | MLflow 2.14.1 (SQLite backend, model registry) |
| **Data** | pandas 2.1.4, numpy 1.26.4, scipy |
| **Visualization** | Streamlit 1.37.0, Plotly 5.22.0, Folium 0.17.0 |
| **Infrastructure** | Docker, GitHub Actions CI, pytest |
| **Python** | 3.11 |

---

## Project Structure

```
PatrollQ/
├── streamlit_app.py              # Home dashboard
├── pages/                        # 5 Streamlit pages
├── src/
│   ├── data/                     # loader.py, preprocessor.py
│   ├── features/                 # engineer.py (cyclical encoding, severity scores)
│   ├── models/                   # geographic, temporal, dimensionality_reduction
│   └── utils/                    # helpers.py (NumpyEncoder, MLflow export)
├── notebooks/                    # 8 sequential pipeline notebooks (.py)
├── artifacts/                    # Pre-computed labels, metrics, MLflow exports
├── scripts/                      # run_full_pipeline.py, reproduce_results.py
├── tests/                        # Unit + integration tests
├── config.py                     # All hyperparameters and paths
├── Dockerfile                    # Single-stage Streamlit image
└── .github/workflows/ci.yml      # Lint → test → docker build
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: chicago_crime_500k.csv.gz` | Run `python scripts/run_full_pipeline.py` first |
| Port 5000 already in use (MLflow UI) | `mlflow ui --port 5001` |
| Streamlit memory error on cloud | Data is pre-sampled to 48K rows — check `data/processed/` exists |
| `ModuleNotFoundError: config` | Activate venv: `venv\Scripts\activate` |

---

## Data Source

[Chicago Data Portal — Crimes 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
— 8.5M records, updated daily. Sampled to 500K most-recent for this project.

---

*GUVI HCL Capstone Project — Unsupervised ML on Public Safety Data*
