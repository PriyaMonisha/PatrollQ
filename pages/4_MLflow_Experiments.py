# filename: pages/4_MLflow_Experiments.py
# purpose:  MLflow experiment tracking display — metrics + best models
# version:  1.0

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MLFLOW_EXPORTS_DIR

st.set_page_config(page_title="MLflow Experiments — PatrolIQ",
                   page_icon="📈", layout="wide")

@st.cache_data(ttl=3600)
def load_all_runs() -> list:
    path = MLFLOW_EXPORTS_DIR / "all_runs.json"
    with open(path) as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_best_models() -> dict:
    path = MLFLOW_EXPORTS_DIR / "best_models.json"
    with open(path) as f:
        return json.load(f)

all_runs    = load_all_runs()
best_models = load_best_models()

# ── Header ────────────────────────────────────────────────────
st.title("📈 MLflow Experiment Tracking")
st.caption("All training runs logged to sqlite:///mlruns/mlflow.db")

# ── GUVI compliance cards ─────────────────────────────────────
n_runs = len(all_runs)
experiments = {r["experiment"] for r in all_runs}

c1, c2, c3 = st.columns(3)
c1.metric("Total Runs Logged",  n_runs,
          "GUVI target >= 6: PASS" if n_runs >= 6 else "FAIL")
c2.metric("Experiments",        len(experiments))
c3.metric("Models Registered",  "1 (PatrolIQ_TemporalClustering)")

st.divider()

# ── Best models per experiment ────────────────────────────────
st.subheader("Best Model Per Experiment")

rows = []
for key, m in best_models.items():
    key_metric = (
        f"Silhouette: {m['silhouette']:.4f}" if "silhouette" in m
        else f"Cumulative Variance: {m['cumulative_variance']*100:.1f}%"
    )
    rows.append({
        "Model": key.replace("_", " ").title(),
        "Experiment": m["experiment"].replace("patroliq_", "").replace("_", " ").title(),
        "Key Metric": key_metric,
        "Clusters / Components": m.get("n_clusters") or m.get("n_components", "N/A"),
        "Run ID": m["run_id"][:8] + "...",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── All runs table ────────────────────────────────────────────
st.subheader("All Runs")

KEY_METRIC_MAP = {
    "geo_elbow":        "best_k_silhouette",
    "geo_kmeans":       "silhouette",
    "geo_dbscan":       "silhouette",
    "geo_hierarchical": "silhouette",
    "temporal_elbow":   "best_k_by_silhouette",
    "temporal_kmeans":  "silhouette",
    "pca":              "cumulative_variance",
    "tsne":             "kl_divergence",
}

# Deduplicate by (experiment, run_name) — keep most recent
seen = set()
unique_runs = []
for run in sorted(all_runs, key=lambda r: r.get("start_time", ""), reverse=True):
    key = (run["experiment"], run["run_name"])
    if key not in seen:
        seen.add(key)
        unique_runs.append(run)

table_rows = []
for run in sorted(unique_runs, key=lambda r: r["experiment"]):
    metric_key = KEY_METRIC_MAP.get(run["run_name"])
    raw_val = run["metrics"].get(metric_key) if metric_key else None
    value_str = f"{raw_val:.4f}" if isinstance(raw_val, float) else str(raw_val or "N/A")
    table_rows.append({
        "Run Name":  run["run_name"],
        "Experiment": run["experiment"].replace("patroliq_", ""),
        "Metric": metric_key or "—",
        "Value":  value_str,
        "Status": run.get("status", "N/A"),
    })

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Silhouette comparison bar ─────────────────────────────────
st.subheader("Silhouette Score Comparison")
sil_data = []
for key, m in best_models.items():
    if "silhouette" in m:
        sil_data.append({
            "Model": key.replace("_", " ").title(),
            "Silhouette": m["silhouette"],
        })

if sil_data:
    fig = px.bar(
        pd.DataFrame(sil_data), x="Model", y="Silhouette",
        color="Silhouette", color_continuous_scale="Blues",
        title="Silhouette Score by Model (Higher = Better Cluster Separation)",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="Target 0.5")
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
