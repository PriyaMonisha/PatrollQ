# filename: pages/2_Temporal_Clustering.py
# purpose:  Temporal crime pattern clustering display — K-Means k=4
# version:  1.0

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR, FAST_MODE, FEATURES_CSV

TEMP_DIR = ARTIFACTS_DIR / "temporal"

st.set_page_config(page_title="Temporal Clustering — PatrolIQ",
                   page_icon="⏰", layout="wide")

@st.cache_data(ttl=3600)
def load_temporal_data() -> pd.DataFrame:
    """Load temporal labels joined with Hour from feature CSV."""
    labels = pd.read_csv(TEMP_DIR / "kmeans_labels.csv")   # row_idx + cluster
    # Load Hour from feature CSV
    _dev = Path(str(FEATURES_CSV).replace(".csv.gz", "_dev.csv.gz"))
    feat_path = _dev if (FAST_MODE and _dev.exists()) else FEATURES_CSV
    hours = pd.read_csv(feat_path, usecols=["Hour", "Is_Weekend"])
    # Align by position (labels were saved in same row order as feature CSV)
    labels["Hour"]       = hours["Hour"].values[:len(labels)]
    labels["Is_Weekend"] = hours["Is_Weekend"].values[:len(labels)]
    return labels

@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    with open(TEMP_DIR / "kmeans_metrics.json") as f:
        return json.load(f)

# ── Load ──────────────────────────────────────────────────────
df   = load_temporal_data()
metrics = load_metrics()

# ── Header ────────────────────────────────────────────────────
st.title("⏰ Temporal Crime Pattern Clustering")
if FAST_MODE:
    st.warning(
        "**FAST_MODE ON** — Results from 50K most-recent records (Feb–Apr 2026). "
        "Only Spring/Winter seasons present. Set `FAST_MODE = False` for full seasonal patterns."
    )

m1, m2, m3 = st.columns(3)
m1.metric("K (clusters)",    metrics.get("n_clusters", "N/A"))
m2.metric("Silhouette",      f"{metrics.get('silhouette', 0):.4f}",
          "Higher = better temporal separation")
m3.metric("Davies-Bouldin",  f"{metrics.get('davies_bouldin', 0):.4f}",
          "Lower = better")

st.divider()

# ── Heatmap: cluster × Hour ───────────────────────────────────
st.subheader("When Each Cluster Is Active — Normalized Activity by Hour")

heatmap_data = (
    df.groupby(["cluster", "Hour"])
    .size()
    .unstack(fill_value=0)
)
# Normalize by cluster total — shows pattern, not cluster size dominance
# replace(0, 1) guards against div-by-zero if a cluster has no crimes in FAST_MODE
heatmap_norm = heatmap_data.div(
    heatmap_data.sum(axis=1).replace(0, 1), axis=0
)

fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_norm.values,
    x=[str(h) for h in heatmap_norm.columns],
    y=[f"Cluster {c}" for c in heatmap_norm.index],
    colorscale="YlOrRd",
    colorbar=dict(title="Fraction of<br>Cluster Crimes"),
))
fig_heat.update_layout(
    title="Normalized Crime Activity: Cluster × Hour of Day",
    xaxis_title="Hour of Day (0 = midnight, 12 = noon)",
    yaxis_title="Temporal Cluster",
    height=350,
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Cluster size + weekend breakdown ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Size Distribution")
    counts = df["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["Cluster", "Count"]
    fig_bar = px.bar(
        counts, x="Cluster", y="Count",
        color="Count", color_continuous_scale="Blues",
        title="Crime Count per Temporal Cluster",
    )
    fig_bar.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Weekend vs Weekday per Cluster")
    wknd = (
        df.groupby(["cluster", "Is_Weekend"])
        .size()
        .reset_index(name="count")
    )
    wknd["Day Type"] = wknd["Is_Weekend"].map({True: "Weekend", False: "Weekday"})
    fig_wk = px.bar(
        wknd, x="cluster", y="count", color="Day Type",
        barmode="group",
        color_discrete_map={"Weekend": "#e41a1c", "Weekday": "#377eb8"},
        title="Weekday vs Weekend by Cluster",
        labels={"cluster": "Cluster", "count": "Count"},
    )
    fig_wk.update_layout(height=300)
    st.plotly_chart(fig_wk, use_container_width=True)

st.caption(
    "**Cluster interpretation (K=4):** "
    "Clusters correspond to ~4 daily activity windows — "
    "late-night, morning rush, afternoon peak, and evening hours."
)
