# filename: pages/1_Geographic_Clustering.py
# purpose:  Geographic crime hotspot clustering display — K-Means, DBSCAN, Hierarchical
# version:  1.0

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR, FAST_MODE, RANDOM_STATE

GEO_DIR = ARTIFACTS_DIR / "geographic"

st.set_page_config(page_title="Geographic Clustering — PatrolIQ",
                   page_icon="🗺️", layout="wide")

# ── Data loaders ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_labels(model_key: str) -> pd.DataFrame:
    path = GEO_DIR / f"{model_key}_labels.csv"
    if not path.exists():
        st.error(f"Artifact missing: {path.name}. Run scripts/run_full_pipeline.py first.")
        st.stop()
    return pd.read_csv(path)

@st.cache_data(ttl=3600)
def load_metrics(model_key: str) -> dict:
    path = GEO_DIR / f"{model_key}_metrics.json"
    if not path.exists():
        st.error(f"Artifact missing: {path.name}. Run scripts/run_full_pipeline.py first.")
        st.stop()
    with open(path) as f:
        return json.load(f)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Model Selection")
model_display = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means (k=8)", "DBSCAN", "Hierarchical (Ward)"],
)
model_key = {"K-Means (k=8)": "kmeans",
             "DBSCAN": "dbscan",
             "Hierarchical (Ward)": "hierarchical"}[model_display]

map_sample = st.sidebar.slider(
    "Map sample size (points)", 1_000, 10_000, 5_000, step=1_000,
    help="More points = slower render. 5K is a good balance."
)

# ── Load ──────────────────────────────────────────────────────
labels_df = load_labels(model_key)
metrics   = load_metrics(model_key)

# ── Header + metrics ──────────────────────────────────────────
st.title("🗺️ Geographic Crime Clustering")
if FAST_MODE:
    st.caption("FAST_MODE — 50K sample | Feb–Apr 2026")

m1, m2, m3 = st.columns(3)
m1.metric("Clusters",    metrics.get("n_clusters", "N/A"))
m2.metric("Silhouette",  f"{metrics.get('silhouette', 0):.4f}" if metrics.get("silhouette") else "N/A",
          "Higher = better separation")
if model_key == "dbscan":
    m3.metric("Noise Fraction", f"{metrics.get('noise_fraction', 0)*100:.1f}%",
              "Target < 10%")
else:
    m3.metric("Davies-Bouldin", f"{metrics.get('davies_bouldin', 0):.4f}",
              "Lower = better")

st.divider()

# ── Folium map ────────────────────────────────────────────────
st.subheader(f"Crime Cluster Map — {model_display}")
st.caption(f"Showing {min(map_sample, len(labels_df)):,} of {len(labels_df):,} points")

# Subsample
df_map = labels_df.dropna(subset=["latitude", "longitude"])
if len(df_map) > map_sample:
    df_map = df_map.sample(map_sample, random_state=RANDOM_STATE)

# Cluster color palette (up to 10 clusters)
COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62",
]

def cluster_color(c: int) -> str:
    if c == -1:
        return "#cccccc"  # DBSCAN noise = gray
    return COLORS[c % len(COLORS)]

# Build map centered on Chicago
fmap = folium.Map(location=[41.83, -87.65], zoom_start=11, tiles="CartoDB positron")

for _, row in df_map.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color=cluster_color(int(row["cluster"])),
        fill=True,
        fill_opacity=0.6,
        popup=f"Cluster {int(row['cluster'])}",
    ).add_to(fmap)

st_folium(fmap, width=1000, height=520)

# ── Cluster size distribution ─────────────────────────────────
st.subheader("Cluster Size Distribution")
cluster_counts = (
    labels_df[labels_df["cluster"] != -1]["cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)
cluster_counts.columns = ["Cluster", "Count"]

fig = px.bar(
    cluster_counts, x="Cluster", y="Count",
    color="Cluster", color_continuous_scale="Viridis",
    title=f"{model_display} — Crime Count per Cluster",
    labels={"Count": "Number of Crimes"},
)
fig.update_layout(showlegend=False, height=350)
st.plotly_chart(fig, use_container_width=True)

if model_key == "dbscan":
    noise = metrics.get("noise_count", 0)
    st.caption(f"Noise points (cluster = -1): {noise:,} "
               f"({metrics.get('noise_fraction', 0)*100:.1f}%)")
