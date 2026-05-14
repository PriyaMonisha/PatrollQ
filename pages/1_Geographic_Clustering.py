# filename: pages/1_Geographic_Clustering.py
# purpose:  Geographic crime hotspot clustering — WHERE do crimes cluster in Chicago?
# version:  2.0

import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR, FAST_MODE, PROCESSED_CSV, RANDOM_STATE

GEO_DIR = ARTIFACTS_DIR / "geographic"

st.set_page_config(page_title="Where Are Crime Hotspots? — PatrolIQ",
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

@st.cache_data(ttl=3600)
def load_enriched(model_key: str) -> pd.DataFrame:
    """Join cluster labels with crime type + hour from processed CSV."""
    labels = load_labels(model_key)
    processed = pd.read_csv(
        PROCESSED_CSV,
        usecols=["case_number", "primary_type", "Hour", "arrest"],
    )
    merged = labels.merge(processed, on="case_number", how="left")
    return merged

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("Filters")

model_display = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means (k=8)", "DBSCAN", "Hierarchical (Ward)"],
)
_ALGO_MAP = {
    "K-Means (k=8)":       "kmeans",
    "DBSCAN":              "dbscan",
    "Hierarchical (Ward)": "hierarchical",
}
model_key = _ALGO_MAP.get(model_display or "", "kmeans")

# Load enriched data for crime type filter
enriched = load_enriched(model_key)
crime_types_available = sorted(enriched["primary_type"].dropna().unique().tolist())

crime_filter = st.sidebar.selectbox(
    "Filter by Crime Type",
    ["All Crime Types"] + crime_types_available,
    help="Select a specific crime to see where it clusters most.",
)

map_sample = st.sidebar.slider(
    "Map points", 1_000, 10_000, 5_000, step=1_000,
    help="More points = slower render. 5K is a good balance.",
)

# ── Load metrics ──────────────────────────────────────────────
metrics = load_metrics(model_key)

# ── Apply crime type filter ───────────────────────────────────
if crime_filter != "All Crime Types":
    df_filtered = enriched[enriched["primary_type"] == crime_filter]
else:
    df_filtered = enriched.copy()

# ── Page header ───────────────────────────────────────────────
st.title("🗺️ Where Are Crime Hotspots in Chicago?")
if crime_filter != "All Crime Types":
    st.subheader(f"Showing: **{crime_filter}** — {len(df_filtered):,} incidents")
if FAST_MODE:
    st.warning(
        "**FAST_MODE ON** — Results from 50K most-recent records (Feb–Apr 2026). "
        "Set `FAST_MODE = False` in config.py and re-run pipeline for full 500K results."
    )

# ── Model metric cards ────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Geographic Clusters", metrics.get("n_clusters", "N/A"))
m2.metric("Silhouette Score",
          f"{metrics.get('silhouette', 0):.4f}" if metrics.get("silhouette") else "N/A",
          "Higher = better separation")
if model_key == "dbscan":
    m3.metric("Noise Fraction", f"{metrics.get('noise_fraction', 0)*100:.1f}%", "Target < 10%")
else:
    m3.metric("Davies-Bouldin", f"{metrics.get('davies_bouldin', 0):.4f}", "Lower = better")

st.divider()

# ── Cluster summary cards (the "insight" layer) ───────────────
st.subheader("Cluster Insights — What Action to Take")

cluster_summary = (
    df_filtered[df_filtered["cluster"] != -1]
    .groupby("cluster")
    .agg(
        crime_count=("cluster", "count"),
        peak_hour=("Hour", lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"),
        arrest_rate=("arrest", lambda x: x.astype(str).str.lower()
                     .map({"true": 1, "false": 0}).mean()),
        top_crime=("primary_type", lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"),
    )
    .reset_index()
    .sort_values("crime_count", ascending=False)
)

# Show top-4 clusters as insight cards
top_clusters = cluster_summary.head(4)
card_cols = st.columns(len(top_clusters))

for col, (_, row) in zip(card_cols, top_clusters.iterrows()):
    peak_h = int(row["peak_hour"]) if row["peak_hour"] != "N/A" else "N/A"
    peak_str = (
        f"{peak_h:02d}:00–{(peak_h+2)%24:02d}:00" if isinstance(peak_h, int) else "N/A"
    )
    arrest_pct = f"{row['arrest_rate']*100:.0f}%" if pd.notna(row["arrest_rate"]) else "N/A"

    col.markdown(
        f"""
        **Cluster {int(row['cluster'])}**
        - Crimes: **{int(row['crime_count']):,}**
        - Peak time: **{peak_str}**
        - Top crime: **{row['top_crime']}**
        - Arrest rate: **{arrest_pct}**
        """
    )

st.divider()

# ── Folium map ────────────────────────────────────────────────
st.subheader(f"Crime Map — {model_display}" + (f" | {crime_filter}" if crime_filter != "All Crime Types" else ""))

df_map = df_filtered.dropna(subset=["latitude", "longitude"])
if len(df_map) > map_sample:
    df_map = df_map.sample(map_sample, random_state=RANDOM_STATE)

st.caption(f"Showing {len(df_map):,} points on map")

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62",
]

def cluster_color(c: int) -> str:
    if c == -1:
        return "#cccccc"
    return COLORS[c % len(COLORS)]

fmap = folium.Map(location=[41.83, -87.65], zoom_start=11, tiles="CartoDB positron")

for _, row in df_map.iterrows():
    c = int(row["cluster"])
    crime = row.get("primary_type", "Unknown")
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color=cluster_color(c),
        fill=True,
        fill_opacity=0.6,
        popup=f"Cluster {c} | {crime}",
        tooltip=f"Cluster {c}",
    ).add_to(fmap)

st_folium(fmap, width=1050, height=520)

# ── Cluster size bar ──────────────────────────────────────────
st.subheader("Crime Volume per Cluster")

cluster_counts = (
    df_filtered[df_filtered["cluster"] != -1]["cluster"]
    .value_counts().sort_index().reset_index()
)
cluster_counts.columns = ["Cluster", "Count"]
cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
           "#ff7f00", "#a65628", "#f781bf", "#999999"]

fig = px.bar(
    cluster_counts, x="Cluster", y="Count",
    color="Cluster",
    color_discrete_sequence=_COLORS,
    title=f"Crime count per cluster — {crime_filter}",
    text="Count",
)
fig.update_traces(texttemplate="%{text:,}", textposition="outside")
fig.update_layout(showlegend=False, height=350, yaxis_title="Number of Crimes")
st.plotly_chart(fig, use_container_width=True)

if model_key == "dbscan":
    noise = metrics.get("noise_count", 0)
    st.caption(
        f"Noise points (unclassified): {noise:,} "
        f"({metrics.get('noise_fraction', 0)*100:.1f}%) — these are isolated incidents"
    )
