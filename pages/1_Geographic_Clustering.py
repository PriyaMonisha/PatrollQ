# filename: pages/1_Geographic_Clustering.py
# purpose:  Where are crime hotspots in Chicago? — geographic clustering dashboard
# version:  3.0

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

st.set_page_config(page_title="Crime Hotspots — PatrolIQ",
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
    return labels.merge(processed, on="case_number", how="left")

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

enriched = load_enriched(model_key)
crime_types_available = sorted(enriched["primary_type"].dropna().unique().tolist())

crime_filter = st.sidebar.selectbox(
    "Filter by Crime Type",
    ["All Crime Types"] + crime_types_available,
    help="Select a specific crime to see where it clusters.",
)

map_sample = st.sidebar.slider(
    "Map points", 500, 5_000, 2_000, step=500,
    help="Fewer points = faster map interaction.",
)

# ── Apply filter ──────────────────────────────────────────────
metrics    = load_metrics(model_key)
df_all     = enriched[enriched["cluster"] != -1]
df_filtered = (
    df_all[df_all["primary_type"] == crime_filter]
    if crime_filter != "All Crime Types" else df_all
)

# ── Page title ────────────────────────────────────────────────
st.title("🗺️ Where Are Crime Hotspots in Chicago?")
if crime_filter != "All Crime Types":
    st.markdown(f"**Filtered to: {crime_filter}** — {len(df_filtered):,} incidents")
if FAST_MODE:
    st.warning(
        "**FAST_MODE** — 50K records (Feb–Apr 2026). "
        "Set `FAST_MODE = False` and re-run pipeline for full 500K results."
    )

# ══════════════════════════════════════════════
# SECTION 1 — ACTIONABLE INSIGHTS (non-tech, top)
# ══════════════════════════════════════════════
st.subheader("📋 Cluster Insights — What to Act On")

cluster_summary = (
    df_filtered.groupby("cluster")
    .agg(
        crime_count=("cluster", "count"),
        peak_hour=("Hour", lambda x: int(x.mode().iloc[0]) if len(x) > 0 else -1),
        top_crime=("primary_type", lambda x: x.mode().iloc[0] if len(x) > 0 else "N/A"),
        arrest_rate=("arrest", lambda x:
                     x.astype(str).str.lower()
                      .map({"true": 1.0, "false": 0.0})
                      .mean()),
    )
    .reset_index()
    .sort_values("crime_count", ascending=False)
)

top_n = min(4, len(cluster_summary))
card_cols = st.columns(top_n)

for col, (_, row) in zip(card_cols, cluster_summary.head(top_n).iterrows()):
    ph = row["peak_hour"]
    peak_str = f"{ph:02d}:00 – {(ph+2)%24:02d}:00" if ph >= 0 else "N/A"
    ar = row["arrest_rate"]
    ar_str = f"{ar*100:.0f}%" if pd.notna(ar) else "N/A"
    with col:
        st.markdown(
            f"**Cluster {int(row['cluster'])}**  \n"
            f"🔴 {int(row['crime_count']):,} crimes  \n"
            f"⏰ Peak: **{peak_str}**  \n"
            f"🔑 Main type: **{row['top_crime']}**  \n"
            f"👮 Arrest rate: **{ar_str}**"
        )

# ══════════════════════════════════════════════
# SECTION 2 — CRIME MAP + VOLUME SIDE BY SIDE
# ══════════════════════════════════════════════
map_col, bar_col = st.columns([3, 1])

with map_col:
    st.markdown(f"**Crime Map** — {model_display}"
                + (f" | {crime_filter}" if crime_filter != "All Crime Types" else "")
                + f"  \n<small>Showing {min(map_sample, len(df_filtered)):,} "
                  f"of {len(df_filtered):,} points. "
                  "**Scroll on the page — not the map** to avoid accidental zoom.</small>",
                unsafe_allow_html=True)

    df_map = df_filtered.dropna(subset=["latitude", "longitude"])
    if len(df_map) > map_sample:
        df_map = df_map.sample(map_sample, random_state=RANDOM_STATE)

    COLORS = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
        "#66c2a5", "#fc8d62",
    ]

    # prefer_canvas=True + scrollWheelZoom=False = much faster, no scroll capture
    fmap = folium.Map(
        location=[41.83, -87.65],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,       # canvas rendering — faster than SVG for many points
        scrollWheelZoom=False,    # prevents page scroll from zooming map accidentally
    )

    for _, row in df_map.iterrows():
        c = int(row["cluster"])
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=COLORS[c % len(COLORS)],
            fill=True,
            fill_opacity=0.65,
            tooltip=f"Cluster {c} | {row.get('primary_type', '')}",
        ).add_to(fmap)

    # returned_objects=[] prevents re-render on every interaction
    st_folium(fmap, width=700, height=460, returned_objects=[])

with bar_col:
    st.markdown("**Volume per Cluster**")
    cluster_counts = (
        df_filtered[df_filtered["cluster"] != -1]["cluster"]
        .value_counts().sort_index().reset_index()
    )
    cluster_counts.columns = ["Cluster", "Count"]
    cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)
    cluster_counts["Label"] = cluster_counts["Count"].apply(lambda x: f"{x:,}")

    fig = px.bar(
        cluster_counts,
        x="Count", y="Cluster",          # horizontal bar — labels never get clipped
        orientation="h",
        color="Cluster",
        color_discrete_sequence=COLORS,
        text="Label",
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        showlegend=False,
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="Cluster",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

if model_key == "dbscan":
    noise = metrics.get("noise_count", 0)
    st.caption(
        f"Unclassified (noise) points excluded: {noise:,} "
        f"({metrics.get('noise_fraction', 0)*100:.1f}%)"
    )

# ══════════════════════════════════════════════
# SECTION 3 — TECHNICAL METRICS (for data team, below)
# ══════════════════════════════════════════════
with st.expander("🔬 Technical Model Metrics", expanded=False):
    t1, t2, t3 = st.columns(3)
    t1.metric("Algorithm",    model_display)
    t2.metric("Silhouette",
              f"{metrics.get('silhouette', 0):.4f}" if metrics.get("silhouette") else "N/A",
              "Higher = better (max 1.0)")
    if model_key == "dbscan":
        t3.metric("Noise Fraction",
                  f"{metrics.get('noise_fraction', 0)*100:.1f}%", "Target < 10%")
    else:
        t3.metric("Davies-Bouldin",
                  f"{metrics.get('davies_bouldin', 0):.4f}", "Lower = better")
    st.caption(
        "Silhouette score measures how well each point fits its own cluster vs neighbours. "
        "Values > 0.5 indicate strong clusters; 0.26–0.41 is moderate (typical for crime data "
        "which has overlapping spatial patterns)."
    )
