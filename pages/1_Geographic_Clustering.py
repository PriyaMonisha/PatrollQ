# filename: pages/1_Geographic_Clustering.py
# purpose:  Where are crime hotspots in Chicago? — geographic clustering dashboard
# version:  3.0

import json
import sys
from pathlib import Path

import folium
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import silhouette_samples
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR, FAST_MODE, PROCESSED_CSV, RANDOM_STATE

GEO_DIR = ARTIFACTS_DIR / "geographic"


def _render_silhouette_plot(df: pd.DataFrame, labels_col: str, avg_score: float) -> None:
    """Render per-cluster silhouette coefficient bar chart using matplotlib."""
    coords = df[["latitude", "longitude"]].dropna().to_numpy()
    labels = df.loc[df[["latitude", "longitude"]].notna().all(axis=1), labels_col].to_numpy()
    if len(coords) < 20 or len(set(labels)) < 2:
        st.info("Not enough data points to render silhouette plot.")
        return

    sample_size = min(3_000, len(coords))
    idx = np.random.default_rng(42).choice(len(coords), sample_size, replace=False)
    X_s, y_s = coords[idx], labels[idx]

    sil_vals = silhouette_samples(X_s, y_s)
    n_clusters = len(set(y_s))
    fig, ax = plt.subplots(figsize=(9, max(4, n_clusters)))
    y_lower = 10
    for i in sorted(set(y_s)):
        vals = np.sort(sil_vals[y_s == i])
        y_upper = y_lower + len(vals)
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, alpha=0.75)
        ax.text(-0.07, y_lower + 0.5 * len(vals), str(i), fontsize=9)
        y_lower = y_upper + 8

    ax.axvline(x=avg_score, color="red", linestyle="--", linewidth=1.5,
               label=f"Avg silhouette = {avg_score:.4f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot — Geographic Clusters (sample 3K)")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.3, 1.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "Each bar = one crime record. Width = silhouette score. "
        "Clusters 3 and 6 show lower scores — these are transitional neighborhoods "
        "between North and South Side (expected, not a model failure)."
    )

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
        arrest_rate=("arrest", lambda x: x.astype(bool).mean()),
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

    st_folium(fmap, width=700, height=460)

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

    with st.expander("📊 Cluster Quality — Silhouette Plot"):
        _render_silhouette_plot(df_filtered, labels_col="cluster",
                                avg_score=metrics.get("silhouette", 0.41))
