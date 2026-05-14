# filename: pages/3_Dimensionality_Reduction.py
# purpose:  PCA and t-SNE 2D projections of crime feature space
# version:  1.0

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR

DIM_DIR = ARTIFACTS_DIR / "dimensionality"
GEO_DIR = ARTIFACTS_DIR / "geographic"
TMP_DIR = ARTIFACTS_DIR / "temporal"

st.set_page_config(page_title="Dimensionality Reduction — PatrolIQ",
                   page_icon="📊", layout="wide")

@st.cache_data(ttl=3600)
def load_pca_2d() -> pd.DataFrame:
    return pd.read_csv(DIM_DIR / "pca_2d.csv")

@st.cache_data(ttl=3600)
def load_pca_metrics() -> dict:
    with open(DIM_DIR / "pca_metrics.json") as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_tsne_2d() -> pd.DataFrame:
    return pd.read_csv(DIM_DIR / "tsne_2d.csv")

@st.cache_data(ttl=3600)
def load_tsne_metrics() -> dict:
    with open(DIM_DIR / "tsne_metrics.json") as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_geo_labels() -> pd.Series:
    df = pd.read_csv(GEO_DIR / "kmeans_labels.csv")
    return df["cluster"]

@st.cache_data(ttl=3600)
def load_temp_labels() -> pd.Series:
    df = pd.read_csv(TMP_DIR / "kmeans_labels.csv")
    return df["cluster"]

# ── Load ──────────────────────────────────────────────────────
pca_2d      = load_pca_2d()
pca_metrics = load_pca_metrics()
tsne_2d     = load_tsne_2d()
tsne_metrics= load_tsne_metrics()
geo_labels  = load_geo_labels()
temp_labels = load_temp_labels()

# ── Header ────────────────────────────────────────────────────
st.title("📊 Dimensionality Reduction")
st.caption(
    "PCA and t-SNE projections of the 14-feature crime space "
    "(lat_norm, lon_norm, cyclical time encoding, severity, crime type codes)"
)

tab_pca, tab_tsne = st.tabs(["PCA — Principal Component Analysis", "t-SNE"])

# ════════════════════════════════════════════
# PCA TAB
# ════════════════════════════════════════════
with tab_pca:
    evr = pca_metrics["explained_variance_ratio"]
    cum = pca_metrics["cumulative_variance"]
    target = pca_metrics["variance_target"]
    met_target = pca_metrics["variance_target_met"]

    mp1, mp2, mp3 = st.columns(3)
    mp1.metric("Components",          pca_metrics["n_components"])
    mp2.metric("Cumulative Variance", f"{cum*100:.1f}%",
               f"Target {target*100:.0f}% — {'MET' if met_target else 'below (FAST_MODE)'}")
    mp3.metric("Samples",             f"{pca_metrics['n_samples']:,}")

    col_scatter, col_scree = st.columns([2, 1])

    with col_scatter:
        # Color by geo cluster — align by index
        n = min(len(pca_2d), len(geo_labels))
        plot_df = pca_2d.iloc[:n].copy()
        plot_df["Geo Cluster"] = geo_labels.iloc[:n].astype(str).values

        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="Geo Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="PCA 2D Projection — Colored by Geographic Cluster",
            opacity=0.5, render_mode="webgl",
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(height=450, legend_title="Geo Cluster")
        st.plotly_chart(fig, use_container_width=True)

    with col_scree:
        pc_labels = [f"PC{i+1}" for i in range(len(evr))]
        cum_vals  = []
        running   = 0.0
        for v in evr:
            running += v
            cum_vals.append(running)

        import plotly.graph_objects as go
        fig_scree = go.Figure()
        fig_scree.add_bar(x=pc_labels, y=[v*100 for v in evr],
                          name="Individual", marker_color="#377eb8")
        fig_scree.add_scatter(x=pc_labels, y=[v*100 for v in cum_vals],
                              mode="lines+markers", name="Cumulative",
                              line=dict(color="#e41a1c", width=2))
        fig_scree.add_hline(y=target*100, line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Target {target*100:.0f}%")
        fig_scree.update_layout(
            title="Scree Plot",
            yaxis_title="Variance Explained (%)",
            height=450, legend=dict(x=0.5, y=0.1),
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    st.info(
        f"**Note:** PCA explains {cum*100:.1f}% cumulative variance with "
        f"{pca_metrics['n_components']} components. "
        "In FAST_MODE (3 months), seasonal and yearly patterns are absent — "
        "expect higher variance in production (500K records, 4+ years)."
    )

# ════════════════════════════════════════════
# t-SNE TAB
# ════════════════════════════════════════════
with tab_tsne:
    mt1, mt2, mt3 = st.columns(3)
    mt1.metric("Subsample",     f"{tsne_metrics['n_subsample']:,}")
    mt2.metric("KL Divergence", f"{tsne_metrics['kl_divergence']:.4f}",
               "Lower = better embedding")
    mt3.metric("Perplexity",    tsne_metrics["perplexity"])

    col_geo, col_temp = st.columns(2)

    with col_geo:
        n_tsne = len(tsne_2d)
        geo_sub = geo_labels.iloc[:n_tsne] if len(geo_labels) >= n_tsne else geo_labels
        plot_t = tsne_2d.iloc[:len(geo_sub)].copy()
        plot_t["Geo Cluster"] = geo_sub.astype(str).values

        fig_tg = px.scatter(
            plot_t, x="tSNE1", y="tSNE2", color="Geo Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="t-SNE — Colored by Geographic Cluster",
            opacity=0.5, render_mode="webgl",
        )
        fig_tg.update_traces(marker=dict(size=3))
        fig_tg.update_layout(height=430)
        st.plotly_chart(fig_tg, use_container_width=True)

    with col_temp:
        temp_sub = temp_labels.iloc[:n_tsne] if len(temp_labels) >= n_tsne else temp_labels
        plot_tt = tsne_2d.iloc[:len(temp_sub)].copy()
        plot_tt["Temporal Cluster"] = temp_sub.astype(str).values

        fig_tt = px.scatter(
            plot_tt, x="tSNE1", y="tSNE2", color="Temporal Cluster",
            color_discrete_sequence=px.colors.qualitative.Set1,
            title="t-SNE — Colored by Temporal Cluster",
            opacity=0.5, render_mode="webgl",
        )
        fig_tt.update_traces(marker=dict(size=3))
        fig_tt.update_layout(height=430)
        st.plotly_chart(fig_tt, use_container_width=True)
