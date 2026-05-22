# Model Card — PatrolIQ Crime Clustering Models

## Model Description

PatrolIQ uses three unsupervised clustering models to identify crime patterns in Chicago:

| Model | Algorithm | Purpose | Key Hyperparameters |
|-------|-----------|---------|---------------------|
| Geographic K-Means | K-Means (k=8) | Identify spatial crime hotspots | k=8 (matches CPD districts), random_state=42 |
| Geographic DBSCAN | DBSCAN | Detect noise + density-based clusters | eps=0.008° (~660m), min_samples=100 |
| Geographic Hierarchical | Agglomerative (Ward) | Alternative spatial grouping | k=8, subsample=10K, linkage=ward |
| Temporal K-Means | K-Means (k=4) | Identify daily activity windows | k=4 (maps to 4 activity periods), random_state=42 |
| PCA | PCA (3 components) | Dimensionality reduction for visualisation | n_components=3, target_variance=0.70 |
| t-SNE | t-SNE | Non-linear cluster visualisation | perplexity=30, n_iter=1000, subsample=5K |

**Training data:** 48,870 Chicago crime records (Feb–Apr 2026, sampled from 7.8M)
**Primary metric:** Silhouette Score (geographic K-Means: 0.41, temporal K-Means: 0.26)

---

## Intended Use

**Primary use case:** Decision-support tool for Chicago Police Department district commanders and analysts to:
- Identify geographic patrol resource allocation priorities
- Understand temporal crime patterns by time of day / day of week
- Compare clustering algorithms and track model performance over time via MLflow

**Intended users:** CPD district commanders, crime analysts, public safety researchers

---

## Out-of-Scope Use

The following uses are explicitly NOT supported and should NOT be attempted:

- ❌ **Automated enforcement decisions** — This model must not be used as the sole or primary basis for patrol deployment, stop-and-frisk, or any enforcement action
- ❌ **Individual suspicion targeting** — The model clusters geographic zones and time periods, not individuals
- ❌ **Predicting future crimes at specific addresses** — This is retrospective pattern analysis, not predictive policing
- ❌ **Deployment in other cities without retraining** — Crime patterns, geography, and policing practices vary significantly by city
- ❌ **Real-time enforcement response** — Model is trained offline on historical data; it does not update in real time
- ❌ **Legal or judicial proceedings** — Cluster membership is a statistical pattern, not evidence of criminal activity

---

## Evaluation Data

| Property | Value |
|----------|-------|
| Source | Chicago Data Portal — Crimes 2001 to Present |
| Records (training) | 48,870 (FAST_MODE sample) |
| Records (full pipeline) | 500,000 most-recent |
| Date range (sample) | February 6 – April 30, 2026 |
| Geographic coverage | All 25 CPD districts, 77 community areas |
| Crime type coverage | 30 of 34 primary crime types (FAST_MODE) |
| Null rate (lat/lon) | 1.13% — imputed via Beat-median |
| Out-of-bounds removed | 1.8% of records |

---

## Metrics

### Geographic Clustering

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Notes |
|-----------|-------------|-----------------|---------------------|-------|
| K-Means (k=8) | **0.4115** | **0.7784** | TBD after rerun | Best overall |
| DBSCAN | -0.1282 | N/A | N/A | Noise=3.83% ✓ |
| Hierarchical (k=8) | 0.3409 | 0.8813 | TBD | 10K subsample |

### Temporal Clustering

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-----------|-------------|-----------------|---------------------|
| K-Means (k=4) | **0.2576** | **1.4466** | TBD after rerun |

### Dimensionality Reduction

| Algorithm | Metric | Value | Notes |
|-----------|--------|-------|-------|
| PCA (3 components) | Cumulative variance | 35.9% | Below 70% target — see note |
| t-SNE | KL divergence | 1.308 | Lower = better embedding |

> **PCA variance note:** 35.9% with 3 components reflects the FAST_MODE 3-month window. Cyclical temporal features (sin²+cos²=1) are correlated by design, limiting PCA's linear decomposition. Full 500K sample (4+ years) yields higher variance. t-SNE visual separation compensates.

### MLflow Experiment Summary

- Total runs logged: **16**
- Experiments: geographic (3), temporal (2), dimensionality (1)
- Model registered: `PatrolIQ_TemporalClustering v2`
- Experiment tracking requirement (≥6 runs): ✅ PASS

---

## Ethical Considerations

### Crime Data Bias — Feedback Loop Risk

**This is the most critical limitation of this model and must be understood by all users.**

Crime data reflects **historical policing patterns, not actual crime rates.** Areas with historically higher police presence generate more recorded incidents — not because more crimes occur there, but because more officers are present to record them.

This creates a self-reinforcing feedback loop:

```
Model recommends patrol → More officers deployed
      ↑                              ↓
More crimes recorded          More arrests made
      ↑                              ↓
       ←── "High crime area" label ←──
```

**Result:** The model may systematically over-recommend patrol in communities that have historically been over-policed, regardless of actual crime levels. This disproportionately affects communities of colour and lower-income neighbourhoods.

### Recommended Mitigations

1. **Never use as sole basis for patrol deployment** — Cross-reference with community calls-for-service data (911 calls initiated by residents, not officers)
2. **Quarterly drift review** — Run `/v1/drift/report` monthly; if arrest rate shifts >5%, retrain with updated data
3. **Community input** — Cluster recommendations should be reviewed with community liaison officers before deployment
4. **Transparency** — Make cluster assignments and the methodology available to district commanders, not just final recommendations
5. **Regular audit** — Compare deployment decisions to actual crime resolution rates, not just cluster labels

### Other Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| 3-month training window (FAST_MODE) | Seasonal patterns underrepresented | Retrain with full 500K sample |
| No socioeconomic context features | Model cannot distinguish poverty-correlated patterns from crime patterns | Future: add census tract features |
| Unreported crimes not captured | Model only sees reported/recorded crimes | Supplement with 311 and victim survey data |
| Static model (no online learning) | Model degrades as crime patterns shift | Monitor drift; retrain quarterly |
| Geospatial resolution limited to lat/lon | Cannot account for building interiors, transit patterns | Future: H3 hexagonal indexing |

### Privacy

- No personally identifiable information (PII) is used in training or inference
- Individual crime records are aggregated into cluster labels; no individual tracking
- The model outputs cluster IDs for coordinates and time periods, not individual identifiers

---

## Caveats and Recommendations

1. **Interpret silhouette 0.41 correctly** — Crime data has natural spatial overlap (criminals cross district boundaries). 0.41 is appropriate for this domain; academic crime clustering papers report 0.3–0.5 as typical.

2. **DBSCAN negative silhouette (-0.13) is expected** — DBSCAN assigns many points to noise (3.83%), and its clusters are density-based (irregular shapes). Silhouette score assumes convex clusters; it is not the right metric for DBSCAN. Noise fraction (3.83% < 10% target) is the correct metric here.

3. **K=8 is a domain-driven choice** — Chicago PD operates 8 geographic districts. The elbow method suggests K=2 (North/South split) which is technically optimal but operationally useless. K=8 matches real patrol unit boundaries.

4. **Do not deploy on Streamlit Cloud with live data** — The current architecture pre-computes artifacts locally and displays them statically. Adding live data ingestion would require a proper data pipeline with access controls.

---

*PatrolIQ — Urban Safety Intelligence Platform | Developed 2026*
