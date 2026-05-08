---
paths:
  - "src/models/**/*.py"
  - "src/features/**/*.py"
  - "src/evaluation/**/*.py"
---

# ML Code Rules — PatrolIQ

## Absolute Rules (Never Override Without Explicit User Decision)
- RANDOM_STATE = 42 on all algorithms: KMeans(random_state=42), etc.
- All hyperparameters from config.py — never hardcode K, eps, min_samples inline
- All experiments logged to MLflow (mandatory)
- Tracking URI: sqlite:///mlruns/mlflow.db (always relative, always consistent)
- Metrics stored to 6 decimal places

## Hierarchical Clustering Special Rule
- NEVER fit AgglomerativeClustering or scipy.cluster.hierarchy.linkage on full 500K
- Always subsample to HIERARCHICAL_SUBSAMPLE (= 10,000) first
- Assign full dataset by finding nearest centroid (KNN approach)
- Comment WHY in the code: "# Ward linkage on 500K requires ~200GB RAM — must subsample"

## t-SNE Special Rule
- NEVER run t-SNE directly on 500K records (O(n²) complexity)
- Always: PCA first (→ 50 components) → t-SNE on TSNE_SUBSAMPLE (= 50,000)
- Comment WHY in the code: "# t-SNE is O(n²) — PCA pre-reduction + subsample is standard practice"
- Stratified subsample by Primary_Type to preserve distribution

## Feature Set Discipline
- GEO_FEATURES = ['lat_norm', 'lon_norm'] — for geographic clustering only
- TEMPORAL_FEATURES = cyclical encodings — for temporal clustering only
- FULL_FEATURES = all engineered features — for PCA/t-SNE only
- Never mix feature sets between algorithms without explicit decision

## MLflow Logging (Mandatory)
Every training run must log:
- All algorithm params: mlflow.log_params(...)
- All evaluation metrics: silhouette, davies_bouldin, calinski_harabasz
- Tags: section, algorithm, feature_set, data_version, project="patroliq"
- Artifact: the output labels CSV
- Figure: cluster visualization

## Forbidden Patterns
- model.fit() or transform() calls in Streamlit pages (pages/ directory)
- GridSearchCV (use manual sweep or Optuna if needed)
- Silencing warnings with warnings.filterwarnings("ignore") at module level
- Hardcoded numeric hyperparameters (use config.py constants)
- Training code without MLflow logging block
