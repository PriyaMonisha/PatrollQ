---
name: run-pipeline
argument-hint: [optional: step-number to run single step]
---

Execute PatrolIQ full pipeline and validate all artifacts:

1. Check that data/processed/chicago_crime_500k.csv.gz exists
   If not — stop and print: "Download the Chicago crime CSV first, then run:
   python scripts/run_full_pipeline.py --step 1"

2. Run the pipeline (or single step if $ARGUMENTS provided):
   python scripts/run_full_pipeline.py $ARGUMENTS

3. After completion, validate all artifacts exist:
   artifacts/geographic/kmeans_labels.csv
   artifacts/geographic/dbscan_labels.csv
   artifacts/geographic/hierarchical_labels.csv
   artifacts/geographic/geo_cluster_metrics.json
   artifacts/temporal/temporal_kmeans_labels.csv
   artifacts/temporal/temporal_metrics.json
   artifacts/dimensionality/pca_2d.csv
   artifacts/dimensionality/tsne_2d.csv
   artifacts/dimensionality/pca_explained_variance.json
   artifacts/dimensionality/pca_feature_loadings.json
   artifacts/mlflow_exports/all_runs.json
   artifacts/mlflow_exports/best_models.json

4. Print: ✓ or ✗ for each artifact
5. If all present: "Pipeline complete. Ready for Streamlit app development."
6. If any missing: "Missing artifacts: [list]. Re-run: python scripts/run_full_pipeline.py"
