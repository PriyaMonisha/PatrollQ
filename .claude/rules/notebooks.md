---
paths:
  - "notebooks/**/*.py"
  - "notebooks/**/*.ipynb"
---

# Notebook Rules — PatrolIQ

## Naming Convention (match section numbers)
01_data_acquisition.py
02_eda.py
03_feature_engineering.py
04_geographic_clustering.py
05_temporal_clustering.py
06_dimensionality_reduction.py
07_mlflow_experiments.py

## Required Structure (every notebook)
Cell 1 — Header:
  # filename, purpose, section number, version

Cell 2 — Setup:
  imports + logging config + RANDOM_STATE = 42

Cell 3 — Load Data:
  always from data/processed/ — NEVER from data/raw/

Last Cell — Summary:
  Key findings from this section
  Decisions made (e.g., optimal K, best algorithm)
  Next section dependencies

## Visualization Standards (Locked Palettes)
- Crime type comparisons: px.colors.qualitative.Set2
- Heatmaps: color_continuous_scale='YlOrRd' (yellow=low, red=high risk)
- Geographic clusters: px.colors.qualitative.Bold
- Figure size: (900, 500) for Plotly, (12, 6) for matplotlib
- Always save figures: plt.savefig('docs/figures/0N_chartname.png', dpi=150, bbox_inches='tight')

## Notebooks Are Not Production
- Notebooks = exploration + documentation
- All reusable logic → src/
- Notebooks must run top-to-bottom without errors before commit
- No sensitive data (credentials, personal info) ever in notebooks
