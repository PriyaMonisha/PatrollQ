# PatrolIQ — Project Intelligence

## Who I Am
You are a senior ML engineer working on "PatrolIQ" — a production-grade urban safety
intelligence platform built on the Chicago crime dataset using unsupervised machine learning.
This is a GUVI HCL capstone project for public safety analytics.

## My Working Style (STRICT)
Before writing ANY code:
1. State your plan in clear steps
2. Call out assumptions explicitly
3. Ask: "Ready to start Section N. Please share the equivalent file(s) from your EMI project."
4. WAIT for user to share reference files
5. Only after reference files are reviewed — write the code

If a task has multiple valid approaches, present them with tradeoffs. Let the user choose.
Never silently pick one.

## Section Completion Checklist (MANDATORY — no exceptions)
At the end of EVERY section, before declaring it complete or moving on:
- [ ] All files saved
- [ ] Tests passed (if applicable)
- [ ] Artifacts validated (if applicable: CSV files present, JSON metrics written)
- [ ] `git add` all changed files
- [ ] `git commit -m "section-X: description"`
- [ ] `git log --oneline` — confirm commit appears
- [ ] `git status` — confirm working tree clean
- [ ] CLAUDE.md progress table updated (move section from In Progress → Completed)
- [ ] Next section dependencies confirmed
Only AFTER all boxes checked → ask if ready for next section.

## How to Start Each Chat
I will open with:
"Continuing PatrolIQ. Completed: [X]. Now building: [Y]."
You respond with your plan for Y — steps, risks, decisions — and WAIT
for reference files before writing any code.

---

## Current Status
**Active Section:** Section 10 — Docker + CI
**Last Working File:** src/utils/helpers.py, scripts/run_full_pipeline.py, notebooks/08_mlflow_experiments.py
**Last Decision Made:** REDUCTION_FEATURES = FULL_FEATURES (14 engineered features, not raw lat/lon); temporal K=4 confirmed by silhouette (0.26)

---

## ⚠️ MANDATORY AFTER EVERY SECTION (no exceptions)
**Update this file.** Move the section from Remaining → Completed. Update Current Status.
This is checked in the Section Completion Checklist below. Do NOT skip.

---

## Progress Tracker

### Completed ✅
- [x] Section 0: .claude/ agent, command, rules files (commit 7e4309d)
- [x] Section 1: config.py, requirements.txt, .gitignore, .env.example, CLAUDE.md, git init
- [x] Section 1b: FAST_MODE in config.py, anchored .gitignore, CLAUDE_CONTEXT.md, PROJECT_KICKOFF_CHECKLIST.md
- [x] Section 2: src/data/loader.py, src/data/preprocessor.py, notebooks/01_data_acquisition.py (commit 1f9fd97)
- [x] Section 2b: notebooks/00_data_cleaning.py, initial_analysing.ipynb, OOM fix via chunked streaming load (commit d8046de, f417cd9)
- [x] Section 3: notebooks/03_eda.py — 7 charts verified + committed (commit 8f9675c, 6277747)
- [x] Section 4: src/features/engineer.py, notebooks/04_feature_engineering.py — verified (commit 3fb59df)
- [x] Section 5: src/models/geographic_clustering.py, notebooks/05_geographic_clustering.py (commit 3046ebe) — K-Means sil=0.41, DBSCAN noise=3.8% PASS
- [x] Section 6: src/models/temporal_clustering.py, notebooks/06_temporal_clustering.py (commit db05bb2) — K=4 sil=0.26
- [x] Section 7: src/models/dimensionality_reduction.py, notebooks/07_dimensionality_reduction.py (commit 89303f4) — PCA 35.9% (FAST_MODE), t-SNE KL=1.31
- [x] Section 8: src/utils/helpers.py, scripts/run_full_pipeline.py, notebooks/08_mlflow_experiments.py (commit cb6dc56) — 16 runs exported, PatrolIQ_TemporalClustering registered
- [x] Section 9: streamlit_app.py, pages/1–5 (commit 9dc6243) — Folium map, temporal heatmap, PCA/tSNE scatter, MLflow table, about page

### In Progress 🔄
(none)

### Remaining 📋
- [ ] Section 8: src/utils/helpers.py, scripts/run_full_pipeline.py, notebooks/08_mlflow_experiments.py
- [ ] Section 9: streamlit_app.py, pages/1–5
- [ ] Section 10: Dockerfile, docker-compose.yml, .github/workflows/ci.yml

---

## Project Architecture (Locked — Do Not Change Without Explicit Decision)

### Two-Phase Pipeline
**Phase A (Local):** Download → Sample → Preprocess → Feature Engineer → Train Models → Export Artifacts + MLflow
**Phase B (Cloud):** Streamlit loads pre-computed artifacts → renders interactive visualizations

Rationale: Training 3 clustering algorithms on 500K records would OOM on Streamlit Cloud.
Streamlit NEVER trains models. It only loads CSVs and JSONs from artifacts/.

### Stack
Python 3.11 | scikit-learn | scipy | MLflow | Streamlit 1.37.0
Folium + streamlit-folium (maps) | Plotly (charts) | Docker + docker-compose
pytest | GitHub Actions CI

### Data Facts
- Full dataset: 7.8M records (2001–2025), 22 columns
- Sample: 500,000 most-recent records
- Saved as: data/processed/chicago_crime_500k.csv.gz (~25–35MB)
- Geographic range: Lat 41.6–42.0, Lon -87.9 to -87.5 (Chicago)
- Crime categories: 33 distinct Primary_Type values

### Locked Decisions
- RANDOM_STATE = 42 everywhere — models, sampling, splits
- ALL hyperparameters in config.py — never hardcoded inline
- Hierarchical clustering: subsample 10K (full 500K → ~200GB RAM for linkage matrix)
- t-SNE: PCA → 50 components first, then t-SNE on 50K subsample (O(n²) issue)
- CSV format (gzipped): pandas reads .csv.gz transparently
- MLflow tracking URI: sqlite:///mlruns/mlflow.db (consistent everywhere)
- No print() in src/ — use logging
- Type hints on all function signatures
- Streamlit version: 1.37.0

### Artifact Strategy
All ML results pre-computed and committed to git:
```
artifacts/geographic/   → kmeans/dbscan/hierarchical labels.csv + metrics.json
artifacts/temporal/     → temporal_kmeans labels.csv + metrics.json
artifacts/dimensionality/ → pca_2d.csv, tsne_2d.csv, variance.json, loadings.json
artifacts/mlflow_exports/ → all_runs.json, best_models.json
```

### Target Metrics (from GUVI PDF)
- Geographic K-Means silhouette score: > 0.5
- PCA explained variance (2–3 components): ≥ 70%
- t-SNE: visually distinct clusters (qualitative)
- DBSCAN noise fraction: < 10%
- MLflow: ≥ 6 runs logged across all experiments

---

## Critical Rules from EMI Project Lessons (Applied to PatrolIQ)

These rules were learned the hard way. Violating them causes silent failures.

| # | Rule | Why (EMI lesson) |
|---|---|---|
| 1 | Use `/data/raw/` in .gitignore (anchored with `/`) | `data/raw/` excluded `src/data/` too → ModuleNotFoundError on cloud (DEP-01) |
| 2 | Use `pages/` routing — not `st.navigation()` | `st.navigation()` + wrong Streamlit version = crash loop on cloud (DEP-02) |
| 3 | NumpyEncoder on every `json.dump()` call | sklearn returns `np.float64` → `TypeError` in `json.dumps` (CQ-03) |
| 4 | `FAST_MODE = True` at top of every training file | Accidentally ran full pipeline in dev; wasted time (TR-02) |
| 5 | Never fit Hierarchical on full 500K | Ward linkage = O(n²) memory. 500K → ~200GB RAM (PQ-01) |
| 6 | Never run t-SNE on full 500K | O(n²) complexity → hours or OOM. Use PCA first + 50K subsample (PQ-02) |
| 7 | Streamlit Cloud reads `requirements.txt` by default | `streamlit-requirements.txt` ignored without explicit config (DEP-02) |
| 8 | No `model.fit()` in `pages/` files | Pages are display-only — training on cloud would OOM and violates arch |
| 9 | DBSCAN eps is in DEGREES not km | Document conversion: 0.008° ≈ 662m at Chicago lat 42°N (PQ-07) |
| 10 | Folium map: subsample to ≤50K points | 500K CircleMarkers crash browser tabs (PQ-04) |

---

## Code Standards (Enforced by .claude/rules/ — No Exceptions)

### Every Source File
```python
# filename: src/models/geographic_clustering.py
# purpose:  Geographic crime hotspot clustering (KMeans, DBSCAN, Hierarchical)
# version:  1.0
```

### Constants
```python
RANDOM_STATE = 42      # all models, splits, sampling — from config.py
```

### Imports
- stdlib → third-party → internal (src.*) — one blank line between groups
- No wildcard imports

### Functions
- Type-annotated signatures: `def fn(df: pd.DataFrame, k: int) -> dict:`
- One-line docstring when function name is not self-explanatory

### Metrics & Artifacts
- Metrics stored to 6 decimal places in JSON
- Labels saved as CSV in artifacts/
- Figures saved to docs/figures/ with section prefix

### Forbidden
- `print()` in src/ (use logging)
- Hardcoded paths (use config.py Path constants)
- Hardcoded hyperparameters (use config.py constants)
- `model.fit()` in any pages/ file
- Missing MLflow logging in training code
