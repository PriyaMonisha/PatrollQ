# PatrolIQ — Project Kickoff Checklist
*Adapted from EMI Predict AI lessons. Use before writing the first line of code for any new section.*

---

## Before Writing Any Code

### Python & Dependencies
- [ ] Confirm Python 3.11 in active venv: `python --version`
- [ ] Pin **all** library versions in `requirements.txt`. Run `pip freeze` after first install.
- [ ] Verify Streamlit Cloud reads the correct requirements file:
  - Streamlit Cloud reads `requirements.txt` by default — NOT `streamlit-requirements.txt`
  - Either configure app settings to point to `streamlit-requirements.txt`, OR ensure `requirements.txt` is cloud-compatible
  - **Never assume** a non-standard filename is used without explicit configuration
- [ ] Check Streamlit API version compatibility:
  - `st.navigation()` requires Streamlit ≥ 1.36.0 — we pin 1.37.0 so it's safe BUT
  - Use `pages/` folder routing instead — it works with ALL versions and avoids risk
- [ ] Verify `streamlit-folium` version (0.22.0) supports `st_folium()` — not the deprecated `folium_static()`

### Data & Architecture
- [ ] Confirm data file exists locally before running anything:
  - `data/raw/chicago_crimes.csv` (full 7.8M — gitignored, download from Chicago Data Portal)
  - `data/processed/chicago_crime_500k.csv.gz` (after first pipeline run — committed to git)
- [ ] Confirm two-phase architecture is respected:
  - Phase A: local training → generate artifacts → commit to git
  - Phase B: Streamlit loads artifacts only — no training on cloud
  - **Grep pages/*.py for model.fit() before every commit** — this is a CRITICAL violation

### Training Design
- [ ] Define `FAST_MODE = True` in `config.py` BEFORE writing any training script
  - FAST_MODE = True: subsampled runs, quick verification
  - FAST_MODE = False: full 500K pipeline, production artifacts
  - Flip to False ONLY for final `run_full_pipeline.py` execution
- [ ] Confirm `RANDOM_STATE = 42` in `config.py` — referenced by all models
- [ ] Confirm `HIERARCHICAL_SUBSAMPLE = 10_000` — NEVER set to full 500K (OOM risk)
- [ ] Confirm `TSNE_SUBSAMPLE = 50_000` — NEVER run t-SNE on full 500K
- [ ] Confirm `NumpyEncoder` exists in `src/utils/helpers.py` before any JSON artifact write
  - sklearn metrics return `np.float64` → crashes `json.dumps()` without custom encoder

### Geographic Data Validation
- [ ] Chicago latitude bounds: 41.6 to 42.0 — assert after preprocessing
- [ ] Chicago longitude bounds: -87.9 to -87.5 — assert after preprocessing
- [ ] DBSCAN eps sanity check before running:
  ```python
  import math
  eps_km = 0.008 * 111_320 * math.cos(math.radians(41.85))
  print(f"DBSCAN eps = {eps_km:.0f}m")  # should print ~660m
  ```
- [ ] Null lat/lon count = 0 after preprocessing (these rows are dropped in step 2)

### Git & Branch Strategy
- [ ] Branch strategy confirmed:
  - `master` → source + committed artifacts → Streamlit Cloud
  - No separate deploy branch needed (no large binary model PKLs)
- [ ] `.gitignore` uses anchored patterns:
  ```
  /data/raw/     ← correct (anchored)
  /mlruns/       ← correct (anchored)
  data/raw/      ← WRONG (could exclude src/data/raw/ if it existed)
  ```
- [ ] Commit flow: all changes to `master` first; never make independent fixes on two branches

---

## Before Writing Docker Files

- [ ] `ENV PYTHONPATH=/app` in the Streamlit `Dockerfile` — required for `src/` layout
- [ ] Non-root user in Dockerfile:
  ```dockerfile
  RUN useradd -m appuser && USER appuser
  ```
- [ ] Multi-stage build: builder stage → runtime stage (keeps image lean)
- [ ] MLflow Docker service uses absolute path for SQLite:
  ```yaml
  command: mlflow server --backend-store-uri sqlite:////mlruns/mlflow.db
  # Four slashes = absolute path inside container. Three slashes = relative = unreliable.
  ```
- [ ] `.dockerignore` excludes: `/data/raw/`, `mlruns/`, `*.pkl`, `*.joblib`, `__pycache__/`, `.git/`
- [ ] `.dockerignore` does NOT accidentally exclude: `src/`, `artifacts/`, `data/processed/`

---

## Before Docker Build

- [ ] Run `docker build --no-cache .` once to verify full build from scratch
- [ ] Verify image size < 1GB after first build (check for accidentally copied data files)
- [ ] Confirm PYTHONPATH is set: `docker run --rm <image> python -c "import src; print('OK')"`

---

## Before GitHub Push

- [ ] `git status` — read every line. Source files under `src/` should NEVER appear as ignored.
- [ ] No files > 50MB in the commit (GitHub hard limit is 100MB):
  ```bash
  find . -name "*.csv" -size +50M
  find . -name "*.pkl" -size +50M
  ```
- [ ] `data/processed/chicago_crime_500k.csv.gz` is ≤50MB (target ~25-35MB)
- [ ] No secrets: `git grep -i "api_key\|password\|secret"` returns nothing
- [ ] Artifacts directory populated (all expected CSV/JSON files present):
  - `artifacts/geographic/kmeans_labels.csv`
  - `artifacts/geographic/dbscan_labels.csv`
  - `artifacts/geographic/hierarchical_labels.csv`
  - `artifacts/geographic/geo_cluster_metrics.json`
  - `artifacts/temporal/temporal_kmeans_labels.csv`
  - `artifacts/temporal/temporal_metrics.json`
  - `artifacts/dimensionality/pca_2d.csv`
  - `artifacts/dimensionality/tsne_2d.csv`
  - `artifacts/dimensionality/pca_explained_variance.json`
  - `artifacts/mlflow_exports/all_runs.json`

---

## Before Streamlit Cloud Deployment

- [ ] Streamlit Cloud app settings configured to use `streamlit-requirements.txt` (not default `requirements.txt`)
  - OR verify `requirements.txt` contains all needed packages for Streamlit Cloud
- [ ] `streamlit_app.py` does NOT call `st.navigation()` — uses `pages/` folder auto-routing
- [ ] All `pages/*.py` use `@st.cache_data` on data loading calls
- [ ] Artifacts directory committed to git (Streamlit Cloud has no way to generate them)
- [ ] No `model.fit()`, `model.transform()`, or heavy computation in any `pages/*.py`
- [ ] Folium map confirmed to use ≤50K points subsample (not the full 500K labels CSV)
- [ ] Test the full app locally first: `streamlit run streamlit_app.py`
  - All 5 pages load without error
  - Folium map renders (map tile loads)
  - Plotly charts render
  - Sidebar filters work

---

## Before Marking a Section Complete

- [ ] All source files have the mandatory header block:
  ```python
  # filename: src/models/geographic_clustering.py
  # purpose:  Geographic crime hotspot clustering (KMeans, DBSCAN, Hierarchical)
  # version:  1.0
  ```
- [ ] No `print()` statements in `src/` (use `logging`): `grep -rn "print(" src/`
- [ ] No hardcoded paths in `src/` (use `config.py`): `grep -rn '"data/' src/`
- [ ] All function signatures have type hints
- [ ] `git status` shows working tree clean after commit
- [ ] CLAUDE.md progress tracker updated (move section to Completed ✅)

---

## PatrolIQ ML Targets (From GUVI PDF)

| Target | Value |
|---|---|
| K-Means silhouette score | > 0.5 |
| PCA explained variance | ≥ 70% in 2–3 components |
| DBSCAN noise fraction | < 10% |
| Temporal clusters identified | 3–5 meaningful patterns |
| MLflow runs logged | ≥ 6 across all experiments |
| Streamlit app pages | 5 functional, interactive pages |

---

## Environment Variable Reference (PatrolIQ)

| Variable | Local | Docker Compose | Streamlit Cloud |
|---|---|---|---|
| `PYTHONPATH` | `$env:PYTHONPATH="."` | `ENV PYTHONPATH=/app` in Dockerfile | N/A |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | `sqlite:////mlruns/mlflow.db` (4 slashes) | N/A (MLflow is local only) |
