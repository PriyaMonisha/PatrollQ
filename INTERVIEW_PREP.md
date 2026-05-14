# PatrolIQ — Interview Preparation Guide
# Questions from India's Big Tech Companies

**Target companies:** HCL, TCS, Wipro, Infosys, Cognizant, Capgemini | Flipkart, Amazon India, Zomato, Swiggy | Google India, Microsoft India

---

## How to Use This File

- **Service companies (HCL/TCS/Wipro):** Focus on Sections 1–5 + 9. They ask concepts + project walkthrough.
- **Product companies (Flipkart/Amazon/Zomato):** Focus on Sections 3–6 + 8. They go deep on ML fundamentals and system design.
- **MNC product (Google/Microsoft):** Focus on Sections 4–8. They push on theory, scalability, and trade-offs.
- For every answer, use the **STAR format** (Situation → Task → Action → Result) when describing your work.

---

## Section 1 — Project Introduction (HR + First Technical Round)

**Q1. Tell me about your PatrolIQ project in 2 minutes.**

> PatrolIQ is a production-grade urban safety intelligence platform I built as a GUVI HCL capstone project. It applies unsupervised machine learning — K-Means, DBSCAN, Hierarchical clustering, PCA, and t-SNE — to 500,000 Chicago crime records to discover spatial and temporal crime patterns. The output is a 5-page interactive Streamlit dashboard where law enforcement can filter by crime type, view geographic hotspots on a map, explore temporal clusters (morning/night/weekend patterns), and inspect dimensionality reduction visualizations. The full ML pipeline is pre-computed locally and committed as artifacts; Streamlit only loads and visualizes — no training on the cloud. The stack includes scikit-learn, MLflow for experiment tracking, Docker for containerization, and GitHub Actions for CI.

**Q2. Why unsupervised learning for crime data? Why not a classifier?**

> Crime data doesn't have ground-truth cluster labels — nobody has pre-labelled which neighbourhoods are "hotspot type A" vs "hotspot type B." Unsupervised learning lets the data reveal its own natural groupings without us imposing our assumptions. A classifier would require labelled training data and would only confirm what we already know. Clustering discovers patterns we didn't expect — for example, that temporal patterns divide naturally into four behavioural clusters (daytime, nighttime, weekend, early-morning) even though we never defined those categories upfront.

**Q3. What problem does this solve? What is the business value?**

> Law enforcement agencies allocate patrol resources based on experience and gut feel. PatrolIQ provides data-driven answers: Where are the geographic crime hotspots? (geographic clustering → 8 zones). When do crimes happen? (temporal clustering → 4 time patterns). What crime types co-occur in the same zones? (cluster profiles). This allows commanders to schedule more patrols in high-risk zones during high-risk times instead of spreading resources uniformly — improving response efficiency and potentially deterring crime.

---

## Section 2 — Data Engineering Questions

**Q4. Your dataset is 7.8 million records and 1.7GB. How did you load it without running out of memory?**

> I used chunked loading with `pd.read_csv(path, chunksize=200_000)`. The C parser runs out of memory trying to load 1.7GB at once on machines with 8–16GB RAM. With chunked loading, I read 200K rows at a time, kept only the most recent 500K records by tracking a rolling heap, then concatenated. The key insight: sorting by date descending in-stream means I can stop reading once I have 500K recent records, rather than loading everything first.

**Q5. Why 500K records specifically? Why not all 7.8 million?**

> Two reasons. First, recency: crimes from 2001 are not useful for current policing — modern Chicago looks completely different geographically and socially from 2001. The 500K most-recent records cover approximately the past 3–4 years and are far more actionable. Second, computational constraints: clustering algorithms like K-Means and DBSCAN scale with O(n·k·i) and O(n²) respectively. On 7.8M records, training times would be hours; on 500K, they complete in minutes. Hierarchical clustering's Ward linkage is O(n²) memory — 500K records would require ~200GB RAM for the linkage matrix, which is impossible.

**Q6. You imputed null lat/lon values using Beat-median. Why not just drop them?**

> Dropping would have discarded ~1.1% of records (about 5,500 rows). Those records still have valid Primary_Type, time, district, and beat — we'd be losing legitimate crime records just because GPS failed. Beat is the smallest administrative unit in Chicago (a few city blocks), so the median lat/lon within a Beat gives ~4–6 block accuracy for imputation — good enough for geographic clustering. The alternative (district-median) would have ~1km error. Mode imputation for Ward/Community Area follows the same logic: Beat never has nulls, so it's the safest parent to group by.

---

## Section 3 — Clustering Algorithm Questions

**Q7. Explain K-Means clustering. What are its assumptions? What are its weaknesses?**

> K-Means partitions n points into K clusters by minimizing within-cluster sum of squares (WCSS). Algorithm: (1) initialize K centroids randomly, (2) assign each point to nearest centroid, (3) recompute centroid as cluster mean, (4) repeat until convergence.
>
> Assumptions: clusters are spherical (similar spread in all directions), roughly equal size, and the feature space is Euclidean. Weaknesses: sensitive to outliers (a single outlier shifts the centroid), fails on non-convex shapes (e.g., rings or crescents), requires K to be specified in advance, and non-deterministic (use random_state=42 for reproducibility). It's also O(n·k·i·d) — linear in n which is acceptable but sensitive to k and iterations.

**Q8. How did you choose K=8 for geographic clustering?**

> I used the Elbow method: plot WCSS against K from 2 to 10, look for the "elbow" where adding more clusters yields diminishing WCSS improvement. I also computed Silhouette score for each K — higher is better (range –1 to +1). K=8 gave silhouette=0.41 with a clear elbow in the WCSS curve. I validated this makes semantic sense: 8 clusters map roughly to Chicago's 8 police districts, which independently supports the choice.

**Q9. What is DBSCAN? When would you choose it over K-Means?**

> DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are densely connected — a point is a core point if it has at least `min_samples` neighbours within radius `eps`. Clusters form by chain-connecting core points. Points that can't join any cluster are labelled noise (–1).
>
> Choose DBSCAN over K-Means when: (1) you don't know K in advance, (2) clusters are non-convex shapes (streets, city boundaries are not circles), (3) you have outliers and want them explicitly flagged as noise rather than force-assigned to a cluster, (4) cluster density varies by region. For geographic crime data, DBSCAN is more natural because crime hotspots are not spherical — they follow streets and neighbourhoods. In PatrolIQ, DBSCAN noise fraction was 3.8% (under 10% target), meaning it found meaningful dense hotspots without over-labelling noise.

**Q10. DBSCAN has an `eps` parameter. How did you set it? What are the units?**

> `eps` in DBSCAN is the neighbourhood radius. When using latitude/longitude directly (not projected coordinates), eps is in **decimal degrees**, not kilometres. At Chicago's latitude (~42°N), the conversion is: km = degrees × 111.32 × cos(42°) ≈ degrees × 82.7. So eps=0.008° ≈ 662 metres. I chose 0.008° by looking at the k-distance plot (sort points by distance to their k-th nearest neighbour; the "knee" in that curve suggests a good eps). I always document this in the code comment to prevent future confusion — eps=0.008 looks like 8 metres if someone assumes kilometres.

**Q11. What is Hierarchical Clustering? Why did you subsample to 10K?**

> Hierarchical clustering builds a tree (dendrogram) by iteratively merging the two closest clusters (agglomerative, bottom-up). Ward linkage minimises the increase in total within-cluster variance when merging. It produces a hierarchy you can cut at any level to get K clusters — no need to specify K upfront.
>
> The subsample is mandatory because Ward linkage requires computing the full pairwise distance matrix: O(n²) memory. For 500K records, that's 500K × 500K × 8 bytes = ~2TB in memory — physically impossible. 10K records requires only 10K² × 8 bytes = ~800MB, which is feasible. I then assigned the remaining 490K records by finding each record's nearest cluster centroid using KNN — this is the standard approach for scaling hierarchical clustering to large datasets.

**Q12. What is Silhouette score? Your geographic score was 0.41 and temporal was 0.26. What does that mean?**

> Silhouette score measures how well each point fits its own cluster vs. the nearest neighbouring cluster. Formula: (b – a) / max(a, b), where a = mean intra-cluster distance, b = mean nearest-cluster distance. Range: –1 (wrong cluster) to +1 (perfectly separated). Rule of thumb: >0.5 = strong, 0.25–0.5 = reasonable, <0.25 = weak.
>
> Geographic score of 0.41 is reasonable — spatial clusters based on lat/lon tend to be well-separated. The GUVI PDF target was >0.5; we're below but the elbow method and semantic alignment (8 police districts) validate the choice.
>
> Temporal score of 0.26 is expected for temporal patterns — crime timing data overlaps more than geographic data. Crimes don't stop at a clean hour boundary. The clusters still reveal meaningful behavioural patterns: daytime commercial crime, nighttime violence, weekend social incidents, early-morning transit crime.

---

## Section 4 — Dimensionality Reduction Questions

**Q13. What is PCA? What does "explained variance" mean? Why is 35.9% low?**

> PCA (Principal Component Analysis) finds new orthogonal axes (principal components) that capture the maximum variance in the data, in order. PC1 has the most variance, PC2 the second most, etc. "Explained variance" is the fraction of total data variance captured by the selected components. 35.9% from 2 PCs (in FAST_MODE on 50K records) means those 2 components capture only a third of the information — significant structure is in the remaining dimensions.
>
> 35.9% is low because we have 14 engineered features (spatial, temporal, categorical encodings) that are moderately correlated but not redundant. In production mode (500K records, full feature set), PCA typically reaches 60–70% in 2–3 components. The GUVI target is ≥70% in 2–3 components. FAST_MODE is a development shortcut, not the final result.

**Q14. Why can't you run t-SNE on 500,000 records directly?**

> t-SNE has O(n log n) to O(n²) time complexity depending on the implementation (Barnes-Hut approximation vs. exact). For 500K records: even at O(n log n), that's 500K × 19 ≈ 9.5M operations per gradient step, and t-SNE runs hundreds of iterations. Memory-wise, exact t-SNE needs the full n×n pairwise similarity matrix: 500K² × 4 bytes = ~1TB. Even Barnes-Hut builds an O(n log n) space structure.
>
> In practice, t-SNE on 500K records would take several hours and likely OOM. The standard solution: first reduce to 50 dimensions with PCA (very fast), then run t-SNE on a 50K stratified subsample of the PCA output. This captures the distribution while making t-SNE feasible in ~10 minutes.

**Q15. What is the difference between PCA and t-SNE? When do you use each?**

> PCA is linear, deterministic, and preserves global structure (distances between far-apart clusters). t-SNE is non-linear, stochastic, and preserves local structure (nearby points stay nearby; far points may be distorted). PCA is used for: feature reduction before modelling, variance explained analysis, interpretable loading vectors. t-SNE is used for: 2D/3D visualisation of cluster separation, verifying cluster structure visually — you should NEVER use t-SNE components as model features. Also: PCA can handle new data (transform without refit); t-SNE requires re-fitting from scratch for any new data.

---

## Section 5 — MLflow and MLOps Questions

**Q16. What is MLflow? What did you use it for in this project?**

> MLflow is an open-source platform for managing the ML lifecycle: experiment tracking, model versioning (Model Registry), project packaging, and deployment. In PatrolIQ I used experiment tracking: every training run (geographic K-Means with K=2 through K=10, DBSCAN, Hierarchical, PCA, t-SNE) logs its parameters (K, eps, min_samples, etc.), metrics (silhouette, Davies-Bouldin, Calinski-Harabasz scores), tags (algorithm, feature_set, data_version), and artifacts (labels CSV, visualisation PNG). This gives a complete audit trail — I can reproduce any run exactly by looking at its logged parameters. The best model per algorithm is registered in the MLflow Model Registry.

**Q17. Why store MLflow tracking in SQLite and not the default file store?**

> MLflow's default file store (`mlruns/`) uses individual JSON files per metric/parameter per run. This creates hundreds of small files and makes querying slow — `mlflow.search_runs()` has to scan a directory tree. SQLite stores everything in a single `.db` file, supports proper SQL queries, and makes `mlflow.search_runs()` significantly faster for 16+ runs. It also avoids Windows path issues with the default store. The trade-off: SQLite is single-writer (fine for a single-developer project, would need PostgreSQL for a team).

**Q18. What is experiment tracking? Why does it matter in ML projects?**

> Experiment tracking is the practice of recording every training run's exact configuration (hyperparameters, data version, feature set) alongside its results (metrics, artifacts). Without it, after running 16 experiments you can't answer: "Which K gave the best silhouette?" "Did the DBSCAN I ran last Tuesday use eps=0.008 or 0.01?" "Which labels CSV corresponds to which run?" MLflow makes this automatic — every run is stamped with timestamp, parameters, metrics, and git commit hash. The Model Registry adds a layer: promote the best run to "Production" stage so downstream consumers always load the best version.

---

## Section 6 — Architecture and System Design Questions

**Q19. Explain your two-phase architecture. Why not train the model when the user opens the app?**

> Phase A (local): download data → preprocess → feature engineer → train all 5 clustering algorithms → export labels as CSVs + metrics as JSONs → log to MLflow. This runs once locally, takes 10–30 minutes, and produces 15+ artifact files committed to git.
>
> Phase B (cloud): Streamlit loads the pre-computed CSV/JSON files from `artifacts/` → renders interactive visualizations. Zero training, zero sklearn model fitting on the cloud.
>
> Why? Three reasons: (1) Memory: training K-Means on 500K records in a 512MB Streamlit Cloud container would OOM. (2) Time: every user would wait 10+ minutes for the app to load. (3) Cost: Streamlit Cloud free tier has compute limits; running sklearn training on every page load would hit those limits in hours. Pre-computing solves all three — the app loads instantly, costs nothing at serve time, and is deterministic.

**Q20. How does your Docker setup work? What's in the image?**

> Single-stage `python:3.11-slim` Dockerfile. The image contains only what Streamlit needs to serve: `streamlit_app.py`, `pages/` (5 pages), `config.py`, `src/utils/helpers.py` (NumpyEncoder for JSON display), and `artifacts/` (pre-computed clustering results baked directly into the image). The image does NOT contain: training scripts (`notebooks/`, `scripts/`), raw data (`data/raw/` — 2.2GB), model training code (`src/models/`, `src/features/`). Only `mlruns/` is volume-mounted at runtime (too large to bake, may be updated by local runs). The `.dockerignore` ensures the 2.2GB raw CSV never enters the build context.

**Q21. Why not use `requirements.txt` for Docker? Why `streamlit-requirements.txt`?**

> `requirements.txt` has the full development stack: scikit-learn, scipy, matplotlib, seaborn, pytest. These are needed for local training but not for serving the dashboard. Installing them adds ~300MB to the Docker image and ~2 minutes to build time. `streamlit-requirements.txt` has only the serve-time deps (streamlit, pandas, numpy, plotly, folium, streamlit-folium, joblib) — about half the size. This is the separation of concerns principle applied to dependency management: dev deps ≠ deploy deps.

**Q22. What does your CI pipeline do? What does it NOT do?**

> CI runs three jobs on every push to main/master: (1) Lint — flake8 with max-line-length=120 catches real bugs like empty f-strings and missing whitespace, while ignoring style-only rules like aligned assignments. (2) Unit tests — pytest on `tests/test_helpers.py`, 12 tests covering NumpyEncoder and save_json, run in ~1 second. (3) Docker build — verifies the image builds cleanly and the container starts.
>
> CI does NOT run the training pipeline (`notebooks/` or `scripts/run_full_pipeline.py`) because the data file is gitignored (it's 2.2GB). CI cannot train models without data. The pipeline is run locally before committing artifacts.

---

## Section 7 — Feature Engineering Questions

**Q23. What features did you engineer from the raw Chicago crime data?**

> From the raw 22 columns I engineered 14 features grouped into four categories:
> - **Cyclical temporal encodings**: hour, day-of-week, and month encoded as sine/cosine pairs (hour_sin, hour_cos, etc.) to preserve the circular nature of time — 11pm and 1am are close, not 22 hours apart
> - **Spatial normalization**: lat_norm, lon_norm (min-max scaled to 0–1 within Chicago bounds)
> - **Severity score**: ordinal encoding of crime types by typical severity (theft < assault < homicide)
> - **Administrative features**: district, beat, community area (label encoded)
> - **Boolean flags**: Is_Weekend, arrest_binary, domestic_binary
>
> The feature set is split by use: GEO_FEATURES (lat_norm, lon_norm only) for geographic clustering, TEMPORAL_FEATURES (cyclical encodings) for temporal clustering, and FULL_FEATURES (all 14) for PCA/t-SNE.

**Q24. Why cyclical encoding for time features instead of raw hours (0–23)?**

> If you encode hour as an integer (0–23), then hour 23 (11pm) and hour 0 (midnight) have distance 23 in feature space, even though they're 1 hour apart. K-Means computes Euclidean distances — it would think 11pm and midnight are in completely different time regions. Cyclical encoding (sin/cos) maps the hour to a point on a unit circle: `hour_sin = sin(2π × hour / 24)`, `hour_cos = cos(2π × hour / 24)`. Now 23 and 0 are adjacent points on the circle, correctly capturing temporal proximity.

---

## Section 8 — Python and Coding Questions

**Q25. You wrote a custom NumpyEncoder for JSON serialization. Why is it needed? Write it.**

> scikit-learn returns numpy scalar types (np.float64, np.int64, np.bool_) for all metrics and predictions. Python's built-in `json.dumps()` only knows about native Python types — it raises `TypeError: Object of type float64 is not JSON serializable` when it encounters a numpy scalar. The encoder converts numpy types to their Python equivalents:

```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):     return bool(obj)    # MUST be before np.integer
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return super().default(obj)
```

> Critical: `np.bool_` must be checked BEFORE `np.integer` because in numpy < 2.0, `np.bool_` is a subclass of `np.integer`. Checking integer first would convert `True` → `1` (int) instead of `True` (bool).

**Q26. How did you implement memory-efficient loading of a 2.2GB CSV file?**

```python
def load_raw_csv(path: Path, n_recent: int = 500_000) -> pd.DataFrame:
    chunks = []
    reader = pd.read_csv(path, chunksize=200_000, low_memory=False)
    try:
        for chunk in reader:
            chunk['Date'] = pd.to_datetime(
                chunk['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce'
            )
            chunks.append(chunk)
    finally:
        reader.close()   # explicit close — avoids open file handle on exception
    df = pd.concat(chunks, ignore_index=True)
    df.sort_values('Date', ascending=False, inplace=True)
    return df.head(n_recent)
```

**Q27. What is `@st.cache_data`? Why is it important?**

> Streamlit re-runs the entire page script on every user interaction (slider move, button click, selectbox change). Without caching, `pd.read_csv(large_file.csv.gz)` would re-read from disk on every interaction — making the app sluggish. `@st.cache_data` stores the function's return value keyed by its arguments. If the same function is called again with the same arguments, Streamlit returns the cached result instead of re-executing. For CSVs and JSON files that don't change, `ttl=3600` means the cache is valid for 1 hour. `@st.cache_resource` is used for model objects (joblib.load) — it caches by object identity, not by serialized value.

---

## Section 9 — Evaluation and Metrics Questions

**Q28. You have Silhouette, Davies-Bouldin, and Calinski-Harabasz. Which is most important? Why log all three?**

> No single metric is perfect:
> - **Silhouette** (higher is better, range –1 to +1): measures both cohesion and separation. Interpretable and widely cited. Weakness: favours convex, equally-sized clusters.
> - **Davies-Bouldin** (lower is better, range 0 to ∞): ratio of within-cluster scatter to between-cluster separation. Sensitive to cluster shape.
> - **Calinski-Harabasz** (higher is better): ratio of between-cluster dispersion to within-cluster dispersion. Tends to favour K-Means-style compact clusters.
>
> They sometimes disagree — a K that maximises Silhouette may not minimise Davies-Bouldin. Logging all three in MLflow lets you make an informed decision rather than trusting a single metric. For GUVI submission, Silhouette is the primary metric (explicitly in the PDF), but the others provide corroborating evidence.

**Q29. Your DBSCAN noise fraction was 3.8%. What would you do if it was 25%?**

> 25% noise means eps is too small or min_samples is too large — the algorithm is being too strict about what counts as a dense region. Remedies: (1) Increase eps (enlarge the neighbourhood radius — try the k-distance plot to find the knee in the curve). (2) Decrease min_samples. (3) Check if the data has legitimate outliers (single crimes far from any cluster) that should be noise. If domain knowledge says there are real isolated hotspots, increase eps. If 25% noise is geographically spread evenly, the data may not have dense structure — consider K-Means instead.

---

## Section 10 — Behavioral / STAR Questions

**Q30. Describe the biggest technical challenge in this project and how you solved it.**

> The biggest challenge was memory management for a 2.2GB dataset on a 16GB machine. When I first tried `pd.read_csv(full_path)`, it crashed with `pandas.errors.ParserError: C error: out of memory` — the C parser ran out of memory before returning a single row.
>
> I diagnosed this by checking memory usage during loading and realised the C tokenizer allocates a buffer proportional to file size before parsing. The solution: chunked loading with `chunksize=200_000`. I read 200K rows at a time, tracked the date to keep only the most recent 500K records, then concatenated. Additionally, I switched from `format='mixed'` to explicit `format='%m/%d/%Y %I:%M:%S %p'` for date parsing, which cut parse time by 3× and eliminated silent misparsing.

**Q31. How did you ensure your results are reproducible?**

> Three practices: (1) `RANDOM_STATE = 42` set globally in `config.py` and passed to every algorithm that accepts it — K-Means, random subsamplers. (2) All hyperparameters in `config.py` — no hardcoded values inline. Any change is in one place, versioned in git. (3) MLflow logs every run's exact parameters, data version, and git commit hash. If I need to reproduce run #7 from two weeks ago, I check out the commit, read the MLflow params, and re-run — identical output guaranteed.

**Q32. A stakeholder asks: "Your silhouette score is 0.41, not the 0.5 target. Is the project a failure?"**

> No. First, the 0.5 target is a guideline, not a hard threshold. Second, silhouette score depends heavily on data geometry — geographic crime data in Chicago is dense and continuous (crimes happen everywhere, not in cleanly separated blobs), which structurally limits silhouette scores. Third, the clusters have strong semantic validity: the 8 geographic clusters align with Chicago's police districts, which were independently designed by domain experts to group similar neighbourhoods. If the clusters were statistically perfect but geographically nonsensical, that would be a failure. Here we have interpretable, actionable clusters that police can actually use — that's the real success criterion.

**Q33. Why did you choose Streamlit over Flask/FastAPI for the frontend?**

> For this project, Streamlit was the right tool for three reasons: (1) The audience is data analysts and police commanders — not software engineers. Streamlit's declarative model (write Python, get interactive UI) lets me build a polished dashboard in a day rather than a week. (2) There is no prediction API — the app only visualizes pre-computed artifacts. Streamlit's artifact-loading + caching + Folium map support is purpose-built for this. (3) Streamlit Community Cloud provides free deployment with secrets management — zero infrastructure cost for a capstone project. FastAPI would add a serving layer that adds complexity with no benefit here.

---

## Section 11 — Advanced / MNC Questions (Google India, Microsoft India, Amazon)

**Q34. If you had to scale PatrolIQ to all 50 US cities, what changes?**

> The current design assumes Chicago-specific constants: geographic bounds, crime categories, column names, coordinate conventions. Scaling requires: (1) A data ingestion layer that normalises city-specific schemas to a common format. (2) Per-city model training — cluster structure in NYC differs from Chicago. This means separate MLflow experiments per city, separate artifact directories. (3) A parameter store (city → optimal K, eps) instead of hardcoded config.py constants. (4) The Streamlit app would need a city selector that loads the correct artifacts. The two-phase architecture handles this well — you'd just add a city dimension to Phase A without touching Phase B.

**Q35. How would you detect drift in crime patterns over time?**

> Two approaches: (1) Statistical drift detection — compute silhouette score on a new month's data against the existing cluster model. A significant drop (e.g., silhouette falls from 0.41 to 0.25) signals the cluster structure has changed. (2) Distributional drift — use Evidently or a KS test to compare the distribution of each feature (hour, lat_norm, crime_type_encoded) between training data and new data. If feature distributions shift significantly (p-value < 0.05), retrain. A practical trigger: retrain every 6 months using the most recent 500K records.

**Q36. Your PCA reached only 35.9% explained variance in FAST_MODE. What does that tell you about your features?**

> It tells me the 14 engineered features are largely independent — they don't have high linear correlations with each other. Spatial features (lat, lon) are somewhat correlated, temporal cyclical features (hour_sin/cos) are designed to be orthogonal, and severity/categorical features add additional orthogonal dimensions. Low explained variance in 2 PCs doesn't mean PCA failed — it means the data is genuinely high-dimensional and can't be compressed to 2D without information loss. For visualization purposes, t-SNE (which uses non-linear structure) would show clusters more clearly than PCA's 2 linear components.

**Q37. What ethical considerations does a crime data project like this raise?**

> Three main concerns: (1) Predictive policing bias — if historical crime data reflects over-policing of certain neighbourhoods (more patrols → more arrests → more crime records), clustering will flag those same neighbourhoods as hotspots, reinforcing the bias in a feedback loop. (2) Privacy — the dataset contains arrest records linked to location and time. Cluster profiles might inadvertently identify individuals. We aggregate to cluster level to mitigate this, but granular analysis needs privacy review. (3) Causation vs. correlation — clusters show where crimes were reported, not where crimes occurred. Unreported crimes (which vary by neighbourhood) create a systematic blind spot. I'd document these limitations prominently in any deployment to law enforcement.

---

## Section 12 — Quick-Fire Conceptual Questions

**Q38.** What is the curse of dimensionality?
> As dimensions increase, data becomes sparse — the distance between any two points converges, making distance-based algorithms (K-Means, KNN, DBSCAN) unreliable. In clustering, all points look equidistant in high dimensions, so cluster separation disappears. Solution: dimensionality reduction (PCA, t-SNE) before clustering.

**Q39.** What is StandardScaler? Did you use it? Why?
> StandardScaler normalises features to zero mean, unit variance: `(x – μ) / σ`. K-Means is distance-based — a feature with range 0–1000 dominates a feature with range 0–1. I used StandardScaler on FULL_FEATURES before PCA/t-SNE. For geographic clustering (lat/lon only), both features are already on the same scale and range, so StandardScaler was not needed. For DBSCAN on geographic data, scaling would distort the eps interpretation (which I keep in degrees for interpretability).

**Q40.** What is the difference between `fit()`, `transform()`, and `fit_transform()`?
> `fit()`: compute statistics from data (centroids for K-Means, μ/σ for StandardScaler). `transform()`: apply learned statistics to new data — used on test/unseen data. `fit_transform()`: fit and transform in one call — used on training data. NEVER call `fit()` on test data (data leakage). In PatrolIQ's Streamlit pages, we never call `fit()` or `transform()` — we load pre-fitted artifacts from CSVs.

**Q41.** What is Davies-Bouldin index?
> Davies-Bouldin (DB) index measures the ratio of within-cluster scatter to between-cluster separation, averaged over all clusters. Lower is better (0 is perfect). DB = average over all clusters i of: max over j≠i of (σᵢ + σⱼ) / d(cᵢ, cⱼ), where σ is cluster scatter and d is centroid distance. Unlike Silhouette, DB penalises clusters that are both scattered internally AND close to neighbouring clusters.

**Q42.** What is a dendrogram?
> A tree diagram produced by hierarchical clustering that shows the order and height at which clusters are merged. The x-axis shows individual points; the y-axis shows the distance/dissimilarity at which clusters were merged. Cutting the dendrogram horizontally at a given height gives you K clusters. The gap between merge heights (a long vertical line before the next merge) suggests a natural K.

---

## Section 13 — Questions About Your Process and Decisions

**Q43.** Why commit pre-computed artifacts to git instead of regenerating them on demand?
> Three reasons: (1) Determinism — the Streamlit Cloud deploy is always in a known state; no "it worked locally but failed on cloud" scenarios. (2) Speed — Streamlit app loads in <5 seconds instead of 10+ minutes. (3) Cloud constraints — Streamlit free tier has 1GB memory limit; training on cloud would OOM. The trade-off: the repo is larger (artifacts are ~50MB). This is acceptable for a capstone project; for a production system, I'd use a separate artifact store (S3) and version artifacts separately from code.

**Q44.** What would you do differently if you built this again?
> Three things: (1) Set up CI (flake8 + pytest) in Section 1, not Section 10 — the 5 lint errors I found in Section 10 would have been caught immediately. (2) Create a `requirements-app.txt` from day one separate from `requirements.txt`, so the Docker image was always lean. (3) Add a `conftest.py` with mock data fixtures from the beginning — testing became harder later because functions assumed the full 500K CSV existed. With fixtures producing 100-row test DataFrames, all pipeline functions could be unit-tested.

---

## Preparation Tips for India's Big Tech Interviews

**Service companies (HCL/TCS/Wipro/Infosys):**
- Expect a 45-min walkthrough of the project. Have a 2-min intro, 5-min deep dive on one algorithm, and 3-min demo ready.
- They will ask "What is your contribution vs. what tools did?" — be clear that you wrote all the code, made all architectural decisions, debugged all the errors yourself.

**Product companies (Flipkart/Amazon/Zomato):**
- Expect a whiteboard system design question: "Design a real-time crime alert system." Use the two-phase architecture as your answer — pre-compute in batch (Airflow/Spark), serve results via API, push alerts via event stream.
- Expect "what would you do differently?" and "how would you scale this?"

**FAANG-adjacent (Google/Microsoft India):**
- Expect probing on theoretical depth: "Prove why K-Means converges." "Derive the PCA objective function." "What is the time complexity of DBSCAN?"
- Also expect ML system design: feature stores, model versioning, A/B testing for model updates.

**Metrics to memorize for every interview:**
- Geographic K-Means silhouette: **0.41** (K=8)
- DBSCAN noise fraction: **3.8%** (eps=0.008°, min_samples=10)
- Temporal silhouette: **0.26** (K=4)
- PCA variance: **35.9%** in FAST_MODE, **target ≥70%** in production
- t-SNE KL divergence: **1.31**
- Dataset: **500K records** from **7.8M** total, **2001–2025**, **22 columns**, **33 crime types**
- MLflow runs: **16 runs** across **3 experiments**
