---
paths:
  - "pages/**/*.py"
  - "streamlit_app.py"
---

# Streamlit Page Rules — PatrolIQ

## The Golden Rule
Pages NEVER train models, run clustering, or do heavy computation.
They ONLY load pre-computed artifacts and render visualizations.
If you see model.fit() or heavy pandas operations in a page — it's wrong.

## Caching Rules (Non-Negotiable)
- @st.cache_data(ttl=3600) on EVERY pd.read_csv() call
- @st.cache_data on EVERY json.load() call for artifacts
- @st.cache_resource on joblib.load() / pickle.load() calls
- Cache key includes the file path so different files get different caches

## Artifact Loading Pattern
```python
from pathlib import Path
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

@st.cache_data
def load_geo_labels(algorithm: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / "geographic" / f"{algorithm}_labels.csv"
    if not path.exists():
        st.error(f"Run scripts/run_full_pipeline.py first to generate artifacts.")
        st.stop()
    return pd.read_csv(path)
```

## Error Handling (Mandatory)
- Wrap artifact loading in try/except FileNotFoundError
- Show st.error() with actionable message when file missing
- Use st.stop() to halt page execution on critical errors
- Use st.warning() for non-critical issues (empty filtered data, etc.)

## Streamlit Version Target
streamlit==1.37.0 — use st.navigation() for multi-page (not deprecated st.sidebar.radio)

## Performance Rules
- Max 50,000 points on Folium map (subsample if df has more)
- Use st.spinner() around any operation > 1 second
- Sidebar filters should re-use cached data (filter in memory, not reload)

## Layout Standards
- Page config: st.set_page_config(layout="wide")
- Use st.columns() for side-by-side charts
- KPI metrics: st.metric() with delta for comparisons
- Tables: st.dataframe() with use_container_width=True

## Forbidden
- model.fit(), model.transform() — absolute ban
- print() — use st.write() or logging
- Blocking I/O without st.spinner()
- Hardcoded absolute paths
