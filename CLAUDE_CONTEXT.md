# Claude Context — PatrolIQ
*Paste this at the start of every new Claude conversation.*

---

## Who I Am

I am a data science student (GUVI capstone program) building production-grade ML systems.
I am comfortable with Python, pandas, sklearn, and basic Docker. I am learning MLOps.
I work on **Windows 11** with **Git Bash** and **PowerShell**. Python version: **3.11**.
I deploy to **Streamlit Community Cloud** (the app IS the product — no separate backend).

---

## How We Work Together (Non-Negotiable)

1. **Plan before coding.** State your plan in clear steps, call out assumptions, flag risks. Wait for me to say "go" or "yes" before writing any code.
2. **Ask for reference files before each section.** Before writing code for any new section, say: "Ready to start Section N. Please share the equivalent file(s) from your EMI project as reference." Then WAIT.
3. **If there are multiple valid approaches, present them with tradeoffs.** Never silently pick one.
4. **Never abbreviate files.** Write the complete file every time. No `# ... rest of file unchanged`.
5. **Verify a file exists before referencing it.** Read it first. Do not assume its contents from memory.
6. **Never invent library APIs.** If unsure whether a function/attribute exists in a given version, say so. Do not guess.
7. **Use Bash tool for all terminal commands** (not PowerShell tool).

---

## My Tech Stack (PatrolIQ — Unsupervised ML + Streamlit)

| Layer | Library / Tool | Version |
|---|---|---|
| Language | Python | 3.11 |
| Data | pandas, numpy, scipy | 2.1.4 / 1.26.4 / 1.11.4 |
| ML | scikit-learn | 1.5.2 |
| Experiment tracking | MLflow | 2.14.1 |
| Maps | Folium + streamlit-folium | 0.17.0 / 0.22.0 |
| Charts | Plotly | 5.22.0 |
| UI | Streamlit | 1.37.0 |
| Containers | Docker Compose | local stack |
| Testing | pytest + pytest-cov | 8.3.2 / 5.0.0 |

**NOT in this project (unlike EMI):** FastAPI, Redis, Airflow, Prometheus, Render.
**Never suggest alternatives to these unless I ask.**

---

## Deployment Setup

### Streamlit Community Cloud (Only Deployment Target)
- Branch: `master`
- **CRITICAL:** Streamlit Cloud reads `requirements.txt` by default — NOT `streamlit-requirements.txt`
  - Either: configure the app settings to explicitly point to `streamlit-requirements.txt`
  - Or: ensure `requirements.txt` is Streamlit Cloud compatible
  - **Never assume** Streamlit Cloud reads non-standard filenames without explicit configuration
- Secrets: stored in Streamlit secrets manager
- Pre-computed artifacts committed to repo → zero training on cloud (artifacts load instantly)

### Branch Strategy (Locked)
```
master  → clean source + committed artifacts → Streamlit Cloud
```
- All fixes go to `master`
- No separate deploy branch needed (no binary model PKLs to force-add for this project)
- `data/raw/` is gitignored; `data/processed/chicago_crime_500k.csv.gz` IS committed
- `artifacts/` directory IS committed (pre-computed clustering results)

---

## Rules Claude Must Always Follow

### Architecture Rule (Most Important)
**Two-phase pipeline. Streamlit NEVER trains models.**
- Phase A (Local): preprocess → feature engineer → cluster → export artifacts → MLflow
- Phase B (Cloud): Streamlit loads CSV/JSON artifacts → renders visualizations
- Any `model.fit()` call in `pages/*.py` is a CRITICAL bug. Report it immediately.

### FAST_MODE Rule
```python
FAST_MODE = True   # dev: smaller subsamples, skip t-SNE, K range 2-5 only
FAST_MODE = False  # production: full 500K pipeline, complete elbow sweep, full t-SNE
```
- Define `FAST_MODE` at the top of every training file before any other code
- Develop with `FAST_MODE = True`
- Flip to `False` only for final `run_full_pipeline.py` to generate production artifacts

### Clustering-Specific Rules

**Hierarchical Clustering:**
```
NEVER fit AgglomerativeClustering or scipy.cluster.hierarchy.linkage on full 500K records.
Ward linkage matrix = O(n²) memory. 500K records → ~200GB RAM. Will crash/OOM.
ALWAYS: fit on HIERARCHICAL_SUBSAMPLE = 10,000 stratified sample.
Assign full dataset: find nearest cluster centroid for each record (KNN).
Comment in code: "# Ward linkage on 500K = ~200GB RAM — must subsample to 10K"
```

**t-SNE:**
```
NEVER run t-SNE directly on 500K records. t-SNE is O(n log n) to O(n²).
ALWAYS: PCA → 50 components first → t-SNE on TSNE_SUBSAMPLE = 50,000.
Comment in code: "# t-SNE is O(n²) — PCA pre-reduction + 50K subsample is standard practice"
Use stratified sample to preserve crime type distribution.
```

**DBSCAN eps units:**
```
eps is in DECIMAL DEGREES, not kilometres.
0.008 degrees ≈ 656m at Chicago's latitude (42°N).
Conversion: km = degrees × 111.320 × cos(lat_in_radians)
At 42°N: 0.008° × 111320 × cos(0.733) = ~0.008 × 82765 = ~662m
ALWAYS document this conversion in the code comment near eps assignment.
```

### JSON Artifact Export Rule (Critical)
```python
# sklearn metrics return numpy types (np.float64, np.int32, etc.)
# json.dumps() CRASHES on numpy types: TypeError: Object of type float64 is not JSON serializable
# ALWAYS use NumpyEncoder for any json.dump() call that includes model output

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        return super().default(obj)

# Usage: json.dump(metrics, f, indent=2, cls=NumpyEncoder)
# Lesson from EMI CQ-03: numpy scalars in json.dumps → TypeError. Always use encoder.
```

### Folium Map Performance Rule
```
Folium maps with >50K CircleMarkers crash browser tabs or render for minutes.
ALWAYS subsample to ≤50,000 points before passing to Folium.
In pages/2_Geographic_Hotspots.py: sample df to 50K before building the map.
The full labels CSV is still loaded — just subsample for visualization only.
```

### Streamlit Caching Rule
```python
@st.cache_data(ttl=3600)    # on ALL pd.read_csv() and json.load() calls in pages/
@st.cache_resource          # on joblib.load() / pickle.load() calls
# Without caching, every interaction reloads CSVs from disk → sluggish app
```

### Code Rules

- **Never hardcode paths.** Always: `from config import ARTIFACTS_DIR` or use `pathlib.Path(__file__).parent.parent / "artifacts"`
- **No print() in src/.** Use Python `logging` throughout. `print()` is allowed only in notebooks.
- **Always wrap file loading in try/except FileNotFoundError** in Streamlit pages:
  ```python
  try:
      df = pd.read_csv(path)
  except FileNotFoundError:
      st.error("Run `python scripts/run_full_pipeline.py` first to generate artifacts.")
      st.stop()
  ```
- **Use pages/ folder routing — NOT st.navigation().** Lesson from EMI DEP-02: st.navigation() required Streamlit ≥1.36.0, caused crash loop when version was wrong. pages/ works with ALL versions.
- **Metric storage:** 6 decimal places in all JSON artifacts. 4dp for display.

### Docker / CI Rules (Section 10)

- **Single-stage Dockerfile for pure Python.** Multi-stage builds are for compiled languages (Go, Rust). For Python/Streamlit, single-stage is correct — zero benefit, less complexity.
- **Bake pre-computed artifacts into the image.** `artifacts/` is static output from the pipeline. Bake it in → image is self-contained, works on any machine without local artifacts. Only `mlruns/` gets a volume mount (it holds the MLflow SQLite DB, too large/dynamic to bake).
- **Use `streamlit-requirements.txt` for Docker** (not the full `requirements.txt`). Full requirements include training deps (scipy, matplotlib, seaborn) not needed at serve time. Separate file = faster build, smaller image.
- **Always create `.dockerignore` before `docker build`.** Without it, the build context includes `data/raw/` (2.2GB CSV), `venv/` (~500MB), `mlruns/` — making build slow or failing. Required entries: `data/`, `venv/`, `mlruns/`, `notebooks/`, `models/`, `.git/`.
- **CI should NEVER run the training pipeline.** `chicago_crime_500k.csv.gz` is gitignored — CI has no data. CI validates code quality (flake8), unit tests (pytest), and image builds cleanly (docker build). That's all.
- **Drop mypy for GUVI capstone projects.** mypy generates 50+ false positives on pandas/MLflow/Streamlit APIs (incomplete stubs). It blocks CI without catching real bugs. Use flake8 only.
- **flake8 ignore list for this project:** `E221,E202,E302,E305,E402,W503,W504,F401` — these are all stylistic (aligned assignments, blank lines, unused imports in training files). Real bugs to catch: `F541` (empty f-strings), `E225` (missing whitespace around operator), `W292` (no newline at end).

### Git Rules

- **Anchor .gitignore patterns with leading slash** for root-level directories:
  ```
  /data/raw/    ← CORRECT (only excludes root-level data/raw/)
  data/raw/     ← WRONG (also excludes src/data/raw/ if it existed)
  ```
  Lesson from EMI DEP-01: unanchored `data/` excluded `src/data/`, causing ModuleNotFoundError.
- **Before every push:** `git status` and verify no source files are accidentally ignored.
- **Never git push --force on master.**

### Windows-Specific Reminders
- Use `$env:PYTHONPATH="."` not `export PYTHONPATH=.`
- Use Bash tool (not PowerShell tool) for terminal commands
- Git LF→CRLF warnings are cosmetic — ignore them

---

## Mistakes From Previous Project (Do Not Repeat)

| # | Mistake | What Happened | Rule |
|---|---|---|---|
| 1 | `st.navigation()` | Required ≥1.36.0, pinned 1.35.0, crash loop on Streamlit Cloud | Use `pages/` routing — works with ALL versions |
| 2 | `data/` in gitignore | Excluded `src/data/` too. ModuleNotFoundError on cloud | Use `/data/` (anchored with leading slash) |
| 3 | `json.dumps()` on sklearn output | `TypeError: float64 is not JSON serializable` | Always use `NumpyEncoder` |
| 4 | t-SNE on full dataset | Would run for hours or OOM | PCA first → t-SNE on 50K subsample only |
| 5 | Hierarchical on full 500K | ~200GB RAM for linkage matrix | Always fit on 10K subsample |
| 6 | Folium with 500K points | Browser tab crashes or hangs | Subsample to ≤50K for map display |
| 7 | Model.fit() in Streamlit page | Trains on every interaction | Pages load artifacts only — ban fit() |
| 8 | DBSCAN eps without unit docs | Misinterpreted as km when it's degrees | Always document: 0.008° ≈ 662m at 42°N |
| 9 | Streamlit Cloud reads requirements.txt | streamlit-requirements.txt ignored | Configure app settings to specify file path |
| 10 | No FAST_MODE upfront | Full pipeline ran during dev, wasted time | Define FAST_MODE=True at top of every training file |
| 11 | Multi-stage Dockerfile for Python | Added complexity with zero benefit (Python doesn't compile) | Use single-stage for pure-Python projects |
| 12 | Missing .dockerignore | 2.2GB raw CSV entered build context, build hung | Always create .dockerignore before docker build |
| 13 | mypy in CI for data science | 50+ false positives on pandas/MLflow stubs, blocked CI | Use flake8 only; add E221/F401 to ignore list |
| 14 | Volume mounting artifacts/ | Empty mount on fresh machine shadows baked artifacts | Bake artifacts/ into image; only mount mlruns/ |

---

## PatrolIQ Specific: Key Numbers to Know

| Item | Value |
|---|---|
| Full dataset | 7.8M records (2001–2025), 22 columns, 1.7GB |
| Sample used | 500,000 most-recent records |
| Processed file | chicago_crime_500k.csv.gz (~25–35MB) |
| Chicago bounds | Lat 41.6–42.0, Lon -87.9 to -87.5 |
| Crime categories | 33 distinct Primary_Type values |
| Geographic K | 8 (tuned; from elbow method) |
| Geographic silhouette | 0.41 (K-Means) |
| DBSCAN noise fraction | 3.8% (< 10% target PASS) |
| Temporal K | 4 (silhouette 0.26) |
| DBSCAN eps | 0.008 degrees ≈ 662m at 42°N |
| Hierarchical subsample | 10,000 records |
| t-SNE subsample | 50,000 records |
| Folium map limit | 50,000 points |
| PCA variance (FAST_MODE) | 35.9% (2 components — FAST_MODE only) |
| PCA variance target | ≥70% in 2–3 components (production) |
| t-SNE KL divergence | 1.31 (lower = better separation) |
| MLflow runs logged | 16 runs across 3 experiments |
| Silhouette target | >0.5 (PDF requirement) |
| Docker base image | python:3.11-slim (~150MB) |
| Unit tests | 12 (NumpyEncoder + save_json) |

---

## My Environment

```
OS:           Windows 11
Shell:        PowerShell (primary) + Git Bash
Python:       3.11 (via venv)
PYTHONPATH:   $env:PYTHONPATH="."  ← PowerShell syntax
IDE:          VS Code
Git user:     configured
Claude tool:  use Bash (not PowerShell) for terminal commands
```

---

## Section Completion Checklist (Use at End of Every Section)

- [ ] All files saved
- [ ] Tests passed (if applicable)
- [ ] Artifacts validated (expected CSVs and JSONs present in artifacts/)
- [ ] `git add` all changed files
- [ ] `git commit -m "section-X: description"`
- [ ] `git log --oneline` — confirm commit appears
- [ ] `git status` — confirm working tree clean
- [ ] CLAUDE.md progress table updated
- [ ] Next section dependencies confirmed

---

## MANDATORY: After Every Section Completed

After completing each major section, output a block like this:

---
### 📋 LESSONS UPDATE — [Section Name]
**Date:** [today]
**Section:** [what we just built]

**What worked:**
- [thing 1]

**Mistakes made:**
- MISTAKE: [what went wrong]
  CAUSE: [why]
  FIX: [what we did]
  PREVENTION: [rule for next time]

**New rules to add to CLAUDE_CONTEXT:**
- [any new PatrolIQ-specific rule discovered]

**Time taken:** [estimated]
**Difficulty:** Easy / Medium / Hard
---

---
### 📋 LESSONS UPDATE — Section 10: Docker + CI
**Date:** 2026-05-14
**Section:** Dockerfile, .dockerignore, docker-compose.yml, .github/workflows/ci.yml, tests/test_helpers.py

**What worked:**
- Single-stage Dockerfile is correct for pure Python — no benefit to multi-stage, cleaner to reason about
- Baking `artifacts/` into the image makes it fully self-contained (works on any machine without local pipeline output)
- `streamlit-requirements.txt` already existed with the right minimal deps — reused directly in Dockerfile
- 12 unit tests for `NumpyEncoder` + `save_json` — comprehensive coverage, all pass in 0.72s
- Extending flake8 ignore list (E221/E202/E302/E305/F401) instead of refactoring 15 files — right call for capstone scope

**Mistakes made:**
- MISTAKE: Planned mypy in CI
  CAUSE: Assumed type checking would catch bugs
  FIX: Dropped mypy from CI plan before implementing
  PREVENTION: For data science projects, mypy generates 50+ false positives on pandas/MLflow/Streamlit stubs. Use flake8 only.

- MISTAKE: 5 pre-existing lint errors found in source files (F541 empty f-strings, E225 missing whitespace, W292 no newline)
  CAUSE: These accumulated across Sections 2–9 without CI to catch them
  FIX: Fixed inline before finalising CI so it starts clean
  PREVENTION: CI should have been set up in Section 1 so every section commit was linted

**New rules added to CLAUDE_CONTEXT:**
- See Docker/CI Rules section above (7 rules)
- Mistakes table extended with rows 11–14

**Time taken:** ~45 min
**Difficulty:** Easy (architecture decisions were the only complexity)
---

---
### 📋 LESSONS UPDATE — Post-Project: Hook + Interview Prep
**Date:** 2026-05-14
**Section:** .claude/settings.json PostToolUse hook, INTERVIEW_PREP.md, CLAUDE_CONTEXT.md updates

**What worked:**
- PostToolUse hook with `"if": "Bash(git commit*)"` fires precisely on section commits — no noise from other bash calls
- `additionalContext` in `hookSpecificOutput` injects the reminder directly into Claude's context without blocking the commit
- Pipe-testing the command with `echo '{}' | bash -c "echo '...'"` before writing to settings — caught shell-escaping issues early
- Lessons stored in TWO places (memory/feedback_lessons_learned.md + CLAUDE_CONTEXT.md) means they survive across both tool-based sessions and fresh-context pastes

**Mistakes made:**
- MISTAKE: CLAUDE_CONTEXT.md "MANDATORY" section was a blank template for all 10 sections — never got filled
  CAUSE: No automated trigger existed; relied on Claude remembering to do it after each commit
  FIX: Added PostToolUse git-commit hook + reinforced in CLAUDE.md mandatory block
  PREVENTION: Set up the hook in Section 1 on the next project, not Section 10

**New rules to add to CLAUDE_CONTEXT:**
- For automation hooks: always use `"if": "Bash(git commit*)"` filter to avoid firing on every bash call. Use `additionalContext` not `systemMessage` — systemMessage shows in UI but doesn't enter Claude's reasoning context.
- Two-layer enforcement: hook injects the reminder AND CLAUDE.md describes what to do when the reminder arrives. One layer alone is fragile.

**Time taken:** ~20 min
**Difficulty:** Easy
---
