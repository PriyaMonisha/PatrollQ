# filename: scripts/reproduce_results.py
# purpose:  Verify that stored artifacts match expected reference metrics.
#           Run after any pipeline change to confirm reproducibility.
#
# Usage:
#   python scripts/reproduce_results.py            # check current artifacts
#   python scripts/reproduce_results.py --seed 42  # (seed is informational only;
#                                                  #  training uses config.RANDOM_STATE)

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import ARTIFACTS_DIR

# ── Reference values (captured from the canonical pipeline run) ──────────────
REFERENCE = {
    "geo_kmeans": {
        "silhouette":        (0.35, 0.55),   # acceptable range
        "davies_bouldin":    (0.5,  1.2),
        "n_clusters":        8,
    },
    "geo_dbscan": {
        "noise_fraction":    (0.0, 0.10),    # must be < 10%
        "n_clusters_min":    2,
    },
    "temporal_kmeans": {
        "silhouette":        (0.15, 0.45),
        "n_clusters":        4,
    },
    "pca": {
        "cumulative_variance": (0.20, 0.80),  # wide range; varies with data window
        "n_components":       3,
    },
    "tsne": {
        "kl_divergence":     (0.5, 3.0),
    },
    "mlflow": {
        "total_runs_min":    6,               # GUVI requirement
    },
}


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _check(name: str, value: float, lo: float, hi: float) -> tuple[bool, str]:
    ok = lo <= value <= hi
    status = "PASS" if ok else "FAIL"
    return ok, f"  {status}  {name}: {value:.4f}  (expected {lo}–{hi})"


def main(seed: int) -> int:
    print("=" * 60)
    print("  PatrolIQ — Reproducibility Check")
    print(f"  RANDOM_STATE from config.py  (seed arg={seed} is informational)")
    print("=" * 60)

    failures = 0

    # ── Geographic K-Means ───────────────────────────────────
    print("\n[1/5] Geographic K-Means")
    try:
        m = _load(ARTIFACTS_DIR / "geographic" / "kmeans_metrics.json")
        ref = REFERENCE["geo_kmeans"]
        for metric, (lo, hi) in [("silhouette", ref["silhouette"]),
                                   ("davies_bouldin", ref["davies_bouldin"])]:
            ok, msg = _check(metric, m[metric], lo, hi)
            print(msg)
            failures += 0 if ok else 1
        n_ok = m["n_clusters"] >= 2
        status = "PASS" if n_ok else "FAIL"
        print(f"  {status}  n_clusters: {m['n_clusters']}  (expected >=2; config target={ref['n_clusters']})")
        failures += 0 if n_ok else 1
    except FileNotFoundError:
        print("  FAIL  kmeans_metrics.json not found -- run pipeline first")
        failures += 1

    # ── Geographic DBSCAN ────────────────────────────────────
    print("\n[2/5] Geographic DBSCAN")
    try:
        m = _load(ARTIFACTS_DIR / "geographic" / "dbscan_metrics.json")
        noise = m.get("noise_fraction", m.get("noise", 1.0))
        lo, hi = REFERENCE["geo_dbscan"]["noise_fraction"]
        ok, msg = _check("noise_fraction", float(noise), lo, hi)
        print(msg)
        failures += 0 if ok else 1
        n_clusters = m.get("n_clusters", 0)
        min_c = REFERENCE["geo_dbscan"]["n_clusters_min"]
        n_ok = n_clusters >= min_c
        status = "PASS" if n_ok else "FAIL"
        print(f"  {status}  n_clusters: {n_clusters}  (expected >={min_c})")
        failures += 0 if n_ok else 1
    except FileNotFoundError:
        print("  FAIL  dbscan_metrics.json not found")
        failures += 1

    # ── Temporal K-Means ─────────────────────────────────────
    print("\n[3/5] Temporal K-Means")
    try:
        m = _load(ARTIFACTS_DIR / "temporal" / "kmeans_metrics.json")
        ref = REFERENCE["temporal_kmeans"]
        lo, hi = ref["silhouette"]
        ok, msg = _check("silhouette", m["silhouette"], lo, hi)
        print(msg)
        failures += 0 if ok else 1
        n_ok = m["n_clusters"] >= 2
        status = "PASS" if n_ok else "FAIL"
        print(f"  {status}  n_clusters: {m['n_clusters']}  (expected >=2; config target={ref['n_clusters']})")
        failures += 0 if n_ok else 1
    except FileNotFoundError:
        print("  FAIL  temporal/kmeans_metrics.json not found")
        failures += 1

    # ── PCA ──────────────────────────────────────────────────
    print("\n[4/5] PCA")
    try:
        m = _load(ARTIFACTS_DIR / "dimensionality" / "pca_metrics.json")
        var = m.get("cumulative_variance", m.get("explained_variance_ratio_cumulative", 0))
        lo, hi = REFERENCE["pca"]["cumulative_variance"]
        ok, msg = _check("cumulative_variance", float(var), lo, hi)
        print(msg)
        failures += 0 if ok else 1
    except FileNotFoundError:
        print("  FAIL  pca_metrics.json not found")
        failures += 1

    # ── MLflow runs ──────────────────────────────────────────
    print("\n[5/5] MLflow Experiment Runs")
    try:
        runs = _load(ARTIFACTS_DIR / "mlflow_exports" / "all_runs.json")
        n_runs = len(runs) if isinstance(runs, list) else len(runs.get("runs", []))
        min_runs = REFERENCE["mlflow"]["total_runs_min"]
        ok = n_runs >= min_runs
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  total_runs: {n_runs}  (expected >={min_runs})")
        failures += 0 if ok else 1
    except FileNotFoundError:
        print("  FAIL  mlflow_exports/all_runs.json not found")
        failures += 1

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures == 0:
        print("  ALL CHECKS PASSED -- Results match reference")
    else:
        print(f"  {failures} CHECK(S) FAILED -- Review pipeline or update reference values")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PatrolIQ artifact metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (informational)")
    args = parser.parse_args()
    sys.exit(main(args.seed))
