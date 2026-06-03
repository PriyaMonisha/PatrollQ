# filename: src/features/engineer.py
# purpose:  Feature engineering pipeline for PatrolIQ crime clustering (Section 4)
# version:  1.0

import logging

import numpy as np
import pandas as pd

from config import SEVERITY_DEFAULT, SEVERITY_SCORES

logger = logging.getLogger(__name__)

# ── Canonical list of features this class ADDS ───────────────
# Used by clustering notebooks and tests to identify engineered columns.
# Does NOT include Is_Weekend, primary_type_code, location_desc_code,
# arrest, domestic — those are already added by preprocessor.py.
NEW_FEATURES = [
    # Group 1: Cyclical temporal encoding
    # WHY: Hour 23 and Hour 0 are 1 step apart — raw integers make them 23
    #      apart. Clustering uses Euclidean distance; sin/cos preserves the
    #      circular nature so midnight continuity is respected.
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "month_sin", "month_cos",
    # Group 2: Geographic normalization
    # WHY: KMeans and DBSCAN use Euclidean distance. Raw lat/lon degrees are
    #      on a different scale than temporal features in FULL_FEATURES.
    #      Normalizing to [0,1] prevents coordinates from dominating the space.
    #      Bounds are learned from training data (fit step).
    "lat_norm", "lon_norm",
    # Group 3: Crime characteristics
    "Crime_Severity_Score",  # int8, 1–10 ordinal from SEVERITY_SCORES lookup
    # Season_Code removed: month_sin/month_cos already capture seasonality
    # in a continuous, distance-metric-safe way. Ordinal 0-3 is redundant.
]


class CrimeFeatureEngineer:
    """
    Sklearn-compatible feature engineering pipeline for PatrolIQ.

    fit():          Learn lat/lon normalization bounds from training data.
    transform():    Append 10 new features (see NEW_FEATURES). Originals kept.
    fit_transform(): Convenience wrapper — fit then transform.

    Input:  Cleaned DataFrame from preprocessor.preprocess_data()
            (requires columns: Hour, Day_of_Week, Month, Season,
             latitude, longitude, primary_type)
    Output: Same DataFrame with NEW_FEATURES columns appended.

    Usage in training (notebooks/03_feature_engineering.py):
        fe = CrimeFeatureEngineer()
        df_features = fe.fit_transform(df_preprocessed)   # learns from full data
        joblib.dump(fe, FEATURE_ENGINEER_PATH)

    Usage in clustering notebooks (already fitted):
        fe = joblib.load(FEATURE_ENGINEER_PATH)
        df_features = fe.transform(df_preprocessed)
    """

    def __init__(self) -> None:
        self.is_fitted_: bool = False
        self._lat_min: float  = 0.0
        self._lat_max: float  = 1.0
        self._lon_min: float  = 0.0
        self._lon_max: float  = 1.0

    # ── Fit ──────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame) -> "CrimeFeatureEngineer":
        """
        Learn lat/lon normalization bounds from X.

        WHY learn from data (not config.py bounds)?
        The sampled 500K records may not span the full theoretical Chicago
        bounds (41.6-42.0, -87.9 to -87.5). Learning from data produces
        tighter, more precise normalization for the actual sample.
        clip(0, 1) in transform() handles any out-of-sample points.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CrimeFeatureEngineer.fit() requires a pandas DataFrame.")

        required = ["latitude", "longitude", "Hour", "Day_of_Week", "Month",
                    "Season", "primary_type"]
        missing  = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for fit(): {missing}\n"
                f"Run preprocessor.preprocess_data() first."
            )

        self._lat_min = float(X["latitude"].min())
        self._lat_max = float(X["latitude"].max())
        self._lon_min = float(X["longitude"].min())
        self._lon_max = float(X["longitude"].max())

        lat_range = self._lat_max - self._lat_min
        lon_range = self._lon_max - self._lon_min

        if lat_range == 0 or lon_range == 0:
            raise ValueError(
                "Latitude or Longitude has zero range — check coordinate data."
            )

        self.is_fitted_ = True
        logger.info(
            "CrimeFeatureEngineer fitted ✓ | "
            f"lat [{self._lat_min:.4f}, {self._lat_max:.4f}] | "
            f"lon [{self._lon_min:.4f}, {self._lon_max:.4f}]"
        )
        return self

    # ── Transform ────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Append 10 engineered features to X. Original columns are preserved.

        Args:
            X: DataFrame with preprocessed columns (from preprocessor.py)

        Returns:
            Copy of X with NEW_FEATURES appended.
        """
        if not self.is_fitted_:
            raise ValueError(
                "Call fit() or fit_transform() before transform()."
            )
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CrimeFeatureEngineer.transform() requires a pandas DataFrame.")

        df = X.copy()

        # ── Group 1: Cyclical temporal encoding ──────────────
        # Encodes circular time variables so distance metrics work correctly.
        # Example: Hour 0 (midnight) and Hour 23 (11pm) should be close.
        #   raw integer distance: |23 - 0| = 23  ← WRONG
        #   sin/cos distance:     ≈ 0.13          ← CORRECT

        df["hour_sin"]  = np.sin(2 * np.pi * df["Hour"]        / 24).astype("float32")
        df["hour_cos"]  = np.cos(2 * np.pi * df["Hour"]        / 24).astype("float32")
        df["day_sin"]   = np.sin(2 * np.pi * df["Day_of_Week"] / 7 ).astype("float32")
        df["day_cos"]   = np.cos(2 * np.pi * df["Day_of_Week"] / 7 ).astype("float32")
        df["month_sin"] = np.sin(2 * np.pi * df["Month"]       / 12).astype("float32")
        df["month_cos"] = np.cos(2 * np.pi * df["Month"]       / 12).astype("float32")

        # ── Group 2: Geographic normalization ────────────────
        # Normalize lat/lon to [0, 1] using bounds learned in fit().
        # clip(0, 1) handles any out-of-sample coordinates without errors.

        lat_range = self._lat_max - self._lat_min
        lon_range = self._lon_max - self._lon_min

        df["lat_norm"] = (
            (df["latitude"]  - self._lat_min) / lat_range
        ).clip(0, 1).astype("float32")

        df["lon_norm"] = (
            (df["longitude"] - self._lon_min) / lon_range
        ).clip(0, 1).astype("float32")

        # ── Group 3: Crime characteristics ───────────────────
        # Severity score: ordinal 1–10 from domain knowledge lookup.
        # Crime types not in SEVERITY_SCORES get SEVERITY_DEFAULT (=2).
        df["Crime_Severity_Score"] = (
            df["primary_type"]
            .map(SEVERITY_SCORES)
            .fillna(SEVERITY_DEFAULT)
            .astype("int8")
        )

        n_in  = X.shape[1]
        n_out = df.shape[1]
        logger.info(
            f"CrimeFeatureEngineer transform complete ✓ | "
            f"{df.shape[0]:,} rows | {n_in} → {n_out} features "
            f"(+{n_out - n_in} new)"
        )
        return df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on X, then transform X. Equivalent to fit(X).transform(X)."""
        return self.fit(X).transform(X)

    @property
    def new_feature_names(self) -> list[str]:
        """Return the list of feature names added by this engineer."""
        return NEW_FEATURES.copy()

    @property
    def normalization_params(self) -> dict[str, float]:
        """Return the fitted lat/lon normalization bounds."""
        if not self.is_fitted_:
            raise ValueError("CrimeFeatureEngineer has not been fitted yet.")
        return {
            "lat_min": self._lat_min,
            "lat_max": self._lat_max,
            "lon_min": self._lon_min,
            "lon_max": self._lon_max,
        }
