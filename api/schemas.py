# filename: api/schemas.py
# purpose:  Pydantic request/response models for PatrolIQ FastAPI

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Geographic prediction ─────────────────────────────────────

class GeoRequest(BaseModel):
    lat: float = Field(..., ge=41.6, le=42.0, description="Latitude (Chicago: 41.6–42.0)")
    lon: float = Field(..., ge=-87.9, le=-87.5, description="Longitude (Chicago: -87.9 to -87.5)")


class GeoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    cluster_id: int
    cluster_label: str
    crime_risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    dominant_crime_type: str
    avg_severity_score: Optional[float] = None
    arrest_rate: Optional[float] = None
    model_name: str
    model_version: str
    prediction_timestamp: str


# ── Temporal prediction ───────────────────────────────────────

class TemporalRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, description="Month (1–12)")
    is_weekend: Optional[bool] = Field(None, description="Override weekend flag (inferred if omitted)")


TEMPORAL_CLUSTER_LABELS = {
    0: "Late Night / Early Morning",
    1: "Morning Commute",
    2: "Midday Activity",
    3: "Evening Peak",
}


class TemporalResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    cluster_id: int
    cluster_label: str
    pattern_description: str
    model_name: str
    model_version: str
    prediction_timestamp: str


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    geo_model_loaded: bool
    temporal_model_loaded: bool
    api_version: str
    timestamp: str


# ── Drift ────────────────────────────────────────────────────

class DriftFeatureResult(BaseModel):
    feature: str
    statistic: float
    p_value: float
    drift_detected: bool
    method: str


class DriftReportResponse(BaseModel):
    overall_drift_detected: bool
    features_checked: int
    features_drifted: int
    results: list[DriftFeatureResult]
    report_timestamp: str
