# filename: api/schemas.py
# purpose:  Pydantic request/response models for PatrolIQ FastAPI

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator


# ── Geographic prediction ─────────────────────────────────────

class GeoRequest(BaseModel):
    lat: float = Field(..., example=41.85, description="Latitude (Chicago: 41.6–42.0)")
    lon: float = Field(..., example=-87.65, description="Longitude (Chicago: -87.9 to -87.5)")

    @validator("lat")
    def validate_lat(cls, v):
        if not (41.6 <= v <= 42.0):
            raise ValueError("Latitude outside Chicago bounds (41.6–42.0)")
        return v

    @validator("lon")
    def validate_lon(cls, v):
        if not (-87.9 <= v <= -87.5):
            raise ValueError("Longitude outside Chicago bounds (-87.9 to -87.5)")
        return v


class GeoResponse(BaseModel):
    cluster_id: int
    cluster_label: str
    crime_risk_level: str           # HIGH / MEDIUM / LOW
    dominant_crime_type: str
    model_name: str
    model_version: str
    prediction_timestamp: str


# ── Temporal prediction ───────────────────────────────────────

class TemporalRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, example=22, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, example=5, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, example=7, description="Month (1–12)")
    is_weekend: Optional[bool] = Field(None, description="Override weekend flag (inferred if omitted)")


TEMPORAL_CLUSTER_LABELS = {
    0: "Late Night / Early Morning",
    1: "Morning Commute",
    2: "Midday Activity",
    3: "Evening Peak",
}


class TemporalResponse(BaseModel):
    cluster_id: int
    cluster_label: str
    pattern_description: str
    model_name: str
    model_version: str
    prediction_timestamp: str


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    geo_model_loaded: bool
    temporal_model_loaded: bool
    api_version: str


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
