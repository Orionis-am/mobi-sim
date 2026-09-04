"""Pydantic v2 request/response schemas for every module_e router (`docs/SUJET.md` MOD-E `models.py`).

Grouped by router in this single file, matching the file breakdown's literal naming (`models.py`,
not `schemas.py`, and no separate per-router schema files).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- auth ---


class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8)
    scope: Literal["admin", "operator", "user"] = "user"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    scope: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# --- catalog ---


class ServiceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: str
    protocol: str
    codec: str | None
    min_mos: float
    min_bitrate_kbps: float
    max_delay_ms: float
    tariff: str


# --- codecs ---


class CodecEvaluateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    codecs: list[Literal["aac", "gsm", "opus"]] = Field(default_factory=lambda: ["aac", "gsm", "opus"])
    sample_rate: int = 8_000
    use_real_whisper: bool = False
    seed: int | None = None


class CodecMetrics(BaseModel):
    snr_db: float
    pesq_nb: float
    log_spectral_distortion: float
    mushra_mean: float
    wer: float


class CodecEvaluateResponse(BaseModel):
    reference_text: str
    sample_rate: int
    results: dict[str, CodecMetrics]


# --- sms ---


class SmsSendRequest(BaseModel):
    to: str
    body: str = Field(min_length=1, max_length=1600)


class SmsSendResponse(BaseModel):
    sid: str
    status: str
    to: str
    from_: str


class SmsStatusResponse(BaseModel):
    sid: str
    status: str
    error_code: int | None
    date_updated: datetime | None


# --- location ---


class LocationEstimateRequest(BaseModel):
    method: Literal["cell_id", "toa", "wifi", "ip"]
    lat: float | None = None
    lon: float | None = None
    ip: str | None = None
    k: int | None = None  # TOA anchor count
    noise_std_db: float | None = None  # Wi-Fi RSSI noise
    seed: int | None = None


class LocationEstimateResponse(BaseModel):
    method: str
    lat: float | None = None
    lon: float | None = None
    error_m: float | None = None
    ip: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None


class PoiOut(BaseModel):
    name: str
    category: str
    lat: float
    lon: float
    distance_m: float


# --- qos ---


class QosPredictRequest(BaseModel):
    codec: Literal["aac", "gsm", "opus"]
    bandwidth_kbps: float = Field(gt=0)
    required_bandwidth_kbps: float = Field(gt=0)
    rtt_ms: float = Field(ge=0)
    jitter_ms: float = Field(ge=0)
    seed: int | None = None


class QosPredictResponse(BaseModel):
    r_factor: float
    mos: float
    delay_ms: float
    loss_pct: float
    codec: str


# --- optimize ---


class OptimizeRunRequest(BaseModel):
    problem: Literal["pb1_codec", "pb2_bts", "pb3_qos"]
    algorithm: str
    params: dict = Field(default_factory=dict)


class OptimizeRunResponse(BaseModel):
    job_id: str
    status: str


class OptimizeJobResponse(BaseModel):
    job_id: str
    problem: str
    algorithm: str
    status: str
    created_at: datetime
    finished_at: datetime | None
    result: dict | None
    error: str | None
