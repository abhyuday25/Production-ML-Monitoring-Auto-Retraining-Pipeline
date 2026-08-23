from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DriftResult:
    detector: str
    score: float | None
    threshold: float | None
    drift_detected: bool
    sample_count: int
    feature: str | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftAssessment:
    overall_score: float
    severity: str
    drift_detected: bool
    triggered_detectors: list[str]
    top_drifting_features: list[str]
    timestamp: datetime
    window_size: int
    window_start: datetime | None = None
    window_end: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
