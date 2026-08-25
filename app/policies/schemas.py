from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


SEVERITY_ORDER = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


@dataclass(frozen=True)
class PolicyContext:
    observation_index: int
    last_retrain_index: int | None = None
    last_policy_trigger_index: int | None = None
    errors: list[int] = field(default_factory=list)
    latest_drift: dict[str, Any] | None = None
    current_model_version: str | None = None


@dataclass(frozen=True)
class RetrainingDecisionResult:
    policy: str
    should_retrain: bool
    reason: str
    trigger_metrics: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    observation_index: int | None = None


@dataclass(frozen=True)
class CostAwareResult:
    approved: bool
    policy: str
    estimated_drift_cost: float
    estimated_retraining_cost: float
    net_benefit: float
    minimum_required_benefit: float
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
