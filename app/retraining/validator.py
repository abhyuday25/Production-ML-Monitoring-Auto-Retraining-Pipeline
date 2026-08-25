from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float]
    metric_name: str


def validate_candidate(candidate_metrics: dict[str, float], champion_metrics: dict[str, float], settings: Settings) -> ValidationResult:
    metric = settings.primary_model_metric
    candidate_value = float(candidate_metrics.get(metric, 0.0))
    champion_value = float(champion_metrics.get(metric, 0.0))
    if candidate_value < settings.min_candidate_metric:
        return ValidationResult(False, "candidate below minimum quality threshold", candidate_metrics, champion_metrics, metric)
    if candidate_value < champion_value - settings.max_allowed_holdout_drop:
        return ValidationResult(False, "candidate holdout regression exceeds allowed drop", candidate_metrics, champion_metrics, metric)
    if candidate_value + 1e-12 < champion_value + settings.min_champion_improvement:
        return ValidationResult(False, "candidate did not beat champion improvement threshold", candidate_metrics, champion_metrics, metric)
    return ValidationResult(True, "candidate passed holdout and champion comparison", candidate_metrics, champion_metrics, metric)
