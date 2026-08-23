from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from app.core.config import Settings
from app.drift.schemas import DriftAssessment, DriftResult


def aggregate_drift_results(results: list[DriftResult], settings: Settings, *, window_size: int) -> DriftAssessment:
    weights = _normalized_weights(
        {
            "psi": settings.drift_weight_psi,
            "ks": settings.drift_weight_ks,
            "adwin": settings.drift_weight_adwin,
            "confidence": settings.drift_weight_confidence,
        }
    )
    detector_scores = {
        "psi": _max_normalized(results, "psi", settings.psi_drift_threshold),
        "ks": _max_ks(results),
        "adwin": 1.0 if any(result.detector == "adwin" and result.drift_detected for result in results) else 0.0,
        "confidence": _max_normalized(results, "confidence", settings.confidence_drift_threshold),
    }
    overall = min(1.0, sum(weights[name] * detector_scores[name] for name in weights))
    severity = _severity(overall, settings)
    triggered = sorted({result.detector for result in results if result.drift_detected})
    top_features = _top_features(results, settings.top_drift_features)
    window_start = next((result.window_start for result in results if result.window_start is not None), None)
    window_end = next((result.window_end for result in results if result.window_end is not None), None)
    return DriftAssessment(
        overall_score=overall,
        severity=severity,
        drift_detected=overall >= settings.drift_low_threshold or bool(triggered),
        triggered_detectors=triggered,
        top_drifting_features=top_features,
        timestamp=datetime.now(timezone.utc),
        window_size=window_size,
        window_start=window_start,
        window_end=window_end,
        metadata={"detector_scores": detector_scores, "weights": weights},
    )


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        return {name: 0.0 for name in weights}
    return {name: max(value, 0.0) / total for name, value in weights.items()}


def _max_normalized(results: list[DriftResult], detector: str, threshold: float) -> float:
    values = [result.score / threshold for result in results if result.detector == detector and result.score is not None and threshold > 0]
    return min(1.0, max(values, default=0.0))


def _max_ks(results: list[DriftResult]) -> float:
    values = [result.score for result in results if result.detector == "ks" and result.score is not None and result.drift_detected]
    return min(1.0, max(values, default=0.0))


def _severity(score: float, settings: Settings) -> str:
    if score >= settings.drift_critical_threshold:
        return "CRITICAL"
    if score >= settings.drift_high_threshold:
        return "HIGH"
    if score >= settings.drift_medium_threshold:
        return "MEDIUM"
    if score >= settings.drift_low_threshold:
        return "LOW"
    return "NONE"


def _top_features(results: list[DriftResult], limit: int) -> list[str]:
    feature_scores: dict[str, float] = defaultdict(float)
    for result in results:
        if result.feature is None or result.score is None:
            continue
        if result.detector == "psi" and result.threshold:
            score = result.score / result.threshold
        elif result.detector == "ks":
            score = result.score
        else:
            score = result.score
        feature_scores[result.feature] += score
    return [feature for feature, _ in sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)[:limit]]
