from __future__ import annotations

from app.policies.schemas import PolicyContext, RetrainingDecisionResult, SEVERITY_ORDER


class DriftTriggeredPolicy:
    name = "drift_triggered"

    def __init__(self, score_threshold: float, min_severity: str, cooldown_observations: int) -> None:
        self.score_threshold = score_threshold
        self.min_severity = min_severity.upper()
        self.cooldown_observations = max(0, cooldown_observations)

    def evaluate(self, context: PolicyContext) -> RetrainingDecisionResult:
        drift = context.latest_drift or {}
        severity = str(drift.get("severity", "NONE")).upper()
        score = float(drift.get("overall_score", 0.0) or 0.0)
        detected = bool(drift.get("drift_detected", False))
        if context.last_retrain_index is not None and context.observation_index - context.last_retrain_index < self.cooldown_observations:
            return RetrainingDecisionResult(
                self.name,
                False,
                "retrain cooldown active",
                {"cooldown_observations": self.cooldown_observations, "last_retrain_index": context.last_retrain_index},
                observation_index=context.observation_index,
            )
        severity_ok = SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(self.min_severity, 3)
        score_ok = score >= self.score_threshold
        should = detected and (severity_ok or score_ok)
        return RetrainingDecisionResult(
            self.name,
            should,
            "drift assessment exceeded retrain threshold" if should else "drift assessment below retrain threshold",
            {
                "overall_score": score,
                "score_threshold": self.score_threshold,
                "severity": severity,
                "min_severity": self.min_severity,
                "drift_detected": detected,
            },
            observation_index=context.observation_index,
        )
