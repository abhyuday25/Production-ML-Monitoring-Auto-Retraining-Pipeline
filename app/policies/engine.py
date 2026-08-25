from __future__ import annotations

from app.core.config import Settings
from app.policies.drift_triggered import DriftTriggeredPolicy
from app.policies.error_threshold import ErrorThresholdPolicy
from app.policies.periodic import PeriodicPolicy
from app.policies.schemas import PolicyContext, RetrainingDecisionResult


class PolicyEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._policies = {
            "periodic": PeriodicPolicy(settings.periodic_retrain_interval),
            "error_threshold": ErrorThresholdPolicy(settings.error_window_size, settings.error_retrain_threshold, settings.min_error_samples),
            "drift_triggered": DriftTriggeredPolicy(
                settings.drift_retrain_score_threshold,
                settings.drift_retrain_min_severity,
                settings.retrain_cooldown_observations,
            ),
        }

    def evaluate(self, policy_name: str, context: PolicyContext) -> RetrainingDecisionResult:
        if policy_name not in self._policies:
            raise ValueError(f"Unsupported policy: {policy_name}")
        return self._policies[policy_name].evaluate(context)

    def evaluate_all(self, context: PolicyContext) -> list[RetrainingDecisionResult]:
        return [policy.evaluate(context) for policy in self._policies.values()]
