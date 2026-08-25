from __future__ import annotations

from app.policies.schemas import PolicyContext, RetrainingDecisionResult


class PeriodicPolicy:
    name = "periodic"

    def __init__(self, interval: int) -> None:
        if interval <= 0:
            raise ValueError("periodic interval must be positive")
        self.interval = interval

    def evaluate(self, context: PolicyContext) -> RetrainingDecisionResult:
        anchor = context.last_retrain_index if context.last_retrain_index is not None else 0
        since_last = context.observation_index - anchor
        already_triggered = context.last_policy_trigger_index is not None and context.last_policy_trigger_index >= anchor + self.interval
        should = since_last >= self.interval and not already_triggered
        reason = "periodic interval reached" if should else "periodic interval not reached"
        if already_triggered:
            reason = "periodic trigger already acknowledged for current interval"
        return RetrainingDecisionResult(
            self.name,
            should,
            reason,
            {"observations_since_last_retrain": since_last, "interval": self.interval},
            observation_index=context.observation_index,
        )
