from __future__ import annotations

from app.core.config import Settings
from app.policies.schemas import CostAwareResult, RetrainingDecisionResult


def evaluate_cost_gate(
    decision: RetrainingDecisionResult,
    settings: Settings,
    *,
    current_error: float | None,
    baseline_error: float,
    training_sample_count: int,
) -> CostAwareResult:
    if not decision.should_retrain:
        return CostAwareResult(False, decision.policy, 0.0, 0.0, 0.0, settings.min_retrain_net_benefit, "policy did not propose retraining")
    effective_error = current_error if current_error is not None else float(decision.trigger_metrics.get("rolling_error", baseline_error))
    performance_degradation = max(0.0, effective_error - baseline_error)
    estimated_drift_cost = performance_degradation * settings.expected_future_requests * settings.business_error_cost_weight
    estimated_retraining_cost = (
        settings.fixed_retrain_cost
        + (training_sample_count / 1000.0) * settings.retrain_cost_per_1000_samples
        + settings.deployment_cost_penalty
    )
    net_benefit = estimated_drift_cost - estimated_retraining_cost
    approved = net_benefit >= settings.min_retrain_net_benefit
    return CostAwareResult(
        approved,
        decision.policy,
        estimated_drift_cost,
        estimated_retraining_cost,
        net_benefit,
        settings.min_retrain_net_benefit,
        "expected benefit exceeds retraining cost" if approved else "expected benefit below retraining cost",
        {
            "current_error": effective_error,
            "baseline_error": baseline_error,
            "performance_degradation": performance_degradation,
            "training_sample_count": training_sample_count,
        },
    )
