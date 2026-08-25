import pytest

from app.core.config import Settings
from app.policies.cost_aware import evaluate_cost_gate
from app.policies.schemas import RetrainingDecisionResult


def test_cost_gate_rejects_when_drift_cost_is_lower_than_retrain_cost():
    settings = Settings(expected_future_requests=100, fixed_retrain_cost=20, retrain_cost_per_1000_samples=0, deployment_cost_penalty=5)
    decision = RetrainingDecisionResult("error_threshold", True, "trigger", {"rolling_error": 0.12})

    result = evaluate_cost_gate(decision, settings, current_error=0.12, baseline_error=0.10, training_sample_count=1000)

    assert not result.approved
    assert result.estimated_drift_cost == pytest.approx(2.0)
    assert result.estimated_retraining_cost == 25.0


def test_cost_gate_approves_when_net_benefit_exceeds_minimum():
    settings = Settings(expected_future_requests=1000, fixed_retrain_cost=10, retrain_cost_per_1000_samples=1, deployment_cost_penalty=0)
    decision = RetrainingDecisionResult("error_threshold", True, "trigger", {"rolling_error": 0.30})

    result = evaluate_cost_gate(decision, settings, current_error=0.30, baseline_error=0.10, training_sample_count=1000)

    assert result.approved
    assert result.estimated_drift_cost == 199.99999999999997
    assert result.estimated_retraining_cost == 11.0
