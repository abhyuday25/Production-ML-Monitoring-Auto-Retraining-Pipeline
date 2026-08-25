from app.policies.drift_triggered import DriftTriggeredPolicy
from app.policies.schemas import PolicyContext


def test_drift_policy_low_severity_no_trigger():
    policy = DriftTriggeredPolicy(0.7, "HIGH", 300)
    decision = policy.evaluate(PolicyContext(observation_index=500, latest_drift={"overall_score": 0.3, "severity": "LOW", "drift_detected": True}))

    assert not decision.should_retrain


def test_drift_policy_high_severity_triggers():
    policy = DriftTriggeredPolicy(0.7, "HIGH", 300)
    decision = policy.evaluate(PolicyContext(observation_index=500, latest_drift={"overall_score": 0.72, "severity": "HIGH", "drift_detected": True}))

    assert decision.should_retrain


def test_drift_policy_cooldown_suppresses_repeated_trigger():
    policy = DriftTriggeredPolicy(0.7, "HIGH", 300)
    decision = policy.evaluate(
        PolicyContext(
            observation_index=600,
            last_retrain_index=500,
            latest_drift={"overall_score": 0.9, "severity": "CRITICAL", "drift_detected": True},
        )
    )

    assert not decision.should_retrain
    assert decision.reason == "retrain cooldown active"
