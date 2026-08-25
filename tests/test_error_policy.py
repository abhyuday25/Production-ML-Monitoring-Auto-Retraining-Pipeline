from app.policies.error_threshold import ErrorThresholdPolicy
from app.policies.schemas import PolicyContext


def test_error_policy_insufficient_labels():
    decision = ErrorThresholdPolicy(10, 0.2, 5).evaluate(PolicyContext(observation_index=3, errors=[1, 0, 1]))

    assert not decision.should_retrain
    assert decision.reason == "insufficient labeled samples"


def test_error_policy_below_threshold():
    decision = ErrorThresholdPolicy(10, 0.5, 5).evaluate(PolicyContext(observation_index=10, errors=[0, 0, 1, 0, 0]))

    assert not decision.should_retrain


def test_error_policy_above_threshold():
    decision = ErrorThresholdPolicy(10, 0.4, 5).evaluate(PolicyContext(observation_index=10, errors=[1, 0, 1, 1, 0]))

    assert decision.should_retrain
    assert decision.trigger_metrics["rolling_error"] == 0.6
