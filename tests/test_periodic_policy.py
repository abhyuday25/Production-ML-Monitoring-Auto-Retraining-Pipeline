import pytest

from app.policies.periodic import PeriodicPolicy
from app.policies.schemas import PolicyContext


def test_periodic_no_trigger_before_interval():
    decision = PeriodicPolicy(100).evaluate(PolicyContext(observation_index=99))

    assert not decision.should_retrain


def test_periodic_triggers_at_interval():
    decision = PeriodicPolicy(100).evaluate(PolicyContext(observation_index=100))

    assert decision.should_retrain


def test_periodic_does_not_repeat_after_acknowledgement():
    decision = PeriodicPolicy(100).evaluate(PolicyContext(observation_index=150, last_policy_trigger_index=100))

    assert not decision.should_retrain


def test_periodic_rejects_invalid_interval():
    with pytest.raises(ValueError):
        PeriodicPolicy(0)
