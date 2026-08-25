from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.db.models import RetrainingDecision
from app.observability.metrics import record_policy_decision
from app.policies.schemas import CostAwareResult, RetrainingDecisionResult


def persist_retraining_decision(
    session: Session,
    decision: RetrainingDecisionResult,
    cost: CostAwareResult,
    *,
    model_version: str | None = None,
) -> RetrainingDecision:
    row = RetrainingDecision(
        policy=decision.policy,
        policy_triggered=int(decision.should_retrain),
        policy_reason=decision.reason,
        trigger_metrics=json.dumps(decision.trigger_metrics, sort_keys=True),
        cost_approved=int(cost.approved),
        estimated_drift_cost=cost.estimated_drift_cost,
        estimated_retraining_cost=cost.estimated_retraining_cost,
        net_benefit=cost.net_benefit,
        final_should_retrain=int(decision.should_retrain and cost.approved),
        model_version=model_version,
        observation_index=decision.observation_index,
        metadata_json=json.dumps({"cost_reason": cost.reason, "cost_metrics": cost.metrics}, sort_keys=True),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    record_policy_decision(
        decision.policy,
        decision.should_retrain,
        cost.approved,
        cost.estimated_drift_cost,
        cost.estimated_retraining_cost,
        cost.net_benefit,
    )
    return row
