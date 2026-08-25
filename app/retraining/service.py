from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import DriftAlert, PredictionLog, RetrainingRun
from app.ml.features import TARGET_COLUMN
from app.ml.model_loader import load_model
from app.observability.metrics import record_retraining_run
from app.policies.cost_aware import evaluate_cost_gate
from app.policies.engine import PolicyEngine
from app.policies.persistence import persist_retraining_decision
from app.policies.schemas import CostAwareResult, PolicyContext, RetrainingDecisionResult
from app.retraining.dataset_builder import build_labeled_production_frame, build_retraining_frame
from app.retraining.deployment import evaluate_post_promotion, evaluate_shadow_canary, promote_candidate, record_deployment_event, rollback_to_previous
from app.retraining.metrics import evaluate_classifier
from app.retraining.trainer import train_candidate
from app.retraining.validator import validate_candidate

logger = logging.getLogger(__name__)


def evaluate_policy_with_cost(settings: Settings, session: Session, policy_name: str) -> tuple[RetrainingDecisionResult, CostAwareResult]:
    context = build_policy_context(settings, session)
    decision = PolicyEngine(settings).evaluate(policy_name, context)
    current_error = _rolling_error(context.errors, settings.error_window_size) if context.errors else None
    training_count = _estimated_training_sample_count(settings, session)
    cost = evaluate_cost_gate(
        decision,
        settings,
        current_error=current_error,
        baseline_error=0.10,
        training_sample_count=training_count,
    )
    persist_retraining_decision(session, decision, cost, model_version=context.current_model_version)
    logger.info("Policy evaluated policy=%s triggered=%s cost_approved=%s", policy_name, decision.should_retrain, cost.approved)
    return decision, cost


def run_retraining_pipeline(settings: Settings, session: Session, policy_name: str) -> dict:
    decision, cost = evaluate_policy_with_cost(settings, session, policy_name)
    if not decision.should_retrain or not cost.approved:
        return {"status": "skipped", "decision": asdict(decision), "cost": asdict(cost)}

    started = time.perf_counter()
    run = RetrainingRun(status="running", policy=policy_name, champion_model_version=None)
    session.add(run)
    session.commit()
    session.refresh(run)
    record_retraining_run(policy_name, "running")
    try:
        champion = load_model(settings)
        holdout = _load_holdout(settings)
        training_frame = build_retraining_frame(session, settings)
        champion_metrics = evaluate_classifier(champion.model, holdout)
        candidate = train_candidate(
            training_frame,
            holdout,
            settings,
            policy=policy_name,
            trigger_reason=decision.reason,
            parent_model_version=champion.model_version,
            register=True,
        )
        validation = validate_candidate(candidate.metrics, champion_metrics, settings)
        run.training_sample_count = candidate.training_sample_count
        run.candidate_run_id = candidate.run_id
        run.candidate_model_version = candidate.model_version
        run.champion_model_version = champion.model_version
        run.validation_result = json.dumps(asdict(validation), sort_keys=True)
        if candidate.registered:
            record_deployment_event(
                session,
                "candidate_registered",
                old_model_version=champion.model_version,
                new_model_version=candidate.model_version,
                reason="candidate registered after training",
                metrics=candidate.metrics,
            )
        if not validation.passed:
            run.status = "rejected"
            run.failure_reason = validation.reason
            record_deployment_event(
                session,
                "candidate_rejected",
                old_model_version=champion.model_version,
                new_model_version=candidate.model_version,
                reason=validation.reason,
                metrics=candidate.metrics,
            )
            return _finish_run(session, run, started, {"status": "rejected", "reason": validation.reason})

        shadow = evaluate_shadow_canary(
            session,
            champion.model,
            candidate.model,
            settings,
            champion_version=champion.model_version,
            candidate_version=candidate.model_version,
        )
        record_deployment_event(
            session,
            "shadow_started",
            old_model_version=champion.model_version,
            new_model_version=candidate.model_version,
            reason=shadow.reason,
            metrics=asdict(shadow),
        )
        if not shadow.passed:
            run.status = "rejected"
            run.failure_reason = shadow.reason
            run.promotion_result = json.dumps(asdict(shadow), sort_keys=True)
            record_deployment_event(
                session,
                "candidate_rejected",
                old_model_version=champion.model_version,
                new_model_version=candidate.model_version,
                reason=shadow.reason,
                metrics=asdict(shadow),
            )
            return _finish_run(session, run, started, {"status": "rejected", "reason": shadow.reason})

        promoted = promote_candidate(settings, candidate.model_version)
        run.status = "promoted" if promoted else "promotion_failed"
        run.promotion_result = json.dumps({"promoted": promoted, "shadow": asdict(shadow)}, sort_keys=True)
        record_deployment_event(
            session,
            "candidate_promoted" if promoted else "candidate_rejected",
            old_model_version=champion.model_version,
            new_model_version=candidate.model_version,
            reason="candidate promoted" if promoted else "candidate could not be promoted",
            metrics={"candidate_metrics": candidate.metrics, "champion_metrics": champion_metrics},
        )
        if promoted and evaluate_post_promotion(shadow.champion_metrics, shadow.candidate_metrics, settings):
            rolled_back = rollback_to_previous(settings, champion.model_version)
            run.status = "rolled_back" if rolled_back else "rollback_failed"
            run.promotion_result = json.dumps({"promoted": promoted, "rolled_back": rolled_back, "shadow": asdict(shadow)}, sort_keys=True)
            record_deployment_event(
                session,
                "rollback",
                old_model_version=candidate.model_version,
                new_model_version=champion.model_version,
                reason="post-promotion regression exceeded rollback threshold",
                metrics={"candidate_metrics": shadow.candidate_metrics, "previous_champion_metrics": shadow.champion_metrics},
            )
        return _finish_run(session, run, started, {"status": run.status, "candidate_version": candidate.model_version})
    except Exception as exc:
        run.status = "failed"
        run.failure_reason = str(exc)
        return _finish_run(session, run, started, {"status": "failed", "reason": str(exc)})


def build_policy_context(settings: Settings, session: Session) -> PolicyContext:
    observation_index = int(session.query(func.count(PredictionLog.id)).scalar() or 0)
    errors = _load_errors(session, settings.error_window_size)
    latest_drift = _latest_drift(session)
    latest_decision = _latest_approved_decision(session)
    last_retrain_index = latest_decision.observation_index if latest_decision is not None else None
    last_policy_trigger_index = _latest_policy_trigger_index(session)
    current_model_version = _latest_model_version(settings)
    return PolicyContext(
        observation_index=observation_index,
        last_retrain_index=last_retrain_index,
        last_policy_trigger_index=last_policy_trigger_index,
        errors=errors,
        latest_drift=latest_drift,
        current_model_version=current_model_version,
    )


def _load_errors(session: Session, limit: int) -> list[int]:
    rows = (
        session.query(PredictionLog)
        .filter(PredictionLog.ground_truth.is_not(None))
        .order_by(desc(PredictionLog.id))
        .limit(limit)
        .all()
    )
    rows.reverse()
    return [0 if int(row.prediction) == int(row.ground_truth) else 1 for row in rows]


def _latest_drift(session: Session) -> dict:
    alert = session.query(DriftAlert).order_by(desc(DriftAlert.id)).first()
    if alert is None:
        return {}
    return {
        "overall_score": alert.overall_score,
        "severity": alert.severity,
        "drift_detected": bool(alert.drift_detected),
        "triggered_detectors": json.loads(alert.triggered_detectors or "[]"),
        "top_drifting_features": json.loads(alert.top_drifting_features or "[]"),
    }


def _latest_approved_decision(session: Session):
    from app.db.models import RetrainingDecision

    return session.query(RetrainingDecision).filter(RetrainingDecision.final_should_retrain == 1).order_by(desc(RetrainingDecision.id)).first()


def _latest_policy_trigger_index(session: Session) -> int | None:
    from app.db.models import RetrainingDecision

    row = session.query(RetrainingDecision).filter(RetrainingDecision.policy_triggered == 1).order_by(desc(RetrainingDecision.id)).first()
    return row.observation_index if row is not None else None


def _latest_model_version(settings: Settings) -> str | None:
    path = Path(settings.data_dir).parent / "models" / "model_metadata.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("model_version"))
    except Exception:
        return None


def _estimated_training_sample_count(settings: Settings, session: Session) -> int:
    base_path = Path(settings.data_dir) / "processed" / "train.csv"
    base_count = len(pd.read_csv(base_path)) if base_path.exists() else 0
    labeled_count = (
        session.query(func.count(PredictionLog.id))
        .filter(PredictionLog.ground_truth.is_not(None))
        .scalar()
        or 0
    )
    return int(base_count + min(labeled_count, settings.max_production_retrain_samples))


def _rolling_error(errors: list[int], window_size: int) -> float:
    window = errors[-window_size:]
    return float(sum(window) / len(window)) if window else 0.0


def _load_holdout(settings: Settings) -> pd.DataFrame:
    path = Path(settings.data_dir) / "processed" / "holdout.csv"
    if not path.exists():
        raise FileNotFoundError(f"Holdout split not found at {path}")
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame:
        raise ValueError("Holdout split is missing target")
    return frame


def _finish_run(session: Session, run: RetrainingRun, started: float, payload: dict) -> dict:
    run.timestamp_end = datetime.now(timezone.utc)
    run.duration_seconds = time.perf_counter() - started
    session.commit()
    record_retraining_run(run.policy, run.status, run.duration_seconds)
    payload["run_id"] = run.id
    return payload
