from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import mlflow
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import DeploymentEvent, PredictionLog, ShadowPredictionLog
from app.ml.features import FEATURE_COLUMNS
from app.observability.metrics import record_deployment_event_metric
from app.retraining.metrics import evaluate_classifier


@dataclass(frozen=True)
class ShadowCanaryResult:
    passed: bool
    reason: str
    samples: int
    champion_metrics: dict[str, float]
    candidate_metrics: dict[str, float]


def evaluate_shadow_canary(
    session: Session,
    champion_model,
    candidate_model,
    settings: Settings,
    *,
    champion_version: str | None,
    candidate_version: str | None,
) -> ShadowCanaryResult:
    rows = session.query(PredictionLog).filter(PredictionLog.ground_truth.is_not(None)).order_by(PredictionLog.id).all()
    records = []
    for row in rows:
        if not _in_canary(row.request_id, settings.canary_percentage):
            continue
        features = json.loads(row.input_features)
        if not all(column in features for column in FEATURE_COLUMNS):
            continue
        frame = pd.DataFrame([{column: float(features[column]) for column in FEATURE_COLUMNS}])
        champion_prediction = int(champion_model.predict(frame)[0])
        candidate_prediction = int(candidate_model.predict(frame)[0])
        champion_probability = _probability(champion_model, frame)
        candidate_probability = _probability(candidate_model, frame)
        session.add(
            ShadowPredictionLog(
                request_id=row.request_id,
                champion_model_version=champion_version,
                candidate_model_version=candidate_version,
                champion_prediction=champion_prediction,
                candidate_prediction=candidate_prediction,
                ground_truth=row.ground_truth,
                champion_probability=champion_probability,
                candidate_probability=candidate_probability,
            )
        )
        record = {column: float(features[column]) for column in FEATURE_COLUMNS}
        record["target"] = int(row.ground_truth)
        records.append(record)
    session.commit()
    if len(records) < settings.min_shadow_labeled_samples:
        return ShadowCanaryResult(False, "insufficient labeled shadow/canary samples", len(records), {}, {})
    frame = pd.DataFrame(records)
    champion_metrics = evaluate_classifier(champion_model, frame)
    candidate_metrics = evaluate_classifier(candidate_model, frame)
    metric = settings.primary_model_metric
    passed = candidate_metrics[metric] >= champion_metrics[metric] - settings.max_allowed_holdout_drop
    return ShadowCanaryResult(
        passed,
        "candidate passed shadow/canary comparison" if passed else "candidate underperformed shadow/canary comparison",
        len(records),
        champion_metrics,
        candidate_metrics,
    )


def record_deployment_event(
    session: Session,
    event_type: str,
    *,
    old_model_version: str | None,
    new_model_version: str | None,
    reason: str,
    metrics: dict,
) -> DeploymentEvent:
    event = DeploymentEvent(
        event_type=event_type,
        old_model_version=old_model_version,
        new_model_version=new_model_version,
        reason=reason,
        metrics=json.dumps(metrics, sort_keys=True),
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    record_deployment_event_metric(event_type)
    return event


def promote_candidate(settings: Settings, candidate_version: str | None) -> bool:
    if candidate_version is None:
        return False
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        client.set_registered_model_alias(settings.mlflow_model_name, settings.mlflow_model_alias, candidate_version)
        return True
    except Exception:
        return False


def rollback_to_previous(settings: Settings, previous_version: str | None) -> bool:
    return promote_candidate(settings, previous_version)


def evaluate_post_promotion(
    previous_metrics: dict[str, float],
    new_metrics: dict[str, float],
    settings: Settings,
) -> bool:
    metric = settings.primary_model_metric
    return float(new_metrics.get(metric, 0.0)) < float(previous_metrics.get(metric, 0.0)) - settings.rollback_metric_drop


def _in_canary(request_id: str, percentage: float) -> bool:
    bucket = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < percentage


def _probability(model, frame: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba(frame)[0]
    return float(probs[1] if len(probs) == 2 else max(probs))
