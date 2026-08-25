from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import Settings
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from app.retraining.metrics import evaluate_classifier
from scripts.train_model import _latest_model_version

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateTrainingResult:
    model: object
    metrics: dict[str, float]
    training_sample_count: int
    duration_seconds: float
    run_id: str | None
    model_version: str | None
    registered: bool


def build_model_pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ]
    )


def train_candidate(
    training_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    settings: Settings,
    *,
    policy: str,
    trigger_reason: str,
    parent_model_version: str | None,
    register: bool = True,
) -> CandidateTrainingResult:
    started = time.perf_counter()
    model = build_model_pipeline(settings.random_seed)
    model.fit(training_frame[FEATURE_COLUMNS], training_frame[TARGET_COLUMN].astype(int))
    metrics = evaluate_classifier(model, holdout_frame)
    duration = time.perf_counter() - started
    run_id = None
    model_version = None
    registered = False
    if register:
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                mlflow.set_tags(
                    {
                        "run_type": "retraining",
                        "policy": policy,
                        "trigger_reason": trigger_reason[:250],
                        "parent_model_version": parent_model_version or "",
                    }
                )
                mlflow.log_params(
                    {
                        "model_type": "LogisticRegression",
                        "feature_count": len(FEATURE_COLUMNS),
                        "training_sample_count": len(training_frame),
                    }
                )
                mlflow.log_metrics(metrics | {"training_duration_seconds": duration})
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=settings.mlflow_model_name,
                    input_example=training_frame[FEATURE_COLUMNS].head(3),
                )
                client = mlflow.tracking.MlflowClient()
                model_version = _latest_model_version(client, settings.mlflow_model_name, run_id)
                registered = model_version is not None
        except Exception as exc:
            logger.warning("Candidate MLflow registration failed: %s", exc)
    return CandidateTrainingResult(model, metrics, len(training_frame), duration, run_id, model_version, registered)
