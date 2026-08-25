import json

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.database import Base
from app.db.models import DeploymentEvent, PredictionLog, ShadowPredictionLog
from app.ml.features import FEATURE_COLUMNS
from app.retraining.deployment import evaluate_post_promotion, evaluate_shadow_canary, record_deployment_event
from app.retraining.validator import validate_candidate


class ConstantModel:
    def __init__(self, value: int):
        self.value = value

    def predict(self, frame: pd.DataFrame):
        return [self.value] * len(frame)

    def predict_proba(self, frame: pd.DataFrame):
        positive = 0.9 if self.value == 1 else 0.1
        return [[1.0 - positive, positive] for _ in range(len(frame))]


def test_poor_candidate_is_rejected():
    result = validate_candidate({"f1": 0.5}, {"f1": 0.8}, Settings(min_candidate_metric=0.7))

    assert not result.passed


def test_better_candidate_passes_validation():
    result = validate_candidate({"f1": 0.83}, {"f1": 0.8}, Settings(min_candidate_metric=0.7))

    assert result.passed


def test_holdout_regression_prevents_promotion():
    result = validate_candidate({"f1": 0.75}, {"f1": 0.82}, Settings(min_candidate_metric=0.7, max_allowed_holdout_drop=0.03))

    assert not result.passed


def test_shadow_canary_records_candidate_predictions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'shadow.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    features = {column: 1.0 for column in FEATURE_COLUMNS}
    with session_factory() as session:
        for idx in range(5):
            session.add(
                PredictionLog(
                    request_id=f"req-{idx}",
                    input_features=json.dumps(features),
                    prediction=1,
                    prediction_probability=0.9,
                    ground_truth=1,
                    model_name="champion",
                    latency_ms=1,
                )
            )
        session.commit()
        result = evaluate_shadow_canary(
            session,
            ConstantModel(1),
            ConstantModel(1),
            Settings(min_shadow_labeled_samples=1, canary_percentage=100),
            champion_version="1",
            candidate_version="2",
        )

        assert result.passed
        assert session.query(ShadowPredictionLog).count() == 5


def test_deployment_event_and_rollback_rule(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'deploy.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        record_deployment_event(session, "rollback", old_model_version="2", new_model_version="1", reason="regression", metrics={"f1": 0.6})
        assert session.query(DeploymentEvent).count() == 1

    assert evaluate_post_promotion({"f1": 0.8}, {"f1": 0.75}, Settings(rollback_metric_drop=0.03))
