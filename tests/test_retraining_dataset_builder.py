import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db.database import Base
from app.db.models import PredictionLog
from app.ml.features import FEATURE_COLUMNS
from app.retraining.dataset_builder import build_labeled_production_frame


def test_dataset_builder_uses_only_labeled_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'builder.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    features = {column: float(idx) for idx, column in enumerate(FEATURE_COLUMNS)}

    with session_factory() as session:
        session.add(PredictionLog(request_id="labeled", input_features=json.dumps(features), prediction=1, prediction_probability=0.8, ground_truth=1, model_name="m", latency_ms=1))
        session.add(PredictionLog(request_id="unlabeled", input_features=json.dumps(features), prediction=1, prediction_probability=0.8, ground_truth=None, model_name="m", latency_ms=1))
        session.commit()

        frame = build_labeled_production_frame(session, Settings(min_retrain_samples=1))

    assert len(frame) == 1
    assert "target" in frame.columns
    assert set(FEATURE_COLUMNS).issubset(frame.columns)


def test_dataset_builder_reports_insufficient_data(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'builder.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        with pytest.raises(ValueError, match="insufficient labeled production samples"):
            build_labeled_production_frame(session, Settings(min_retrain_samples=1))
