from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.ml.features import FEATURE_COLUMNS
from app.services.prediction_service import ModelBundle


class DummyModel:
    def predict(self, frame):
        return [int(frame.iloc[0].sum() > 0)]

    def predict_proba(self, frame):
        positive = 0.8 if frame.iloc[0].sum() > 0 else 0.2
        return [[1.0 - positive, positive]]


@pytest.fixture()
def feature_payload() -> dict[str, float]:
    return {column: float(idx + 1) / 10 for idx, column in enumerate(FEATURE_COLUMNS)}


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test_predictions.db'}",
        skip_model_load=True,
    )
    bundle = ModelBundle(
        model=DummyModel(),
        feature_columns=FEATURE_COLUMNS,
        model_name="test-model",
        model_version="test-version",
        model_run_id="test-run",
    )
    with TestClient(create_app(settings=settings, model_bundle=bundle)) as test_client:
        yield test_client

