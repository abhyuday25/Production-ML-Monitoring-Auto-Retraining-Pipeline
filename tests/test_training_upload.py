from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.api import training as training_module
from app.core.config import Settings
from app.main import create_app
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from app.services.prediction_service import ModelBundle


class DummyModel:
    def predict(self, frame):
        return [0 for _ in range(len(frame))]

    def predict_proba(self, frame):
        return [[0.7, 0.3] for _ in range(len(frame))]


def test_upload_training_dataset_creates_completed_job(monkeypatch, tmp_path: Path):
    training_module._jobs.clear()
    calls = {"train_data_dir": None, "load_model": 0}

    def fake_train_model(data_dir: Path) -> dict:
        calls["train_data_dir"] = data_dir
        return {
            "run_id": "run-1",
            "model_uri": "runs:/run-1/model",
            "model_name": "test-model",
            "model_alias": "champion",
            "model_version": "3",
            "feature_columns": FEATURE_COLUMNS,
            "metrics": {"accuracy": 0.91, "precision": 0.9, "recall": 0.92, "f1": 0.91, "roc_auc": 0.95},
        }

    def fake_load_model(settings):
        calls["load_model"] += 1
        return ModelBundle(
            model=DummyModel(),
            feature_columns=FEATURE_COLUMNS,
            model_name=settings.mlflow_model_name,
            model_version="3",
            model_run_id="run-1",
        )

    monkeypatch.setattr(training_module, "train_model", fake_train_model)
    monkeypatch.setattr(training_module, "load_model", fake_load_model)

    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'predictions.db'}",
        data_dir=str(tmp_path / "data"),
        skip_model_load=True,
    )
    with TestClient(create_app(settings=settings, model_bundle=dummy_model_bundle())) as client:
        csv_bytes = _dataset_frame().to_csv(index=False).encode("utf-8")
        response = client.post(
            "/training/upload",
            files={"file": ("dataset.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        job_id = response.json()["id"]

        job_response = client.get(f"/training/jobs/{job_id}")

    assert job_response.status_code == 200
    job = job_response.json()
    assert job["status"] == "completed"
    assert job["result"]["model_version"] == "3"
    assert job["result"]["metrics"]["f1"] == 0.91
    assert calls["load_model"] == 1
    assert (calls["train_data_dir"] / "processed" / "train.csv").exists()
    assert (calls["train_data_dir"] / "processed" / "holdout.csv").exists()


def test_upload_training_dataset_rejects_missing_columns(tmp_path: Path):
    training_module._jobs.clear()
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'predictions.db'}",
        data_dir=str(tmp_path / "data"),
        skip_model_load=True,
    )
    with TestClient(create_app(settings=settings, model_bundle=dummy_model_bundle())) as client:
        response = client.post(
            "/training/upload",
            files={"file": ("dataset.csv", b"feature_0,feature_1\n1,0\n", "text/csv")},
        )

    assert response.status_code == 422
    assert "target column" in response.json()["detail"]


def test_prepare_uploaded_dataset_accepts_creditcard_schema(tmp_path: Path):
    source_path = tmp_path / "creditcard.csv"
    _creditcard_frame().to_csv(source_path, index=False)
    settings = Settings(data_dir=str(tmp_path / "data"))

    data_root = training_module._prepare_uploaded_dataset(source_path, settings, "job-1")

    metadata = json.loads((data_root / "processed" / "dataset_metadata.json").read_text(encoding="utf-8"))
    train = pd.read_csv(data_root / "processed" / "train.csv")
    holdout = pd.read_csv(data_root / "processed" / "holdout.csv")
    assert "Class" not in train.columns
    assert TARGET_COLUMN in train.columns
    assert "V1" in train.columns
    assert "Amount" in train.columns
    assert metadata["source_target_column"] == "Class"
    assert metadata["feature_columns"] == ["Time", *[f"V{idx}" for idx in range(1, 29)], "Amount"]
    assert set(holdout[TARGET_COLUMN]) == {0, 1}


def _dataset_frame() -> pd.DataFrame:
    rows = []
    for idx in range(80):
        row = {column: float(idx + offset) / 10 for offset, column in enumerate(FEATURE_COLUMNS)}
        row[TARGET_COLUMN] = idx % 2
        rows.append(row)
    return pd.DataFrame(rows)


def _creditcard_frame() -> pd.DataFrame:
    rows = []
    for idx in range(80):
        row = {"Time": float(idx)}
        for feature_idx in range(1, 29):
            row[f"V{feature_idx}"] = float(idx + feature_idx) / 100
        row["Amount"] = float(idx % 25) + 0.5
        row["Class"] = idx % 2
        rows.append(row)
    return pd.DataFrame(rows)


def dummy_model_bundle() -> ModelBundle:
    return ModelBundle(
        model=DummyModel(),
        feature_columns=FEATURE_COLUMNS,
        model_name="test-model",
        model_version="1",
        model_run_id="run-0",
    )
