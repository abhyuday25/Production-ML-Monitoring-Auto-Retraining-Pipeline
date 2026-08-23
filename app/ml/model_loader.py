from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn

from app.core.config import Settings
from app.ml.features import FEATURE_COLUMNS
from app.services.prediction_service import ModelBundle


def load_model(settings: Settings) -> ModelBundle:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    model = mlflow.sklearn.load_model(model_uri)
    metadata = _load_local_metadata(settings)
    return ModelBundle(
        model=model,
        feature_columns=metadata.get("feature_columns", FEATURE_COLUMNS),
        model_name=settings.mlflow_model_name,
        model_version=str(metadata.get("model_version") or settings.mlflow_model_alias),
        model_run_id=metadata.get("run_id"),
    )


def _load_local_metadata(settings: Settings) -> dict:
    metadata_path = Path(settings.data_dir).parent / "models" / "model_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))
