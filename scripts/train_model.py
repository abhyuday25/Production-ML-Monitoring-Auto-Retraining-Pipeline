from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.features import TARGET_COLUMN

logger = logging.getLogger(__name__)


def train_model(data_dir: Path) -> dict:
    settings = get_settings()
    train_path = data_dir / "processed" / "train.csv"
    holdout_path = data_dir / "processed" / "holdout.csv"
    metadata_path = data_dir / "processed" / "dataset_metadata.json"
    if not train_path.exists() or not holdout_path.exists():
        raise FileNotFoundError("Prepared datasets are missing. Run python scripts/prepare_data.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    train = pd.read_csv(train_path)
    holdout = pd.read_csv(holdout_path)

    x_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]
    x_holdout = holdout[feature_columns]
    y_holdout = holdout[TARGET_COLUMN]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info("Training baseline model")
    with mlflow.start_run() as run:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_holdout)
        probabilities = pipeline.predict_proba(x_holdout)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_holdout, predictions),
            "precision": precision_score(y_holdout, predictions, zero_division=0),
            "recall": recall_score(y_holdout, predictions, zero_division=0),
            "f1": f1_score(y_holdout, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_holdout, probabilities),
        }
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": 1000,
                "feature_count": len(feature_columns),
                "train_rows": len(train),
                "holdout_rows": len(holdout),
            }
        )
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name,
            input_example=x_train.head(3),
        )
        client = mlflow.tracking.MlflowClient()
        model_version = _latest_model_version(client, settings.mlflow_model_name, run.info.run_id)
        if model_version is not None:
            client.set_registered_model_alias(settings.mlflow_model_name, settings.mlflow_model_alias, model_version)

        output = {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "model_name": settings.mlflow_model_name,
            "model_alias": settings.mlflow_model_alias,
            "model_version": model_version,
            "feature_columns": feature_columns,
            "metrics": metrics,
        }
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        (models_dir / "model_metadata.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info("Training complete run_id=%s model_version=%s metrics=%s", run.info.run_id, model_version, metrics)
        print(json.dumps(output, indent=2))
        return output


def _latest_model_version(client: mlflow.tracking.MlflowClient, model_name: str, run_id: str) -> str | None:
    versions = client.search_model_versions(f"name = '{model_name}'")
    matching = [version for version in versions if version.run_id == run_id]
    if not matching:
        return None
    return max(matching, key=lambda version: int(version.version)).version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register the Week 1 baseline model.")
    parser.add_argument("--data-dir", default="data", help="Root data directory containing prepared splits.")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    train_model(Path(args.data_dir))
