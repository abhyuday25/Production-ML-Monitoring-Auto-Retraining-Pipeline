from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import mlflow.sklearn

from app.core.config import get_settings
from app.core.logging import configure_logging
from scripts.prepare_data import prepare_datasets
from scripts.train_model import train_model

logger = logging.getLogger(__name__)


def champion_is_loadable() -> bool:
    settings = get_settings()
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        version = client.get_model_version_by_alias(settings.mlflow_model_name, settings.mlflow_model_alias)
        model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
        mlflow.sklearn.load_model(model_uri)
        logger.info("MLflow champion alias is loadable model_version=%s", version.version)
        return True
    except Exception as exc:
        logger.warning("MLflow champion alias is missing or not loadable; retraining bootstrap model: %s", exc)
        return False


def ensure_model(data_dir: Path, rows: int, random_seed: int, force: bool) -> dict:
    if champion_is_loadable() and not force:
        logger.info("MLflow champion alias already exists and artifacts are loadable; skipping bootstrap training")
        return {"status": "exists"}

    train_path = data_dir / "processed" / "train.csv"
    reference_path = data_dir / "reference" / "reference.csv"
    production_path = data_dir / "production" / "production.csv"
    holdout_path = data_dir / "processed" / "holdout.csv"
    if not all(path.exists() for path in [train_path, reference_path, production_path, holdout_path]):
        logger.info("Prepared datasets missing; generating deterministic splits")
        prepare_datasets(data_dir, rows=rows, random_seed=random_seed)

    logger.info("Training and registering champion model")
    output = train_model(data_dir)
    if not champion_is_loadable():
        raise RuntimeError("Champion model was trained but could not be loaded from MLflow artifacts")
    return {"status": "trained", "model_version": output.get("model_version")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure the MLflow champion model exists for local/Docker startup.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Train a new champion even if the alias already exists.")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    print(ensure_model(Path(args.data_dir), args.rows, args.random_seed, args.force))
