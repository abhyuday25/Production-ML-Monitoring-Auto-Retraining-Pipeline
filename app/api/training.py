from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

import pandas as pd
from fastapi import BackgroundTasks, HTTPException, Request, UploadFile, status
from sklearn.model_selection import train_test_split

from app.core.config import Settings
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from app.ml.model_loader import load_model
from app.observability.metrics import metrics, record_retraining_run
from app.services.prediction_service import PredictionService
from scripts.train_model import train_model


@dataclass
class TrainingJob:
    id: str
    status: str
    filename: str
    created_at: str
    updated_at: str
    message: str
    result: dict | None = None
    error: str | None = None
    grafana_url: str = "http://localhost:3000/d/mlops-monitoring-auto-retraining/ml-monitoring-auto-retraining"
    mlflow_url: str = "http://localhost:5000"


_jobs: dict[str, TrainingJob] = {}
_jobs_lock = threading.Lock()


def create_training_job(request: Request, background_tasks: BackgroundTasks, upload: UploadFile) -> dict:
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a CSV file.")

    settings: Settings = getattr(request.app.state, "settings", Settings())
    job_id = uuid.uuid4().hex
    created_at = _now()
    job = TrainingJob(
        id=job_id,
        status="queued",
        filename=upload.filename,
        created_at=created_at,
        updated_at=created_at,
        message="Dataset uploaded. Training is queued.",
    )
    _store_job(job)

    try:
        source_path = _save_upload(upload, settings, job_id)
        data_root = _prepare_uploaded_dataset(source_path, settings, job_id)
    except ValueError as exc:
        job.status = "failed"
        job.message = "Dataset validation failed."
        job.error = str(exc)
        job.updated_at = _now()
        _store_job(job)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    background_tasks.add_task(_run_training_job, request.app, job_id, data_root)
    return _job_payload(job)


def list_training_jobs() -> dict:
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda item: item.created_at, reverse=True)
        return {"jobs": [_job_payload(job) for job in jobs]}


def get_training_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found.")
    return _job_payload(job)


def _run_training_job(app, job_id: str, data_root: Path) -> None:
    _update_job(job_id, status="running", message="Training model and registering candidate in MLflow.")
    record_retraining_run("manual_upload", "running")
    try:
        result = train_model(data_root)
        _refresh_prediction_service(app)
        model_version = result.get("model_version")
        if model_version is not None and str(model_version).isdigit():
            metrics.active_model_version.set(float(model_version))
        model_metrics = result.get("metrics", {})
        if "accuracy" in model_metrics:
            metrics.model_accuracy.set(float(model_metrics["accuracy"]))
        if "f1" in model_metrics:
            metrics.model_f1.set(float(model_metrics["f1"]))
        record_retraining_run("manual_upload", "promoted")
        _update_job(
            job_id,
            status="completed",
            message="Training completed and the API model was refreshed.",
            result=result,
        )
    except Exception as exc:
        record_retraining_run("manual_upload", "failed")
        _update_job(job_id, status="failed", message="Training failed.", error=str(exc))


def _refresh_prediction_service(app) -> None:
    settings = app.state.settings
    session_factory = app.state.session_factory
    bundle = load_model(settings)
    app.state.prediction_service = PredictionService(bundle, session_factory)


def _save_upload(upload: UploadFile, settings: Settings, job_id: str) -> Path:
    upload_dir = Path(settings.data_dir) / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    source_path = upload_dir / "source.csv"
    source_path.write_bytes(upload.file.read())
    return source_path


def _prepare_uploaded_dataset(source_path: Path, settings: Settings, job_id: str) -> Path:
    uploaded_frame = pd.read_csv(source_path)
    frame, feature_columns, source_target_column = _normalize_dataset(uploaded_frame)
    _validate_dataset(frame, feature_columns)

    train, remaining = train_test_split(
        frame,
        test_size=0.40,
        stratify=frame[TARGET_COLUMN],
        random_state=settings.random_seed,
    )
    reference, remaining = train_test_split(
        remaining,
        test_size=0.625,
        stratify=remaining[TARGET_COLUMN],
        random_state=settings.random_seed,
    )
    production, holdout = train_test_split(
        remaining,
        test_size=0.40,
        stratify=remaining[TARGET_COLUMN],
        random_state=settings.random_seed,
    )

    data_root = Path(settings.data_dir) / "training_jobs" / job_id
    paths = {
        "train": data_root / "processed" / "train.csv",
        "reference": data_root / "reference" / "reference.csv",
        "production": data_root / "production" / "production.csv",
        "holdout": data_root / "processed" / "holdout.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(paths["train"], index=False)
    reference.to_csv(paths["reference"], index=False)
    production.to_csv(paths["production"], index=False)
    holdout.to_csv(paths["holdout"], index=False)
    metadata = {
        "dataset": source_path.name,
        "source_path": str(source_path),
        "random_seed": settings.random_seed,
        "rows": int(len(frame)),
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "source_target_column": source_target_column,
        "split_rows": {name: int(pd.read_csv(path).shape[0]) for name, path in paths.items()},
        "notes": "Uploaded dataset split into 60% train, 15% reference, 15% production, 10% holdout.",
    }
    (data_root / "processed" / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data_root


def _normalize_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    if all(column in frame.columns for column in [*FEATURE_COLUMNS, TARGET_COLUMN]):
        return frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].copy(), FEATURE_COLUMNS, TARGET_COLUMN

    source_target_column = _detect_target_column(frame)
    normalized = frame.rename(columns={source_target_column: TARGET_COLUMN}).copy()
    feature_columns = [column for column in normalized.columns if column != TARGET_COLUMN]
    numeric_feature_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(normalized[column])
    ]
    non_numeric = sorted(set(feature_columns) - set(numeric_feature_columns))
    if non_numeric:
        raise ValueError(f"Uploaded dataset contains non-numeric feature columns: {', '.join(non_numeric)}")
    if not numeric_feature_columns:
        raise ValueError("Uploaded dataset must contain at least one numeric feature column.")
    return normalized[[*numeric_feature_columns, TARGET_COLUMN]], numeric_feature_columns, source_target_column


def _detect_target_column(frame: pd.DataFrame) -> str:
    if TARGET_COLUMN in frame.columns:
        return TARGET_COLUMN
    if "Class" in frame.columns:
        return "Class"
    if "class" in frame.columns:
        return "class"
    raise ValueError(
        "Dataset must include a target column. Supported target names are: target, Class, class."
    )


def _validate_dataset(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")
    if frame[TARGET_COLUMN].value_counts().min() < 5:
        raise ValueError("Each target class must contain at least 5 rows for stratified train/holdout splits.")
    numeric_columns = [*feature_columns, TARGET_COLUMN]
    non_numeric = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError(f"Dataset columns must be numeric: {', '.join(non_numeric)}")
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")


def _store_job(job: TrainingJob) -> None:
    with _jobs_lock:
        _jobs[job.id] = job


def _update_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = _now()


def _job_payload(job: TrainingJob) -> dict:
    return asdict(job)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
