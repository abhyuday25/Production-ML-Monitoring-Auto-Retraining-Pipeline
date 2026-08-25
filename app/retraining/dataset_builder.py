from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import PredictionLog
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN


def build_labeled_production_frame(session: Session, settings: Settings) -> pd.DataFrame:
    rows = (
        session.query(PredictionLog)
        .filter(PredictionLog.ground_truth.is_not(None))
        .order_by(desc(PredictionLog.id))
        .limit(settings.max_production_retrain_samples)
        .all()
    )
    rows.reverse()
    records = []
    for row in rows:
        features = json.loads(row.input_features)
        if not all(column in features for column in FEATURE_COLUMNS):
            continue
        record = {column: float(features[column]) for column in FEATURE_COLUMNS}
        record[TARGET_COLUMN] = int(row.ground_truth)
        records.append(record)
    frame = pd.DataFrame(records)
    if len(frame) < settings.min_retrain_samples:
        raise ValueError(f"insufficient labeled production samples: {len(frame)} < {settings.min_retrain_samples}")
    return frame


def build_retraining_frame(session: Session, settings: Settings) -> pd.DataFrame:
    train_path = Path(settings.data_dir) / "processed" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found at {train_path}")
    base = pd.read_csv(train_path)
    production = build_labeled_production_frame(session, settings)
    combined = pd.concat([base[FEATURE_COLUMNS + [TARGET_COLUMN]], production[FEATURE_COLUMNS + [TARGET_COLUMN]]], ignore_index=True)
    return combined
