from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import PredictionLog
from app.drift.adwin import ADWINConceptDriftDetector
from app.drift.aggregator import aggregate_drift_results
from app.drift.confidence import calculate_confidence_drift
from app.drift.ks import run_ks
from app.drift.persistence import persist_drift_results
from app.drift.psi import run_psi
from app.drift.schemas import DriftAssessment, DriftResult
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from app.ml.model_loader import load_model

logger = logging.getLogger(__name__)


def run_drift_monitoring(settings: Settings, session: Session, *, persist: bool = True) -> tuple[list[DriftResult], DriftAssessment]:
    logger.info("Drift monitoring started")
    reference = load_reference_frame(settings)
    production, rows = load_latest_prediction_window(session, settings.drift_window_size)
    window_size = len(production)
    window_start = rows[0].timestamp if rows else None
    window_end = rows[-1].timestamp if rows else None

    if window_size < settings.min_drift_samples:
        result = DriftResult(
            "ensemble",
            None,
            None,
            False,
            window_size,
            window_start=window_start,
            window_end=window_end,
            status="insufficient_data",
            metadata={"required_samples": settings.min_drift_samples},
        )
        assessment = aggregate_drift_results([result], settings, window_size=window_size)
        if persist:
            persist_drift_results(session, [result], assessment)
        logger.info("Drift monitoring skipped insufficient samples=%s", window_size)
        return [result], assessment

    feature_columns = _feature_columns(reference, production)
    results = run_psi(
        reference,
        production,
        feature_columns,
        bins=settings.psi_bins,
        threshold=settings.psi_drift_threshold,
        min_samples=settings.min_drift_samples,
    )
    results.extend(run_ks(reference, production, feature_columns, alpha=settings.ks_alpha, min_samples=settings.min_drift_samples))
    results.append(_confidence_result(settings, reference, production))
    results.extend(_adwin_results(settings, rows))
    results = [_with_window(result, window_start, window_end) for result in results]
    assessment = aggregate_drift_results(results, settings, window_size=window_size)
    if persist:
        persist_drift_results(session, results, assessment)
    logger.info("Drift monitoring completed severity=%s score=%.3f", assessment.severity, assessment.overall_score)
    return results, assessment


def load_reference_frame(settings: Settings) -> pd.DataFrame:
    path = Path(settings.data_dir) / "reference" / "reference.csv"
    if not path.exists():
        raise FileNotFoundError(f"Reference dataset not found at {path}")
    return pd.read_csv(path)


def load_latest_prediction_window(session: Session, window_size: int) -> tuple[pd.DataFrame, list[PredictionLog]]:
    rows = list(session.query(PredictionLog).order_by(desc(PredictionLog.id)).limit(window_size))
    rows.reverse()
    records = []
    for row in rows:
        features = json.loads(row.input_features)
        record = {name: features.get(name) for name in FEATURE_COLUMNS}
        record["prediction"] = row.prediction
        record["prediction_probability"] = row.prediction_probability
        record["ground_truth"] = row.ground_truth
        record["timestamp"] = row.timestamp
        records.append(record)
    return pd.DataFrame(records), rows


def _feature_columns(reference: pd.DataFrame, production: pd.DataFrame) -> list[str]:
    return [column for column in FEATURE_COLUMNS if column in reference.columns and column in production.columns]


def _confidence_result(settings: Settings, reference: pd.DataFrame, production: pd.DataFrame) -> DriftResult:
    if "prediction_probability" not in production.columns:
        return DriftResult("confidence", None, settings.confidence_drift_threshold, False, len(production), status="missing_probability")
    try:
        bundle = load_model(settings)
        probabilities = bundle.model.predict_proba(reference[bundle.feature_columns])
        reference_probs = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities.max(axis=1)
    except Exception as exc:
        logger.warning("Reference confidence unavailable: %s", exc)
        return DriftResult(
            "confidence",
            None,
            settings.confidence_drift_threshold,
            False,
            int(production["prediction_probability"].notna().sum()),
            status="reference_confidence_unavailable",
        )
    return calculate_confidence_drift(
        reference_probs,
        production["prediction_probability"],
        threshold=settings.confidence_drift_threshold,
        min_samples=settings.min_drift_samples,
    )


def _adwin_results(settings: Settings, rows: list[PredictionLog]) -> list[DriftResult]:
    detector = ADWINConceptDriftDetector(delta=settings.adwin_delta, min_samples=settings.min_adwin_samples)
    stream = [
        {
            "prediction": row.prediction,
            "ground_truth": row.ground_truth,
            "stream_index": offset,
        }
        for offset, row in enumerate(rows)
    ]
    return detector.process(stream)


def _with_window(result: DriftResult, window_start, window_end) -> DriftResult:
    return DriftResult(
        detector=result.detector,
        score=result.score,
        threshold=result.threshold,
        drift_detected=result.drift_detected,
        sample_count=result.sample_count,
        feature=result.feature,
        window_start=window_start,
        window_end=window_end,
        status=result.status,
        metadata=result.metadata,
    )
