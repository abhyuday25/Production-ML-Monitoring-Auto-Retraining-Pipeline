from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.db.models import DriftAlert, DriftMeasurement
from app.drift.schemas import DriftAssessment, DriftResult
from app.observability.metrics import record_drift_results


def persist_drift_results(session: Session, results: list[DriftResult], assessment: DriftAssessment) -> DriftAlert:
    for result in results:
        session.add(
            DriftMeasurement(
                window_start=result.window_start,
                window_end=result.window_end,
                detector=result.detector,
                feature=result.feature,
                score=result.score,
                threshold=result.threshold,
                drift_detected=int(result.drift_detected),
                sample_count=result.sample_count,
                status=result.status,
                metadata_json=json.dumps(result.metadata, sort_keys=True),
            )
        )
    alert = DriftAlert(
        window_start=assessment.window_start,
        window_end=assessment.window_end,
        overall_score=assessment.overall_score,
        severity=assessment.severity,
        drift_detected=int(assessment.drift_detected),
        triggered_detectors=json.dumps(assessment.triggered_detectors),
        top_drifting_features=json.dumps(assessment.top_drifting_features),
        window_size=assessment.window_size,
        metadata_json=json.dumps(assessment.metadata, sort_keys=True),
    )
    session.add(alert)
    session.commit()
    session.refresh(alert)
    record_drift_results(results, assessment)
    return alert
