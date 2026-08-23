from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.database import configure_database, create_tables
from app.drift.service import run_drift_monitoring


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 2 drift detection on prediction logs.")
    parser.add_argument("--no-persist", action="store_true", help="Compute without writing drift tables.")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    session_factory = configure_database(settings.database_url)
    create_tables()
    with session_factory() as session:
        results, assessment = run_drift_monitoring(settings, session, persist=not args.no_persist)
    print(
        json.dumps(
            {
                "overall_score": assessment.overall_score,
                "severity": assessment.severity,
                "drift_detected": assessment.drift_detected,
                "triggered_detectors": assessment.triggered_detectors,
                "top_drifting_features": assessment.top_drifting_features,
                "window_size": assessment.window_size,
                "detector_results": [
                    {
                        "detector": result.detector,
                        "feature": result.feature,
                        "score": result.score,
                        "threshold": result.threshold,
                        "drift_detected": result.drift_detected,
                        "sample_count": result.sample_count,
                        "status": result.status,
                    }
                    for result in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
