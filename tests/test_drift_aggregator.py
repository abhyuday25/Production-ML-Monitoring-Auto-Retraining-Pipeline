from app.core.config import Settings
from app.drift.aggregator import aggregate_drift_results
from app.drift.schemas import DriftResult


def test_no_detector_triggered_is_none():
    assessment = aggregate_drift_results([DriftResult("psi", 0.01, 0.2, False, 100, "feature_1")], Settings(), window_size=100)

    assert assessment.severity == "NONE"
    assert not assessment.triggered_detectors


def test_one_moderate_detector_has_expected_trigger():
    assessment = aggregate_drift_results([DriftResult("psi", 0.25, 0.2, True, 100, "feature_1")], Settings(), window_size=100)

    assert assessment.drift_detected
    assert assessment.severity in {"LOW", "MEDIUM"}
    assert assessment.triggered_detectors == ["psi"]
    assert assessment.top_drifting_features == ["feature_1"]


def test_multiple_strong_detectors_raise_severity():
    settings = Settings(drift_high_threshold=0.6)
    results = [
        DriftResult("psi", 0.8, 0.2, True, 100, "feature_2"),
        DriftResult("ks", 0.9, 0.05, True, 100, "feature_2"),
        DriftResult("adwin", 0.7, 0.002, True, 100),
        DriftResult("confidence", 0.3, 0.1, True, 100),
    ]

    assessment = aggregate_drift_results(results, settings, window_size=100)

    assert assessment.severity in {"HIGH", "CRITICAL"}
    assert set(assessment.triggered_detectors) == {"psi", "ks", "adwin", "confidence"}
