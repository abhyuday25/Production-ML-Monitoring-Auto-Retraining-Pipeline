import numpy as np

from app.drift.confidence import calculate_confidence_drift


def test_stable_confidence_has_low_drift():
    reference = np.array([0.8, 0.82, 0.78, 0.81] * 30)
    production = np.array([0.79, 0.83, 0.77, 0.82] * 30)

    result = calculate_confidence_drift(reference, production, threshold=0.1, min_samples=30)

    assert result.score < 0.1
    assert not result.drift_detected


def test_lower_confidence_triggers_drift():
    reference = np.array([0.9, 0.88, 0.92, 0.91] * 30)
    production = np.array([0.51, 0.52, 0.48, 0.49] * 30)

    result = calculate_confidence_drift(reference, production, threshold=0.1, min_samples=30)

    assert result.drift_detected
    assert result.metadata["production_value"] < result.metadata["reference_value"]
