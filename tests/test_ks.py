import numpy as np
import pandas as pd

from app.drift.ks import calculate_ks


def test_similar_samples_do_not_trigger_ks():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 300))
    production = pd.Series(rng.normal(0, 1, 300))

    result = calculate_ks(reference, production, alpha=0.001, min_samples=30)

    assert "p_value" in result.metadata
    assert result.score is not None
    assert not result.drift_detected


def test_shifted_samples_trigger_ks():
    reference = pd.Series(np.random.default_rng(42).normal(0, 1, 300))
    production = pd.Series(np.random.default_rng(43).normal(2, 1, 300))

    result = calculate_ks(reference, production, alpha=0.05, min_samples=30)

    assert result.drift_detected
    assert result.metadata["p_value"] < 0.05


def test_empty_inputs_are_handled():
    result = calculate_ks(pd.Series([], dtype=float), pd.Series([], dtype=float), min_samples=10)

    assert result.status == "insufficient_data"
    assert result.score is None
