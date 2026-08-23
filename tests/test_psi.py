import numpy as np
import pandas as pd

from app.drift.psi import calculate_psi


def test_identical_distributions_have_low_psi():
    values = pd.Series(np.linspace(-1, 1, 200))
    result = calculate_psi(values, values.copy(), threshold=0.2, min_samples=30)

    assert result.status == "ok"
    assert result.score < 0.01
    assert not result.drift_detected


def test_shifted_distribution_has_higher_psi():
    reference = pd.Series(np.random.default_rng(42).normal(0, 1, 300))
    production = pd.Series(np.random.default_rng(43).normal(2, 1, 300))
    result = calculate_psi(reference, production, threshold=0.2, min_samples=30)

    assert result.score > 0.2
    assert result.drift_detected


def test_constant_reference_does_not_crash():
    result = calculate_psi(pd.Series([1.0] * 100), pd.Series([1.0] * 100), min_samples=30)

    assert result.score == 0.0
    assert not result.drift_detected


def test_insufficient_samples_are_structured():
    result = calculate_psi(pd.Series([1.0, 2.0]), pd.Series([1.0]), min_samples=10)

    assert result.status == "insufficient_data"
    assert result.score is None
