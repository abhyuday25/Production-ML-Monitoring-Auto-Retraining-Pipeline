from __future__ import annotations

import numpy as np
import pandas as pd

from app.drift.schemas import DriftResult


def calculate_psi(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    *,
    bins: int = 10,
    threshold: float = 0.20,
    min_samples: int = 30,
    feature: str | None = None,
) -> DriftResult:
    expected_values = _clean_numeric(expected)
    actual_values = _clean_numeric(actual)
    sample_count = int(actual_values.size)
    if expected_values.size < min_samples or actual_values.size < min_samples:
        return DriftResult("psi", None, threshold, False, sample_count, feature, status="insufficient_data")

    edges = _reference_bin_edges(expected_values, bins)
    if edges.size < 2:
        return DriftResult("psi", 0.0, threshold, False, sample_count, feature, metadata={"reason": "constant_reference"})

    expected_counts, _ = np.histogram(expected_values, bins=edges)
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    epsilon = 1e-6
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), epsilon)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), epsilon)
    score = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return DriftResult("psi", score, threshold, score >= threshold, sample_count, feature, metadata={"bins": int(edges.size - 1)})


def run_psi(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    feature_columns: list[str],
    *,
    bins: int,
    threshold: float,
    min_samples: int,
) -> list[DriftResult]:
    results: list[DriftResult] = []
    for feature in feature_columns:
        if feature not in reference.columns or feature not in production.columns:
            results.append(DriftResult("psi", None, threshold, False, 0, feature, status="missing_feature"))
            continue
        results.append(calculate_psi(reference[feature], production[feature], bins=bins, threshold=threshold, min_samples=min_samples, feature=feature))
    return results


def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return series.to_numpy(dtype=float)


def _reference_bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size < 2:
        return edges
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges
