from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.drift.schemas import DriftResult


def calculate_ks(
    reference_values: pd.Series | np.ndarray,
    production_values: pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
    min_samples: int = 30,
    feature: str | None = None,
) -> DriftResult:
    reference = _clean_numeric(reference_values)
    production = _clean_numeric(production_values)
    sample_count = int(production.size)
    if reference.size < min_samples or production.size < min_samples:
        return DriftResult("ks", None, alpha, False, sample_count, feature, status="insufficient_data")
    if np.unique(reference).size == 1 and np.unique(production).size == 1 and reference[0] == production[0]:
        return DriftResult("ks", 0.0, alpha, False, sample_count, feature, metadata={"p_value": 1.0})

    statistic, p_value = ks_2samp(reference, production, alternative="two-sided", mode="auto")
    return DriftResult(
        "ks",
        float(statistic),
        alpha,
        bool(p_value < alpha),
        sample_count,
        feature,
        metadata={"p_value": float(p_value), "ks_statistic": float(statistic)},
    )


def run_ks(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float,
    min_samples: int,
) -> list[DriftResult]:
    results: list[DriftResult] = []
    for feature in feature_columns:
        if feature not in reference.columns or feature not in production.columns:
            results.append(DriftResult("ks", None, alpha, False, 0, feature, status="missing_feature"))
            continue
        results.append(calculate_ks(reference[feature], production[feature], alpha=alpha, min_samples=min_samples, feature=feature))
    return results


def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return series.to_numpy(dtype=float)
