from __future__ import annotations

import numpy as np
import pandas as pd

from app.drift.schemas import DriftResult


def binary_confidence(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    probs = pd.to_numeric(pd.Series(probabilities), errors="coerce").dropna().clip(0.0, 1.0).to_numpy(dtype=float)
    return np.maximum(probs, 1.0 - probs)


def binary_entropy(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    probs = pd.to_numeric(pd.Series(probabilities), errors="coerce").dropna().clip(1e-12, 1.0 - 1e-12).to_numpy(dtype=float)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def calculate_confidence_drift(
    reference_probabilities: pd.Series | np.ndarray,
    production_probabilities: pd.Series | np.ndarray,
    *,
    threshold: float = 0.10,
    min_samples: int = 30,
    method: str = "mean_confidence",
) -> DriftResult:
    reference_conf = binary_confidence(reference_probabilities)
    production_conf = binary_confidence(production_probabilities)
    sample_count = int(production_conf.size)
    if reference_conf.size < min_samples or production_conf.size < min_samples:
        return DriftResult("confidence", None, threshold, False, sample_count, status="insufficient_data")

    reference_value = float(np.mean(reference_conf))
    production_value = float(np.mean(production_conf))
    score = abs(production_value - reference_value)
    return DriftResult(
        "confidence",
        score,
        threshold,
        score >= threshold,
        sample_count,
        metadata={
            "method": method,
            "reference_value": reference_value,
            "production_value": production_value,
        },
    )
