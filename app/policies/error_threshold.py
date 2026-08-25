from __future__ import annotations

import numpy as np

from app.policies.schemas import PolicyContext, RetrainingDecisionResult


class ErrorThresholdPolicy:
    name = "error_threshold"

    def __init__(self, window_size: int, threshold: float, min_samples: int) -> None:
        if window_size <= 0:
            raise ValueError("error window size must be positive")
        if min_samples <= 0:
            raise ValueError("minimum error samples must be positive")
        self.window_size = window_size
        self.threshold = threshold
        self.min_samples = min_samples

    def evaluate(self, context: PolicyContext) -> RetrainingDecisionResult:
        window = context.errors[-self.window_size :]
        samples = len(window)
        if samples < self.min_samples:
            return RetrainingDecisionResult(
                self.name,
                False,
                "insufficient labeled samples",
                {"samples": samples, "required_samples": self.min_samples},
                observation_index=context.observation_index,
            )
        rolling_error = float(np.mean(window))
        should = rolling_error >= self.threshold
        return RetrainingDecisionResult(
            self.name,
            should,
            "rolling error exceeded threshold" if should else "rolling error below threshold",
            {"rolling_error": rolling_error, "threshold": self.threshold, "samples": samples},
            observation_index=context.observation_index,
        )
