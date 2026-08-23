from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from app.drift.schemas import DriftResult

try:
    from river.drift import ADWIN as RiverADWIN
except Exception:  # pragma: no cover - exercised only when optional dependency is absent
    RiverADWIN = None


@dataclass
class ADWINConceptDriftDetector:
    delta: float = 0.002
    min_samples: int = 30

    def __post_init__(self) -> None:
        self._detector = RiverADWIN(delta=self.delta) if RiverADWIN is not None else None
        self._errors: list[float] = []
        self.labeled_samples = 0

    def update(self, prediction: int | None, ground_truth: int | None, stream_index: int | None = None) -> DriftResult | None:
        if prediction is None or ground_truth is None:
            return None
        error = 0.0 if int(prediction) == int(ground_truth) else 1.0
        self.labeled_samples += 1
        self._errors.append(error)
        drift_detected = False
        width = None
        estimation = None
        if self._detector is not None:
            self._detector.update(error)
            drift_detected = bool(self._detector.drift_detected) and self.labeled_samples >= self.min_samples
            width = float(self._detector.width)
            estimation = float(self._detector.estimation)
        elif self.labeled_samples >= max(self.min_samples, 20):
            recent = np.mean(self._errors[-self.min_samples :])
            prior = np.mean(self._errors[: -self.min_samples]) if len(self._errors) > self.min_samples else recent
            drift_detected = bool(recent - prior > 0.35)
            width = float(len(self._errors))
            estimation = float(recent)

        if not drift_detected:
            return None
        return DriftResult(
            "adwin",
            estimation,
            self.delta,
            True,
            self.labeled_samples,
            metadata={"stream_index": stream_index, "width": width, "error_rate": estimation},
        )

    def process(self, rows: Iterable[dict]) -> list[DriftResult]:
        events: list[DriftResult] = []
        for idx, row in enumerate(rows):
            event = self.update(row.get("prediction"), row.get("ground_truth"), row.get("stream_index", idx))
            if event is not None:
                events.append(event)
        if events:
            return events
        return [
            DriftResult(
                "adwin",
                None,
                self.delta,
                False,
                self.labeled_samples,
                status="insufficient_data" if self.labeled_samples < self.min_samples else "ok",
                metadata={"labeled_samples": self.labeled_samples},
            )
        ]
