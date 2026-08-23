from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.drift.adwin import ADWINConceptDriftDetector
from app.drift.aggregator import aggregate_drift_results
from app.drift.confidence import calculate_confidence_drift
from app.drift.ks import run_ks
from app.drift.psi import run_psi
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from scripts.simulate_production import inject_drift


def run_benchmark(
    *,
    reference_path: Path,
    production_path: Path,
    rows: int,
    drift_start: int,
    drift_type: str,
    drift_strength: float,
    window_size: int,
    step_size: int,
    random_seed: int,
    output_csv: Path,
) -> pd.DataFrame:
    settings = get_settings()
    reference = pd.read_csv(reference_path)
    production = pd.read_csv(production_path).head(rows)
    stream, metadata = inject_drift(
        production,
        drift_start=drift_start,
        drift_type=drift_type,
        drift_strength=drift_strength,
        random_seed=random_seed,
    )
    if metadata is None:
        raise ValueError("Benchmark requires drift_type other than none")
    metadata_path = output_csv.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    reference_probs = _synthetic_probabilities(reference, random_seed)
    stream_probs = _synthetic_probabilities(stream, random_seed, drift_start=drift_start)
    stream_predictions = _synthetic_predictions(stream, drift_start=drift_start, random_seed=random_seed)
    detector_alerts: dict[str, list[int]] = {"psi": [], "ks": [], "adwin": [], "confidence": [], "ensemble": []}

    for end in range(window_size, len(stream) + 1, step_size):
        window = stream.iloc[end - window_size : end]
        alert_index = end - 1
        psi_results = run_psi(
            reference,
            window,
            FEATURE_COLUMNS,
            bins=settings.psi_bins,
            threshold=settings.psi_drift_threshold,
            min_samples=min(settings.min_drift_samples, window_size),
        )
        ks_results = run_ks(
            reference,
            window,
            FEATURE_COLUMNS,
            alpha=settings.ks_alpha,
            min_samples=min(settings.min_drift_samples, window_size),
        )
        confidence_result = calculate_confidence_drift(
            reference_probs,
            stream_probs[end - window_size : end],
            threshold=settings.confidence_drift_threshold,
            min_samples=min(settings.min_drift_samples, window_size),
        )
        if any(result.drift_detected for result in psi_results):
            detector_alerts["psi"].append(alert_index)
        if any(result.drift_detected for result in ks_results):
            detector_alerts["ks"].append(alert_index)
        if confidence_result.drift_detected:
            detector_alerts["confidence"].append(alert_index)
        assessment = aggregate_drift_results(psi_results + ks_results + [confidence_result], settings, window_size=window_size)
        if assessment.drift_detected:
            detector_alerts["ensemble"].append(alert_index)

    adwin = ADWINConceptDriftDetector(delta=settings.adwin_delta, min_samples=settings.min_adwin_samples)
    for idx, (prediction, truth) in enumerate(zip(stream_predictions, stream[TARGET_COLUMN].astype(int), strict=False)):
        event = adwin.update(int(prediction), int(truth), idx)
        if event is not None:
            detector_alerts["adwin"].append(idx)

    rows_out = [
        evaluate_alerts(detector, alerts, drift_start=drift_start, rows=rows, tolerance=window_size)
        | {"scenario": "synthetic_stream", "drift_type": drift_type}
        for detector, alerts in detector_alerts.items()
    ]
    result = pd.DataFrame(rows_out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def evaluate_alerts(detector: str, alerts: list[int], *, drift_start: int, rows: int, tolerance: int) -> dict:
    deduped = _dedupe(alerts, cooldown=max(1, tolerance // 2))
    false_positives = [alert for alert in deduped if alert < drift_start - tolerance]
    true_alerts = [alert for alert in deduped if alert >= drift_start]
    first_detection = min(true_alerts) if true_alerts else None
    latency = None if first_detection is None else int(first_detection - drift_start)
    true_positives = 1 if first_detection is not None else 0
    false_negatives = 0 if true_positives else 1
    pre_drift_windows = max(1, drift_start // max(1, tolerance))
    return {
        "detector": detector,
        "drift_start": drift_start,
        "detections": json.dumps(deduped),
        "true_positives": true_positives,
        "false_positives": len(false_positives),
        "false_negatives": false_negatives,
        "detection_rate": float(true_positives),
        "false_alarm_rate": float(len(false_positives) / pre_drift_windows),
        "detection_latency": latency,
    }


def _dedupe(alerts: list[int], *, cooldown: int) -> list[int]:
    deduped: list[int] = []
    for alert in sorted(alerts):
        if not deduped or alert - deduped[-1] >= cooldown:
            deduped.append(int(alert))
    return deduped


def _synthetic_probabilities(frame: pd.DataFrame, random_seed: int, drift_start: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    base = np.where(frame[TARGET_COLUMN].to_numpy(dtype=int) == 1, 0.82, 0.18)
    probs = np.clip(base + rng.normal(0, 0.04, size=len(frame)), 0.01, 0.99)
    if drift_start is not None:
        probs[drift_start:] = np.clip(0.5 + rng.normal(0, 0.05, size=len(frame) - drift_start), 0.01, 0.99)
    return probs


def _synthetic_predictions(frame: pd.DataFrame, drift_start: int, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    truth = frame[TARGET_COLUMN].to_numpy(dtype=int)
    predictions = truth.copy()
    before_flips = rng.random(len(frame)) < 0.05
    after_flips = rng.random(len(frame)) < 0.65
    flips = before_flips
    flips[drift_start:] = after_flips[drift_start:]
    predictions[flips] = 1 - predictions[flips]
    return predictions


def print_table(result: pd.DataFrame) -> None:
    print("Detector | Detection Rate | False Alarm Rate | Detection Latency")
    print("--- | ---: | ---: | ---:")
    for _, row in result.iterrows():
        latency = "" if pd.isna(row["detection_latency"]) else int(row["detection_latency"])
        print(f"{row['detector']} | {row['detection_rate']:.2f} | {row['false_alarm_rate']:.2f} | {latency}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Week 2 drift detectors against known synthetic drift.")
    parser.add_argument("--reference-path", default="data/reference/reference.csv")
    parser.add_argument("--production-path", default="data/production/production.csv")
    parser.add_argument("--rows", type=int, default=750)
    parser.add_argument("--drift-start", type=int, default=350)
    parser.add_argument("--drift-type", choices=["sudden", "gradual"], default="sudden")
    parser.add_argument("--drift-strength", type=float, default=1.5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-csv", default="artifacts/drift_benchmark.csv")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    table = run_benchmark(
        reference_path=Path(args.reference_path),
        production_path=Path(args.production_path),
        rows=args.rows,
        drift_start=args.drift_start,
        drift_type=args.drift_type,
        drift_strength=args.drift_strength,
        window_size=args.window_size,
        step_size=args.step_size,
        random_seed=args.random_seed,
        output_csv=Path(args.output_csv),
    )
    print_table(table)
