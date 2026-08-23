from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import requests

from app.core.logging import configure_logging
from app.ml.features import FEATURE_COLUMNS
from app.ml.features import TARGET_COLUMN

logger = logging.getLogger(__name__)


def build_payload(row: pd.Series, feature_columns: list[str], include_ground_truth: bool) -> dict:
    payload = {"features": {column: float(row[column]) for column in feature_columns}}
    if include_ground_truth:
        payload["ground_truth"] = int(row[TARGET_COLUMN])
    return payload


def inject_drift(
    frame: pd.DataFrame,
    *,
    drift_start: int | None,
    drift_type: str,
    drift_strength: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict | None]:
    if drift_start is None or drift_type == "none":
        return frame, None
    drifted = frame.copy()
    shuffled_index = pd.Series(range(len(drifted))).sample(frac=1.0, random_state=random_seed).to_numpy()
    affected_features = [column for column in FEATURE_COLUMNS[:3] if column in drifted.columns]
    if drift_start >= len(drifted):
        raise ValueError("--drift-start must be smaller than the number of streamed rows")
    drift_slice = drifted.index[drift_start:]
    if drift_type == "sudden":
        for column in affected_features:
            drifted.loc[drift_slice, column] = drifted.loc[drift_slice, column] + drift_strength
    elif drift_type == "gradual":
        ramp = pd.Series(range(len(drift_slice)), index=drift_slice, dtype=float) / max(len(drift_slice) - 1, 1)
        for column in affected_features:
            drifted.loc[drift_slice, column] = drifted.loc[drift_slice, column] + drift_strength * ramp
    else:
        raise ValueError(f"Unsupported drift type: {drift_type}")
    metadata = {
        "drift_type": drift_type,
        "drift_start_index": drift_start,
        "drift_end_index": int(len(drifted) - 1),
        "affected_features": affected_features,
        "drift_strength": drift_strength,
        "seed": random_seed,
        "shuffled_index_checksum": int(shuffled_index[:10].sum()),
    }
    return drifted, metadata


def stream_rows(
    api_url: str,
    production_path: Path,
    limit: int | None,
    delay: float,
    include_ground_truth: bool,
    *,
    drift_start: int | None = None,
    drift_type: str = "none",
    drift_strength: float = 1.5,
    random_seed: int = 42,
    metadata_path: Path | None = None,
) -> None:
    frame = pd.read_csv(production_path)
    if limit is not None:
        frame = frame.head(limit)
    frame, drift_metadata = inject_drift(
        frame,
        drift_start=drift_start,
        drift_type=drift_type,
        drift_strength=drift_strength,
        random_seed=random_seed,
    )
    if drift_metadata is not None:
        metadata_path = metadata_path or Path("artifacts") / "synthetic_drift_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(drift_metadata, indent=2), encoding="utf-8")
        logger.info("Wrote drift metadata to %s", metadata_path)
    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    endpoint = api_url.rstrip("/") + "/predict"

    successes = 0
    failures = 0
    for idx, row in frame.iterrows():
        payload = build_payload(row, feature_columns, include_ground_truth)
        try:
            response = requests.post(endpoint, json=payload, timeout=10)
            response.raise_for_status()
            successes += 1
            print(f"{idx + 1}/{len(frame)} request_id={response.json().get('request_id')} prediction={response.json().get('prediction')}")
        except requests.RequestException as exc:
            failures += 1
            logger.error("Simulation request failed at row %s: %s", idx, exc)
        if delay > 0:
            time.sleep(delay)
    print(f"Simulation complete successes={successes} failures={failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream production-like records to the prediction API.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base API URL.")
    parser.add_argument("--production-path", default="data/production/production.csv", help="Production CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to send.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests in seconds.")
    parser.add_argument("--include-ground-truth", action="store_true", help="Include labels in prediction requests.")
    parser.add_argument("--drift-start", type=int, default=None, help="Zero-based row index where synthetic drift begins.")
    parser.add_argument("--drift-type", choices=["none", "sudden", "gradual"], default="none", help="Synthetic drift pattern.")
    parser.add_argument("--drift-strength", type=float, default=1.5, help="Feature shift size after drift starts.")
    parser.add_argument("--random-seed", type=int, default=42, help="Deterministic seed for drift metadata.")
    parser.add_argument("--metadata-path", default="artifacts/synthetic_drift_metadata.json", help="Where to write known drift metadata.")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    stream_rows(
        args.api_url,
        Path(args.production_path),
        args.limit,
        args.delay,
        args.include_ground_truth,
        drift_start=args.drift_start,
        drift_type=args.drift_type,
        drift_strength=args.drift_strength,
        random_seed=args.random_seed,
        metadata_path=Path(args.metadata_path),
    )
