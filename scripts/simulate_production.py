from __future__ import annotations

import argparse
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
from app.ml.features import TARGET_COLUMN

logger = logging.getLogger(__name__)


def build_payload(row: pd.Series, feature_columns: list[str], include_ground_truth: bool) -> dict:
    payload = {"features": {column: float(row[column]) for column in feature_columns}}
    if include_ground_truth:
        payload["ground_truth"] = int(row[TARGET_COLUMN])
    return payload


def stream_rows(api_url: str, production_path: Path, limit: int | None, delay: float, include_ground_truth: bool) -> None:
    frame = pd.read_csv(production_path)
    if limit is not None:
        frame = frame.head(limit)
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
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    stream_rows(args.api_url, Path(args.production_path), args.limit, args.delay, args.include_ground_truth)
