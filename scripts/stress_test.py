from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests

from app.ml.features import FEATURE_COLUMNS


def run_stress_test(api_url: str, requests_count: int, timeout: float) -> dict:
    endpoint = api_url.rstrip("/") + "/predict"
    payload = {"features": {column: (idx + 1) / 10 for idx, column in enumerate(FEATURE_COLUMNS)}, "ground_truth": 1}
    latencies = []
    successes = 0
    failures = 0
    started = time.perf_counter()
    for _ in range(requests_count):
        request_started = time.perf_counter()
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            successes += 1
        except requests.RequestException:
            failures += 1
        latencies.append((time.perf_counter() - request_started) * 1000)
    duration = time.perf_counter() - started
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)
    result = {
        "requests": requests_count,
        "successes": successes,
        "failures": failures,
        "throughput_rps": successes / duration if duration > 0 else 0.0,
        "average_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": p95,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight API stress test.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output-json", default="artifacts/stress_test.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run_stress_test(args.api_url, args.requests, args.timeout)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
