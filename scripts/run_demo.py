from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.benchmark_detectors import run_benchmark
from scripts.benchmark_policies import run_policy_benchmark
from scripts.run_business_scenario import run_business_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible local demo artifacts without requiring Docker services.")
    parser.add_argument("--scenario", choices=["synthetic", "business", "all"], default="all")
    args = parser.parse_args()
    output: dict = {}
    if args.scenario in {"synthetic", "all"}:
        detector = run_benchmark(
            reference_path=Path("data/reference/reference.csv"),
            production_path=Path("data/production/production.csv"),
            rows=750,
            drift_start=350,
            drift_type="sudden",
            drift_strength=1.5,
            window_size=100,
            step_size=25,
            random_seed=42,
            output_csv=Path("artifacts/drift_benchmark.csv"),
        )
        policy, _ = run_policy_benchmark(
            production_path=Path("data/production/production.csv"),
            rows=750,
            drift_start=350,
            drift_type="sudden",
            drift_strength=1.5,
            window_size=100,
            step_size=50,
            random_seed=42,
            output_csv=Path("artifacts/policy_benchmark.csv"),
            timeseries_csv=Path("artifacts/policy_performance_timeseries.csv"),
        )
        output["synthetic"] = {"detectors": detector.to_dict(orient="records"), "policies": policy.to_dict(orient="records")}
    if args.scenario in {"business", "all"}:
        output["business"] = run_business_scenario(Path("artifacts/business"), 1800, 350, 123)
    Path("artifacts/results").mkdir(parents=True, exist_ok=True)
    Path("artifacts/results/demo_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
