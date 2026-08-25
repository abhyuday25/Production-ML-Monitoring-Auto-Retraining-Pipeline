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

from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from scripts.benchmark_detectors import run_benchmark
from scripts.benchmark_policies import run_policy_benchmark


def prepare_business_dataset(output_dir: Path, rows: int, random_seed: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(random_seed)
    income = rng.normal(70_000, 20_000, rows)
    debt_ratio = rng.beta(2, 5, rows)
    delinquencies = rng.poisson(0.4, rows)
    credit_age = rng.normal(8, 3, rows)
    utilization = rng.beta(2.5, 4, rows)
    recent_inquiries = rng.poisson(1.2, rows)
    loan_amount = rng.normal(22_000, 8_000, rows)
    rate = rng.normal(0.12, 0.03, rows)
    employment_years = rng.normal(6, 4, rows)
    savings_ratio = rng.beta(2, 6, rows)
    logit = (
        -1.2
        - 0.000018 * income
        + 2.8 * debt_ratio
        + 0.45 * delinquencies
        - 0.08 * credit_age
        + 1.6 * utilization
        + 0.2 * recent_inquiries
        + 0.000025 * loan_amount
        + 4.0 * rate
        - 0.04 * employment_years
        - 0.8 * savings_ratio
    )
    probability = 1 / (1 + np.exp(-logit))
    target = (rng.random(rows) < probability).astype(int)
    frame = pd.DataFrame(
        {
            "feature_0": income / 100_000,
            "feature_1": debt_ratio,
            "feature_2": delinquencies,
            "feature_3": credit_age,
            "feature_4": utilization,
            "feature_5": recent_inquiries,
            "feature_6": loan_amount / 50_000,
            "feature_7": rate,
            "feature_8": employment_years,
            "feature_9": savings_ratio,
            TARGET_COLUMN: target,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "credit_reference.csv"
    production_path = output_dir / "credit_production.csv"
    frame.iloc[: rows // 3][FEATURE_COLUMNS + [TARGET_COLUMN]].to_csv(reference_path, index=False)
    frame.iloc[rows // 3 :][FEATURE_COLUMNS + [TARGET_COLUMN]].to_csv(production_path, index=False)
    metadata = {
        "scenario": "synthetic_credit_risk",
        "rows": rows,
        "seed": random_seed,
        "target": "loan default risk",
        "notes": "Business-style secondary scenario generated deterministically; feature columns are mapped to the project model schema.",
    }
    (output_dir / "credit_scenario_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return reference_path, production_path


def run_business_scenario(output_dir: Path, rows: int, drift_start: int, random_seed: int) -> dict:
    reference_path, production_path = prepare_business_dataset(output_dir, rows, random_seed)
    detector_csv = output_dir / "business_drift_benchmark.csv"
    policy_csv = output_dir / "business_policy_benchmark.csv"
    timeseries_csv = output_dir / "business_policy_performance_timeseries.csv"
    detector = run_benchmark(
        reference_path=reference_path,
        production_path=production_path,
        rows=min(700, rows - rows // 3),
        drift_start=drift_start,
        drift_type="sudden",
        drift_strength=1.0,
        window_size=100,
        step_size=25,
        random_seed=random_seed,
        output_csv=detector_csv,
    )
    policy, _ = run_policy_benchmark(
        production_path=production_path,
        rows=min(700, rows - rows // 3),
        drift_start=drift_start,
        drift_type="sudden",
        drift_strength=1.0,
        window_size=100,
        step_size=50,
        random_seed=random_seed,
        output_csv=policy_csv,
        timeseries_csv=timeseries_csv,
    )
    return {
        "reference_path": str(reference_path),
        "production_path": str(production_path),
        "detector_csv": str(detector_csv),
        "policy_csv": str(policy_csv),
        "timeseries_csv": str(timeseries_csv),
        "detector_rows": detector.to_dict(orient="records"),
        "policy_rows": policy.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the secondary credit-risk business scenario.")
    parser.add_argument("--output-dir", default="artifacts/business")
    parser.add_argument("--rows", type=int, default=1800)
    parser.add_argument("--drift-start", type=int, default=350)
    parser.add_argument("--random-seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_business_scenario(Path(args.output_dir), args.rows, args.drift_start, args.random_seed)
    print(json.dumps(result, indent=2))
