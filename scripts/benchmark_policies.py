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
from sklearn.metrics import accuracy_score, f1_score

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.policies.cost_aware import evaluate_cost_gate
from app.policies.engine import PolicyEngine
from app.policies.schemas import PolicyContext
from scripts.simulate_production import inject_drift


POLICIES = ["periodic", "error_threshold", "drift_triggered"]


def run_policy_benchmark(
    *,
    production_path: Path,
    rows: int,
    drift_start: int,
    drift_type: str,
    drift_strength: float,
    window_size: int,
    step_size: int,
    random_seed: int,
    output_csv: Path,
    timeseries_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    stream = pd.read_csv(production_path).head(rows)
    stream, metadata = inject_drift(stream, drift_start=drift_start, drift_type=drift_type, drift_strength=drift_strength, random_seed=random_seed)
    if metadata is None:
        raise ValueError("Policy benchmark requires a drift scenario")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_results = []
    all_timeseries = []
    for policy_name in POLICIES:
        summary, series = _simulate_policy(policy_name, stream, settings, drift_start, window_size, step_size, random_seed)
        all_results.append(summary)
        all_timeseries.extend(series)
    result_frame = pd.DataFrame(all_results)
    timeseries_frame = pd.DataFrame(all_timeseries)
    result_frame.to_csv(output_csv, index=False)
    timeseries_frame.to_csv(timeseries_csv, index=False)
    return result_frame, timeseries_frame


def _simulate_policy(
    policy_name: str,
    stream: pd.DataFrame,
    settings: Settings,
    drift_start: int,
    window_size: int,
    step_size: int,
    random_seed: int,
) -> tuple[dict, list[dict]]:
    engine = PolicyEngine(settings)
    truth = stream["target"].astype(int).to_numpy()
    champion_predictions = _predictions(truth, drift_start, before_error=0.08, after_error=0.34, random_seed=random_seed)
    adapted_predictions = _predictions(truth, drift_start, before_error=0.08, after_error=0.16, random_seed=random_seed + 10)
    active_predictions = champion_predictions.copy()
    errors: list[int] = []
    last_retrain_index = None
    last_trigger_index = None
    proposals = approvals = cost_rejections = retrain_count = promotions = rollbacks = 0
    retraining_cost = drift_cost = 0.0
    series = []

    for end in range(step_size, len(stream) + 1, step_size):
        start = max(0, end - window_size)
        window_truth = truth[start:end]
        window_predictions = active_predictions[start:end]
        window_errors = [0 if pred == actual else 1 for pred, actual in zip(window_predictions, window_truth, strict=False)]
        errors.extend(window_errors)
        rolling_error = float(np.mean(errors[-settings.error_window_size :])) if errors else 0.0
        latest_drift = _drift_signal(end, drift_start)
        context = PolicyContext(
            observation_index=end,
            last_retrain_index=last_retrain_index,
            last_policy_trigger_index=last_trigger_index,
            errors=errors,
            latest_drift=latest_drift,
            current_model_version="benchmark-active",
        )
        decision = engine.evaluate(policy_name, context)
        if decision.should_retrain:
            proposals += 1
            last_trigger_index = end
        cost = evaluate_cost_gate(
            decision,
            settings,
            current_error=rolling_error,
            baseline_error=0.10,
            training_sample_count=3000 + end,
        )
        drift_cost += cost.estimated_drift_cost
        if decision.should_retrain and cost.approved:
            approvals += 1
            retrain_count += 1
            promotions += 1
            retraining_cost += cost.estimated_retraining_cost
            last_retrain_index = end
            active_predictions[end:] = adapted_predictions[end:]
        elif decision.should_retrain:
            cost_rejections += 1

        acc = float(accuracy_score(window_truth, active_predictions[start:end]))
        f1 = float(f1_score(window_truth, active_predictions[start:end], zero_division=0))
        series.append({"policy": policy_name, "window_end": end, "active_model_version": f"{policy_name}-sim", "accuracy": acc, "f1": f1})

    final_truth = truth[-window_size:]
    final_predictions = active_predictions[-window_size:]
    final_accuracy = float(accuracy_score(final_truth, final_predictions))
    final_f1 = float(f1_score(final_truth, final_predictions, zero_division=0))
    average_accuracy = float(np.mean([row["accuracy"] for row in series]))
    average_f1 = float(np.mean([row["f1"] for row in series]))
    return (
        {
            "policy": policy_name,
            "retrain_proposals": proposals,
            "retrain_approved": approvals,
            "retrain_rejected_by_cost": cost_rejections,
            "retrain_count": retrain_count,
            "successful_promotions": promotions,
            "rollbacks": rollbacks,
            "average_accuracy": average_accuracy,
            "average_f1": average_f1,
            "final_accuracy": final_accuracy,
            "final_f1": final_f1,
            "retraining_cost": retraining_cost,
            "estimated_drift_cost": drift_cost,
            "total_cost": retraining_cost + drift_cost,
        },
        series,
    )


def _drift_signal(observation_index: int, drift_start: int) -> dict:
    if observation_index < drift_start:
        return {"overall_score": 0.05, "severity": "NONE", "drift_detected": False}
    if observation_index < drift_start + 100:
        return {"overall_score": 0.55, "severity": "MEDIUM", "drift_detected": True}
    return {"overall_score": 0.82, "severity": "HIGH", "drift_detected": True, "triggered_detectors": ["psi", "ks"]}


def _predictions(truth: np.ndarray, drift_start: int, *, before_error: float, after_error: float, random_seed: int) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    predictions = truth.copy()
    probs = rng.random(len(truth))
    flips = probs < before_error
    flips[drift_start:] = probs[drift_start:] < after_error
    predictions[flips] = 1 - predictions[flips]
    return predictions


def print_table(result: pd.DataFrame) -> None:
    print("Policy | Retrains | Avg F1 | Final F1 | Retrain Cost | Drift Cost | Total Cost")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for _, row in result.iterrows():
        print(
            f"{row['policy']} | {int(row['retrain_count'])} | {row['average_f1']:.3f} | {row['final_f1']:.3f} | "
            f"{row['retraining_cost']:.2f} | {row['estimated_drift_cost']:.2f} | {row['total_cost']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Week 3 retraining policies on the same deterministic stream.")
    parser.add_argument("--production-path", default="data/production/production.csv")
    parser.add_argument("--rows", type=int, default=750)
    parser.add_argument("--drift-start", type=int, default=350)
    parser.add_argument("--drift-type", choices=["sudden", "gradual"], default="sudden")
    parser.add_argument("--drift-strength", type=float, default=1.5)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-csv", default="artifacts/policy_benchmark.csv")
    parser.add_argument("--timeseries-csv", default="artifacts/policy_performance_timeseries.csv")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    results, _ = run_policy_benchmark(
        production_path=Path(args.production_path),
        rows=args.rows,
        drift_start=args.drift_start,
        drift_type=args.drift_type,
        drift_strength=args.drift_strength,
        window_size=args.window_size,
        step_size=args.step_size,
        random_seed=args.random_seed,
        output_csv=Path(args.output_csv),
        timeseries_csv=Path(args.timeseries_csv),
    )
    print_table(results)
