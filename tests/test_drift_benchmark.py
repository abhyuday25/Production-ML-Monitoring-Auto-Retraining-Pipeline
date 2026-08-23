from pathlib import Path

import pandas as pd

from scripts.benchmark_detectors import evaluate_alerts, run_benchmark
from scripts.prepare_data import prepare_datasets
from scripts.simulate_production import inject_drift


def test_inject_drift_preserves_metadata():
    frame = pd.DataFrame({"feature_0": [0.0] * 5, "feature_1": [0.0] * 5, "feature_2": [0.0] * 5, "target": [0] * 5})
    drifted, metadata = inject_drift(frame, drift_start=2, drift_type="sudden", drift_strength=1.5, random_seed=42)

    assert metadata["drift_start_index"] == 2
    assert drifted.loc[2, "feature_0"] == 1.5


def test_evaluate_alerts_latency_and_false_alarm():
    metrics = evaluate_alerts("psi", [50, 130, 150], drift_start=120, rows=200, tolerance=20)

    assert metrics["false_positives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["detection_latency"] == 10


def test_benchmark_generates_csv(tmp_path: Path):
    data_dir = tmp_path / "data"
    prepare_datasets(data_dir, rows=900, random_seed=42)
    output_csv = tmp_path / "artifacts" / "drift_benchmark.csv"

    result = run_benchmark(
        reference_path=data_dir / "reference" / "reference.csv",
        production_path=data_dir / "production" / "production.csv",
        rows=120,
        drift_start=60,
        drift_type="sudden",
        drift_strength=1.2,
        window_size=30,
        step_size=15,
        random_seed=42,
        output_csv=output_csv,
    )

    assert output_csv.exists()
    assert {"psi", "ks", "adwin", "confidence", "ensemble"} == set(result["detector"])
