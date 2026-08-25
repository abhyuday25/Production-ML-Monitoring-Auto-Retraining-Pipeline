from pathlib import Path

from scripts.benchmark_policies import run_policy_benchmark
from scripts.prepare_data import prepare_datasets


def test_policy_benchmark_generates_outputs(tmp_path: Path):
    data_dir = tmp_path / "data"
    prepare_datasets(data_dir, rows=900, random_seed=42)
    output_csv = tmp_path / "artifacts" / "policy_benchmark.csv"
    timeseries_csv = tmp_path / "artifacts" / "policy_performance_timeseries.csv"

    result, series = run_policy_benchmark(
        production_path=data_dir / "production" / "production.csv",
        rows=150,
        drift_start=70,
        drift_type="sudden",
        drift_strength=1.2,
        window_size=50,
        step_size=25,
        random_seed=42,
        output_csv=output_csv,
        timeseries_csv=timeseries_csv,
    )

    assert output_csv.exists()
    assert timeseries_csv.exists()
    assert {"periodic", "error_threshold", "drift_triggered"} == set(result["policy"])
    assert not series.empty
