from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import urllib.error
import urllib.request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings


def check_url(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight project readiness checks.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--mlflow-url", default="http://localhost:5000")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--grafana-url", default="http://localhost:3000")
    args = parser.parse_args()
    settings = get_settings()
    checks = {
        "config_loads": True,
        "reference_data_exists": (Path(settings.data_dir) / "reference" / "reference.csv").exists(),
        "production_data_exists": (Path(settings.data_dir) / "production" / "production.csv").exists(),
        "holdout_data_exists": (Path(settings.data_dir) / "processed" / "holdout.csv").exists(),
        "benchmark_outputs_exist": Path("artifacts/drift_benchmark.csv").exists() and Path("artifacts/policy_benchmark.csv").exists(),
        "api_health_reachable": check_url(args.api_url.rstrip("/") + "/health"),
        "api_metrics_reachable": check_url(args.api_url.rstrip("/") + "/metrics"),
        "mlflow_reachable": check_url(args.mlflow_url),
        "prometheus_reachable": check_url(args.prometheus_url.rstrip("/") + "/-/ready"),
        "grafana_reachable": check_url(args.grafana_url.rstrip("/") + "/api/health"),
    }
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.replace("sqlite:///", "", 1))
        checks["database_file_exists"] = db_path.exists()
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                checks["prediction_log_table_exists"] = bool(conn.execute("select name from sqlite_master where type='table' and name='prediction_logs'").fetchone())
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
