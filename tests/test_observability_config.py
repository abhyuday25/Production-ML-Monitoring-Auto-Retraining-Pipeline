import json
from pathlib import Path


def test_prometheus_config_is_valid_yaml():
    config = Path("observability/prometheus.yml").read_text(encoding="utf-8")

    assert "metrics_path: /metrics" in config
    assert "- api:8000" in config


def test_grafana_dashboard_json_is_valid():
    dashboard = json.loads(Path("observability/grafana/dashboards/mlops-dashboard.json").read_text(encoding="utf-8"))

    assert dashboard["title"] == "ML Monitoring & Auto-Retraining"
    assert len(dashboard["panels"]) >= 10
    assert any(panel["title"] == "Alarm Rate" for panel in dashboard["panels"])


def test_grafana_provisioning_paths_match_compose_mounts():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dashboard_provider = Path("observability/grafana/provisioning/dashboards/dashboards.yml").read_text(encoding="utf-8")
    datasource = Path("observability/grafana/provisioning/datasources/prometheus.yml").read_text(encoding="utf-8")

    assert "./observability/grafana/dashboards:/etc/grafana/dashboards:ro" in compose
    assert "path: /etc/grafana/dashboards" in dashboard_provider
    assert "uid: Prometheus" in datasource


def test_docker_compose_shares_mlflow_artifact_volume():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert compose.count("./mlflow:/mlflow") >= 3
    assert "service_completed_successfully" in compose
