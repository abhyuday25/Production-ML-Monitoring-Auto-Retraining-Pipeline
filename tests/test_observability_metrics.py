import json

from sqlalchemy import text

from app.db import database
from app.db.models import DeploymentEvent, DriftAlert, DriftMeasurement, RetrainingDecision
from app.observability.metrics import metrics, record_deployment_event_metric, record_policy_decision


def test_metrics_endpoint_returns_prometheus_text(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "mlops_predictions_total" in response.text


def test_prediction_updates_metrics(client, feature_payload):
    response = client.post("/predict", json={"features": feature_payload, "ground_truth": 1})
    assert response.status_code == 200

    metrics_response = client.get("/metrics")

    assert "mlops_predictions_total" in metrics_response.text
    assert "mlops_prediction_latency_seconds" in metrics_response.text


def test_drift_metrics_refresh_from_database(client):
    with database.SessionLocal() as session:
        session.add(DriftAlert(overall_score=0.75, severity="HIGH", drift_detected=1, triggered_detectors=json.dumps(["psi"]), top_drifting_features=json.dumps(["feature_0"]), window_size=100))
        session.add(DriftMeasurement(detector="psi", feature="feature_0", score=0.31, threshold=0.2, drift_detected=1, sample_count=100))
        session.commit()

    response = client.get("/metrics")

    assert "mlops_drift_score" in response.text
    assert 'mlops_psi_score{feature="feature_0"}' in response.text


def test_retraining_metrics_update_from_policy_decision():
    record_policy_decision("periodic", True, False, 1.0, 5.0, -4.0)

    body = metrics.render().decode("utf-8")

    assert "mlops_retrain_proposals_total" in body
    assert "mlops_retrain_cost_rejected_total" in body


def test_lifecycle_metrics_update_from_event():
    record_deployment_event_metric("candidate_promoted")
    record_deployment_event_metric("rollback")

    body = metrics.render().decode("utf-8")

    assert "mlops_model_promotions_total" in body
    assert "mlops_model_rollbacks_total" in body


def test_lifecycle_cost_metrics_refresh_from_database(client):
    with database.SessionLocal() as session:
        session.add(
            RetrainingDecision(
                policy="drift_triggered",
                policy_triggered=1,
                policy_reason="test",
                trigger_metrics="{}",
                cost_approved=1,
                estimated_drift_cost=10.0,
                estimated_retraining_cost=4.0,
                net_benefit=6.0,
                final_should_retrain=1,
            )
        )
        session.add(DeploymentEvent(event_type="candidate_promoted", old_model_version="1", new_model_version="2", reason="test", metrics="{}"))
        session.commit()

    response = client.get("/metrics")

    assert 'mlops_estimated_drift_cost{policy="drift_triggered"}' in response.text
    assert "mlops_active_model_version" in response.text
