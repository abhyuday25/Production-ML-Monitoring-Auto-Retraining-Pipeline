import json

from sqlalchemy import text

from app.db import database
from app.db.models import PredictionLog


def test_prediction_creates_database_row(client, feature_payload):
    response = client.post("/predict", json={"features": feature_payload, "ground_truth": 1})

    assert response.status_code == 200
    request_id = response.json()["request_id"]
    with database.SessionLocal() as session:
        row = session.query(PredictionLog).filter_by(request_id=request_id).one()

    assert row.request_id == request_id
    assert json.loads(row.input_features) == feature_payload
    assert row.prediction in [0, 1]
    assert row.timestamp is not None
    assert row.model_name == "test-model"
    assert row.model_version == "test-version"
    assert row.model_run_id == "test-run"
    assert row.ground_truth == 1


def test_multiple_predictions_create_multiple_rows(client, feature_payload):
    for _ in range(3):
        response = client.post("/predict", json={"features": feature_payload})
        assert response.status_code == 200

    with database.SessionLocal() as session:
        count = session.execute(text("SELECT COUNT(*) FROM prediction_logs")).scalar_one()

    assert count == 3
