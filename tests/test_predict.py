from app.ml.features import FEATURE_COLUMNS


def test_predict_accepts_valid_features(client, feature_payload):
    response = client.post("/predict", json={"features": feature_payload, "ground_truth": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["probability"] is not None
    assert body["model_name"] == "test-model"
    assert body["model_version"] == "test-version"
    assert body["request_id"]


def test_predict_ground_truth_is_optional(client, feature_payload):
    response = client.post("/predict", json={"features": feature_payload})

    assert response.status_code == 200
    assert response.json()["request_id"]


def test_predict_rejects_missing_feature(client, feature_payload):
    feature_payload.pop(FEATURE_COLUMNS[0])

    response = client.post("/predict", json={"features": feature_payload})

    assert response.status_code == 422
    assert "Missing required features" in response.json()["detail"]


def test_predict_rejects_unexpected_feature(client, feature_payload):
    feature_payload["extra_feature"] = 1.0

    response = client.post("/predict", json={"features": feature_payload})

    assert response.status_code == 422
    assert "Unexpected features" in response.json()["detail"]

