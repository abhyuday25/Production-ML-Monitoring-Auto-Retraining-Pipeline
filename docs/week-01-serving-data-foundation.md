# Week 1 - Serving + Data Foundation

## Implemented

Week 1 builds the local foundation for model serving and prediction logging.

Implemented components:

- Deterministic synthetic classification dataset generation.
- Train, reference, production simulation, and holdout CSV splits.
- Scikit-learn baseline pipeline with `StandardScaler` and `LogisticRegression`.
- MLflow experiment logging, model artifact logging, model registration, and `champion` alias assignment.
- FastAPI application with `/health` and `/predict`.
- MLflow model loading at application startup.
- SQLite-backed prediction log table using SQLAlchemy.
- Request validation for missing and unexpected features.
- Optional ground-truth logging.
- Production traffic simulator that streams rows from the production split to the API.
- Dockerfile and Docker Compose services for API and MLflow.
- Pytest coverage for health, prediction, validation, optional ground truth, and database logging.

## What The Feature Does

The Week 1 feature establishes the data-to-serving path required by later monitoring and retraining work:

```text
prepared dataset -> baseline training -> MLflow registration -> FastAPI model load -> prediction -> persistent log row
```

Every successful prediction stores the request ID, timestamp, input features, prediction, confidence, optional ground truth, model metadata, and latency. These records are the production-like history that future weeks can use for drift detection, policy decisions, and retraining workflows.

## How To Use

Run the workflow in order:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
python scripts/train_model.py
uvicorn app.main:app --reload
python scripts/simulate_production.py --limit 100 --include-ground-truth
python scripts/count_prediction_logs.py
pytest
```

## Not Implemented Yet

This week intentionally does not include drift detection, retraining policies, retraining DAGs, canary deployment, rollback, Prometheus, Grafana, or benchmarking. Those belong to later project weeks.

