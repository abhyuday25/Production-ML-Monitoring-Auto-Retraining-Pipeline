# Production ML Monitoring & Auto-Retraining Pipeline

## Week 1 - Serving + Data Foundation

This repository now contains the Week 1 foundation: deterministic dataset preparation, baseline model training, MLflow registration, FastAPI serving, prediction logging, production-stream simulation, Docker support, and focused tests.

### Install

```bash
pip install -r requirements.txt
```

### Prepare Data

```bash
python scripts/prepare_data.py
```

This creates:

- `data/processed/train.csv`
- `data/reference/reference.csv`
- `data/production/production.csv`
- `data/processed/holdout.csv`

### Start MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Train and Register the Baseline Model

In a second terminal:

```bash
python scripts/train_model.py
```

The model is registered as `production-monitoring-model` and assigned the `champion` alias.

### Start the API

```bash
uvicorn app.main:app --reload
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Prediction Example

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":{\"feature_0\":0.1,\"feature_1\":0.2,\"feature_2\":0.3,\"feature_3\":0.4,\"feature_4\":0.5,\"feature_5\":0.6,\"feature_6\":0.7,\"feature_7\":0.8,\"feature_8\":0.9,\"feature_9\":1.0},\"ground_truth\":1}"
```

### Run the Production Simulator

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 100 --delay 0.1 --include-ground-truth
```

### Inspect Prediction Logs

```bash
python scripts/count_prediction_logs.py
```

Or, if the SQLite CLI is installed:

```bash
sqlite3 predictions.db "SELECT COUNT(*) FROM prediction_logs;"
```

### Run Tests

```bash
pytest
```

### Docker Compose

```bash
docker compose up --build
```

For Docker Compose, prepare data and train the model against the compose MLflow server before starting the API with a registered model available.

