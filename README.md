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

## Week 2 - Drift Detection Ensemble

Week 2 adds a local drift-monitoring layer on top of the existing reference split and `prediction_logs` table.

Implemented detectors:

- PSI feature drift using deterministic reference quantile bins.
- KS-test feature drift using `scipy.stats.ks_2samp`.
- ADWIN concept drift using prediction error when `ground_truth` is logged.
- Prediction-confidence drift using absolute change in mean binary confidence.
- Weighted drift aggregation with severity levels, triggered detectors, and top drifting features.

Drift measurements are stored in `drift_measurements`; aggregate alerts are stored in `drift_alerts`.

### Run Drift Detection

Start the normal Week 1 dependencies first if you want fresh prediction logs:

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
python scripts/train_model.py
uvicorn app.main:app --reload
```

Simulate normal production:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 250 --include-ground-truth
```

Simulate known sudden drift:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 750 --drift-start 350 --drift-type sudden --drift-strength 1.5 --include-ground-truth --metadata-path artifacts/synthetic_drift_metadata.json
```

Run the detector ensemble on the latest prediction-log window:

```bash
python scripts/run_drift_detection.py
```

If fewer than `MIN_DRIFT_SAMPLES` prediction rows exist, the monitor persists an `insufficient_data` result instead of raising a misleading alert.

### Benchmark Detectors

Run the reproducible benchmark:

```bash
python scripts/benchmark_detectors.py --rows 750 --drift-start 350 --drift-type sudden --window-size 100 --step-size 25 --output-csv artifacts/drift_benchmark.csv
```

Definitions:

- Detection latency is the first deduplicated alert index at or after the known drift start minus `drift_start`.
- False alarms are deduplicated alerts before `drift_start - window_size`.
- Alert deduplication uses a cooldown of half the benchmark window size.

Latest generated benchmark:

| Detector | Detection Rate | False Alarm Rate | Detection Latency |
| --- | ---: | ---: | ---: |
| psi | 1.00 | 0.33 | 24 |
| ks | 1.00 | 1.00 | 24 |
| adwin | 1.00 | 0.00 | 18 |
| confidence | 1.00 | 0.00 | 49 |
| ensemble | 1.00 | 1.33 | 49 |

Outputs:

- `artifacts/drift_benchmark.csv`
- `artifacts/drift_benchmark.metadata.json`

Verify persisted drift state:

```bash
python -c "import sqlite3; c=sqlite3.connect('predictions.db'); print(c.execute('select count(*) from drift_measurements').fetchone()); print(c.execute('select severity, overall_score, drift_detected, window_size from drift_alerts order by id desc limit 1').fetchone())"
```
