# Project Demo Guide

## Project In Short

This project is a production-style ML monitoring and auto-retraining pipeline.

The basic idea is:

```text
Train model
→ serve predictions through FastAPI
→ log production predictions
→ detect data/model drift
→ decide whether retraining is worth it
→ retrain and validate candidate model
→ register/promote/reject model safely
→ monitor everything in Grafana
```

The project is not just a model API. It shows how an ML system is operated after deployment.

## What Each Application Is Used For

| Tool | Why We Use It In This Project |
| --- | --- |
| FastAPI | Serves the ML model through `/predict`, exposes `/health`, `/metrics`, and Swagger docs. |
| SQLite | Stores prediction logs, drift alerts, retraining decisions, retraining runs, and deployment events locally. |
| MLflow | Tracks training runs and stores registered model versions. The API loads the current `champion` model from MLflow. |
| Prometheus | Scrapes live metrics from FastAPI `/metrics`, such as prediction count, latency, drift score, and retraining events. |
| Grafana | Visual dashboard for Prometheus metrics: serving health, drift, model performance, retraining decisions, and lifecycle events. |
| Docker Compose | Starts the full local system: MLflow, model bootstrap, API, Prometheus, and Grafana. |
| pytest | Verifies serving, drift detection, retraining policy, persistence, and observability behavior. |

## Important URLs

After Docker is running:

| Service | URL |
| --- | --- |
| FastAPI Swagger | <http://localhost:8000/docs> |
| FastAPI health | <http://localhost:8000/health> |
| FastAPI metrics | <http://localhost:8000/metrics> |
| MLflow | <http://localhost:5000> |
| Prometheus | <http://localhost:9090> |
| Grafana | <http://localhost:3000> |

Grafana login:

```text
username: admin
password: admin
```

The FastAPI root page `http://localhost:8000/` may show:

```json
{"detail":"Not Found"}
```

That is normal. Use `/docs`, `/health`, or `/metrics`.

## First-Time Setup

From the project root:

```bash
copy .env.example .env
docker compose down -v
docker compose build --no-cache
docker compose up
```

On macOS/Linux:

```bash
cp .env.example .env
docker compose down -v
docker compose build --no-cache
docker compose up
```

Wait until these services are healthy/running:

```text
mlflow
model-init
api
prometheus
grafana
```

`model-init` is a one-time bootstrap container. It prepares data and registers the MLflow `champion` model before the API starts.

## Demo Flow To Show Teacher

### 1. Start With The Problem

Say:

```text
This project solves the problem of monitoring an ML model after deployment.
It detects drift, measures whether retraining is justified, retrains safely, and exposes the whole pipeline through dashboards.
```

Open the README and show the architecture diagram:

```text
README.md
```

### 2. Show The Running Services

Open:

```text
http://localhost:8000/docs
```

Say:

```text
This is the FastAPI model-serving layer. The main endpoint is /predict.
```

Open:

```text
http://localhost:5000
```

Say:

```text
This is MLflow. It tracks training runs and stores model versions. The API loads the champion model from here.
```

Open:

```text
http://localhost:3000
```

Go to:

```text
Dashboards → MLOps → ML Monitoring & Auto-Retraining
```

Say:

```text
This is Grafana. It visualizes live metrics scraped by Prometheus from the API.
```

### 3. Show The API Works

Open:

```text
http://localhost:8000/health
```

Expected:

```json
{"status":"healthy","model_loaded":true}
```

Then open:

```text
http://localhost:8000/docs
```

Use the `/predict` endpoint or run this command:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 100 --include-ground-truth
```

Say:

```text
This simulates production traffic and logs predictions into the database.
```

### 4. Show Metrics

Open:

```text
http://localhost:8000/metrics
```

Search for:

```text
mlops_predictions_total
mlops_prediction_latency_seconds
mlops_model_accuracy
```

Say:

```text
FastAPI exposes Prometheus metrics. Prometheus scrapes these metrics, and Grafana visualizes them.
```

### 5. Show Grafana Dashboard Populating

Open Grafana:

```text
http://localhost:3000
```

Dashboard:

```text
MLOps / ML Monitoring & Auto-Retraining
```

Show panels:

```text
Total Predictions
Prediction Throughput
Prediction Latency
Rolling Model Performance
```

If panels are empty, generate more traffic:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 250 --include-ground-truth
```

Wait 5-10 seconds because Prometheus scrapes every few seconds.

### 6. Inject Drift

Run:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 750 --drift-start 350 --drift-type sudden --drift-strength 1.5 --include-ground-truth
python scripts/run_drift_detection.py
```

Say:

```text
Now we inject synthetic drift after row 350. The drift detection service compares production logs against reference data.
```

In Grafana, show:

```text
Current Drift Score
Drift Score and Alerts
PSI by Feature
KS and Confidence Drift
Alarm Rate
```

### 7. Explain Drift Detectors

Say:

```text
PSI compares feature distribution shifts.
KS-test statistically compares reference and production feature distributions.
ADWIN detects concept drift from prediction errors when ground truth is available.
Confidence drift checks whether model confidence changes.
The ensemble combines these into one drift score.
```

### 8. Run Retraining Policy

Run:

```bash
python scripts/run_retraining_policy.py --policy drift_triggered
```

Say:

```text
The policy engine decides whether drift should propose retraining.
But retraining is not automatic. The cost-aware gate checks whether retraining is worth it.
```

Show Grafana panels:

```text
Retraining Event Markers
Cost-Aware Decisions
```

### 9. Run Retraining Flow

Run:

```bash
python scripts/run_retraining_flow.py --policy drift_triggered
```

Say:

```text
If retraining is approved, the pipeline builds a labeled dataset, trains a candidate, validates it on holdout data, compares it with the champion, evaluates shadow/canary behavior, and either promotes or rejects it.
```

Open MLflow:

```text
http://localhost:5000
```

Show:

```text
training runs
registered model
model versions
champion alias
```

### 10. Show Benchmark Results

Open:

```text
docs/RESULTS.md
```

Say:

```text
The project also benchmarks drift detectors and retraining policies on deterministic scenarios.
```

Important results to mention:

```text
ADWIN detected drift fastest in the synthetic benchmark.
Drift-triggered retraining reached the same final F1 as error-threshold retraining with fewer retrains and lower total cost units.
```

### 11. Show Business Scenario

Run:

```bash
python scripts/run_business_scenario.py --output-dir artifacts/business --rows 1800 --drift-start 350 --random-seed 123
```

Say:

```text
This creates a deterministic credit-risk style business scenario, so the project is not limited to an abstract synthetic dataset.
```

Generated files:

```text
artifacts/business/business_drift_benchmark.csv
artifacts/business/business_policy_benchmark.csv
artifacts/business/business_policy_performance_timeseries.csv
```

## Short 3-Minute Demo Script

Use this if time is limited.

1. Show README architecture.
2. Open FastAPI `/docs`.
3. Open MLflow and explain champion model.
4. Open Grafana dashboard.
5. Run normal traffic:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 100 --include-ground-truth
```

6. Run drift traffic:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 750 --drift-start 350 --drift-type sudden --include-ground-truth
python scripts/run_drift_detection.py
```

7. Show Grafana drift panels.
8. Run:

```bash
python scripts/run_retraining_policy.py --policy drift_triggered
```

9. Open `docs/RESULTS.md` and explain benchmark findings.

## If Something Looks Empty

### Grafana dashboard list empty

Run:

```bash
docker compose up -d --force-recreate grafana
```

Open:

```text
http://localhost:3000/dashboards
```

Expected dashboard:

```text
MLOps / ML Monitoring & Auto-Retraining
```

### Grafana panels empty

Generate traffic and events:

```bash
python scripts/simulate_production.py --api-url http://localhost:8000 --limit 250 --include-ground-truth
python scripts/run_drift_detection.py
python scripts/run_retraining_policy.py --policy drift_triggered
```

Wait 5-10 seconds.

### FastAPI root shows Not Found

Use:

```text
http://localhost:8000/docs
```

not:

```text
http://localhost:8000/
```

## What To Say In One Paragraph

```text
This project is a complete local MLOps pipeline. A model is trained and registered in MLflow, served through FastAPI, and every prediction is logged. The system monitors production data for drift using PSI, KS-test, ADWIN, and confidence drift. It aggregates those signals, evaluates retraining policies, applies a cost-aware gate, and safely trains and validates candidate models before promotion. Prometheus collects live metrics and Grafana visualizes serving, drift, performance, retraining, cost, and lifecycle events.
```

