# Results

## Synthetic Drift Detector Benchmark

Generated with:

```bash
python scripts/benchmark_detectors.py --rows 750 --drift-start 350 --drift-type sudden --window-size 100 --step-size 25 --output-csv artifacts/drift_benchmark.csv
```

| Detector | Detection Rate | False Alarm Rate | Detection Latency |
| --- | ---: | ---: | ---: |
| psi | 1.00 | 0.33 | 24 |
| ks | 1.00 | 1.00 | 24 |
| adwin | 1.00 | 0.00 | 18 |
| confidence | 1.00 | 0.00 | 49 |
| ensemble | 1.00 | 1.33 | 49 |

## Synthetic Retraining Policy Benchmark

Generated with:

```bash
python scripts/benchmark_policies.py --rows 750 --drift-start 350 --drift-type sudden --window-size 100 --step-size 50 --output-csv artifacts/policy_benchmark.csv --timeseries-csv artifacts/policy_performance_timeseries.csv
```

| Policy | Retrains | Avg F1 | Final F1 | Retrain Cost | Drift Cost | Total Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| periodic | 1 | 0.843 | 0.857 | 32.00 | 255.00 | 287.00 |
| error_threshold | 7 | 0.858 | 0.857 | 225.40 | 880.00 | 1105.40 |
| drift_triggered | 2 | 0.858 | 0.857 | 64.40 | 195.00 | 259.40 |

## Business Scenario

Generated with:

```bash
python scripts/run_business_scenario.py --output-dir artifacts/business --rows 1800 --drift-start 350 --random-seed 123
```

The secondary scenario is a deterministic synthetic credit-risk stream with business-style features mapped into the project feature schema. Outputs:

- `artifacts/business/business_drift_benchmark.csv`
- `artifacts/business/business_policy_benchmark.csv`
- `artifacts/business/business_policy_performance_timeseries.csv`

Business policy benchmark:

| Policy | Retrains | Avg F1 | Final F1 | Retrain Cost | Drift Cost | Total Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| periodic | 1 | 0.804 | 0.769 | 32.00 | 155.00 | 187.00 |
| error_threshold | 5 | 0.799 | 0.769 | 160.60 | 590.00 | 750.60 |
| drift_triggered | 1 | 0.799 | 0.769 | 31.90 | 140.00 | 171.90 |

## Stress Test

Generated against a local FastAPI process on `127.0.0.1:8001`:

```bash
python scripts/stress_test.py --api-url http://127.0.0.1:8001 --requests 50 --output-json artifacts/stress_test.json
```

| Requests | Successes | Failures | Throughput RPS | Avg Latency ms | P95 Latency ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 50 | 0 | 33.29 | 30.04 | 47.29 |

## Findings

- In the synthetic detector benchmark, ADWIN had the fastest detection latency among detectors at 18 observations and no false alarms in this run.
- KS and the ensemble detected drift, but both generated more pre-drift alerts in this scenario.
- In the synthetic policy benchmark, `drift_triggered` reached the same final F1 as `error_threshold` with fewer retrains and lower total abstract cost units.
- The business-style credit-risk scenario preserved a relatable cost/performance comparison while remaining deterministic and small enough for local execution.

## Limitations

- Cost values are abstract units, not currency.
- The business scenario is generated deterministically rather than downloaded from a large external dataset.
- Docker, Prometheus, and Grafana provisioning were added, but Docker was not available in this execution shell for live container startup validation.

