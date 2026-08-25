from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover - exercised when prometheus-client is not installed
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    CollectorRegistry = None
    Counter = Gauge = Histogram = None
    generate_latest = None

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import DeploymentEvent, DriftAlert, DriftMeasurement, PredictionLog, RetrainingDecision, RetrainingRun


SEVERITY_VALUE = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


class _FallbackMetric:
    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self.values: dict[tuple[str, ...], float] = defaultdict(float)

    def labels(self, *labelvalues: str, **labelkwargs: str):
        if labelkwargs:
            labelvalues = tuple(labelkwargs[name] for name in self.labelnames)
        child = _FallbackChild(self, tuple(str(value) for value in labelvalues))
        return child

    def inc(self, amount: float = 1.0) -> None:
        self.values[()] += amount

    def set(self, value: float) -> None:
        self.values[()] = value

    def observe(self, value: float) -> None:
        self.values[()] += value


class _FallbackChild:
    def __init__(self, metric: _FallbackMetric, labels: tuple[str, ...]) -> None:
        self.metric = metric
        self.labels = labels

    def inc(self, amount: float = 1.0) -> None:
        self.metric.values[self.labels] += amount

    def set(self, value: float) -> None:
        self.metric.values[self.labels] = value

    def observe(self, value: float) -> None:
        self.metric.values[self.labels] += value


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry() if CollectorRegistry is not None else None
        self._fallback_metrics: list[_FallbackMetric] = []
        self.predictions_total = self._counter("mlops_predictions_total", "Total predictions served.", ("model_version", "prediction_class"))
        self.prediction_errors_total = self._counter("mlops_prediction_errors_total", "Total prediction failures.")
        self.prediction_latency_seconds = self._histogram("mlops_prediction_latency_seconds", "Prediction latency in seconds.")
        self.model_accuracy = self._gauge("mlops_model_accuracy", "Rolling labeled prediction accuracy.")
        self.model_error_rate = self._gauge("mlops_model_error_rate", "Rolling labeled prediction error rate.")
        self.model_f1 = self._gauge("mlops_model_f1", "Rolling labeled prediction F1 score.")
        self.drift_score = self._gauge("mlops_drift_score", "Latest aggregate drift score.")
        self.drift_severity = self._gauge("mlops_drift_severity", "Latest drift severity numeric value: NONE=0 LOW=1 MEDIUM=2 HIGH=3 CRITICAL=4.")
        self.drift_alerts_total = self._counter("mlops_drift_alerts_total", "Total drift alerts.")
        self.drifting_features = self._gauge("mlops_drifting_features", "Latest number of top drifting features.")
        self.psi_score = self._gauge("mlops_psi_score", "Latest PSI score by feature.", ("feature",))
        self.ks_statistic = self._gauge("mlops_ks_statistic", "Latest KS statistic by feature.", ("feature",))
        self.adwin_detections_total = self._counter("mlops_adwin_detections_total", "Total ADWIN drift detections.")
        self.confidence_drift_score = self._gauge("mlops_confidence_drift_score", "Latest confidence drift score.")
        self.retrain_proposals_total = self._counter("mlops_retrain_proposals_total", "Retraining proposals by policy.", ("policy",))
        self.retrain_approved_total = self._counter("mlops_retrain_approved_total", "Cost-approved retrains by policy.", ("policy",))
        self.retrain_cost_rejected_total = self._counter("mlops_retrain_cost_rejected_total", "Retrain proposals rejected by cost gate.", ("policy",))
        self.retraining_runs_total = self._counter("mlops_retraining_runs_total", "Retraining runs by policy and status.", ("policy", "status"))
        self.retraining_success_total = self._counter("mlops_retraining_success_total", "Successful retraining promotions.", ("policy",))
        self.retraining_failures_total = self._counter("mlops_retraining_failures_total", "Failed or rejected retraining runs.", ("policy", "status"))
        self.retraining_duration_seconds = self._histogram("mlops_retraining_duration_seconds", "Retraining duration in seconds.", ("policy", "status"))
        self.candidates_registered_total = self._counter("mlops_candidates_registered_total", "Candidate model registrations.")
        self.candidates_rejected_total = self._counter("mlops_candidates_rejected_total", "Candidate model rejections.")
        self.model_promotions_total = self._counter("mlops_model_promotions_total", "Model promotions.")
        self.model_rollbacks_total = self._counter("mlops_model_rollbacks_total", "Model rollbacks.")
        self.active_model_version = self._gauge("mlops_active_model_version", "Active numeric model version.")
        self.estimated_retraining_cost = self._gauge("mlops_estimated_retraining_cost", "Latest estimated retraining cost in abstract cost units.", ("policy",))
        self.estimated_drift_cost = self._gauge("mlops_estimated_drift_cost", "Latest estimated drift cost in abstract cost units.", ("policy",))
        self.retraining_net_benefit = self._gauge("mlops_retraining_net_benefit", "Latest retraining net benefit in abstract cost units.", ("policy",))

    def _counter(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()):
        if Counter is not None:
            return Counter(name, documentation, labelnames, registry=self.registry)
        return self._fallback(name, documentation, labelnames)

    def _gauge(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()):
        if Gauge is not None:
            return Gauge(name, documentation, labelnames, registry=self.registry)
        return self._fallback(name, documentation, labelnames)

    def _histogram(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()):
        if Histogram is not None:
            return Histogram(name, documentation, labelnames, registry=self.registry)
        return self._fallback(name, documentation, labelnames)

    def _fallback(self, name: str, documentation: str, labelnames: tuple[str, ...]):
        metric = _FallbackMetric(name, documentation, labelnames)
        self._fallback_metrics.append(metric)
        return metric

    def render(self) -> bytes:
        if generate_latest is not None:
            return generate_latest(self.registry)
        lines: list[str] = []
        for metric in self._fallback_metrics:
            lines.append(f"# HELP {metric.name} {metric.documentation}")
            lines.append(f"# TYPE {metric.name} gauge")
            for labels, value in metric.values.items():
                if labels:
                    label_text = ",".join(f'{name}="{value}"' for name, value in zip(metric.labelnames, labels, strict=False))
                    lines.append(f"{metric.name}{{{label_text}}} {value}")
                else:
                    lines.append(f"{metric.name} {value}")
        return ("\n".join(lines) + "\n").encode("utf-8")


metrics = Metrics()


def record_prediction(model_version: str | None, prediction: int, latency_seconds: float) -> None:
    metrics.predictions_total.labels(model_version=str(model_version or "unknown"), prediction_class=str(prediction)).inc()
    metrics.prediction_latency_seconds.observe(latency_seconds)
    if model_version is not None and str(model_version).isdigit():
        metrics.active_model_version.set(float(model_version))


def record_prediction_error() -> None:
    metrics.prediction_errors_total.inc()


def record_drift_results(results: list[Any], assessment: Any) -> None:
    metrics.drift_score.set(float(assessment.overall_score))
    metrics.drift_severity.set(float(SEVERITY_VALUE.get(str(assessment.severity).upper(), 0)))
    metrics.drifting_features.set(float(len(assessment.top_drifting_features)))
    if assessment.drift_detected:
        metrics.drift_alerts_total.inc()
    for result in results:
        if result.score is None:
            continue
        if result.detector == "psi" and result.feature:
            metrics.psi_score.labels(feature=result.feature).set(float(result.score))
        elif result.detector == "ks" and result.feature:
            metrics.ks_statistic.labels(feature=result.feature).set(float(result.score))
        elif result.detector == "adwin" and result.drift_detected:
            metrics.adwin_detections_total.inc()
        elif result.detector == "confidence":
            metrics.confidence_drift_score.set(float(result.score))


def record_policy_decision(policy: str, policy_triggered: bool, cost_approved: bool, estimated_drift_cost: float, estimated_retraining_cost: float, net_benefit: float) -> None:
    if policy_triggered:
        metrics.retrain_proposals_total.labels(policy=policy).inc()
    if cost_approved:
        metrics.retrain_approved_total.labels(policy=policy).inc()
    elif policy_triggered:
        metrics.retrain_cost_rejected_total.labels(policy=policy).inc()
    metrics.estimated_drift_cost.labels(policy=policy).set(float(estimated_drift_cost))
    metrics.estimated_retraining_cost.labels(policy=policy).set(float(estimated_retraining_cost))
    metrics.retraining_net_benefit.labels(policy=policy).set(float(net_benefit))


def record_retraining_run(policy: str, status: str, duration_seconds: float | None = None) -> None:
    metrics.retraining_runs_total.labels(policy=policy, status=status).inc()
    if status == "promoted":
        metrics.retraining_success_total.labels(policy=policy).inc()
    elif status not in {"running"}:
        metrics.retraining_failures_total.labels(policy=policy, status=status).inc()
    if duration_seconds is not None:
        metrics.retraining_duration_seconds.labels(policy=policy, status=status).observe(float(duration_seconds))


def record_deployment_event_metric(event_type: str) -> None:
    if event_type == "candidate_registered":
        metrics.candidates_registered_total.inc()
    elif event_type == "candidate_rejected":
        metrics.candidates_rejected_total.inc()
    elif event_type == "candidate_promoted":
        metrics.model_promotions_total.inc()
    elif event_type == "rollback":
        metrics.model_rollbacks_total.inc()


def refresh_metrics_from_database(session: Session) -> None:
    _refresh_performance(session)
    _refresh_drift(session)
    _refresh_lifecycle(session)


def _refresh_performance(session: Session, limit: int = 200) -> None:
    rows = (
        session.query(PredictionLog)
        .filter(PredictionLog.ground_truth.is_not(None))
        .order_by(desc(PredictionLog.id))
        .limit(limit)
        .all()
    )
    if not rows:
        return
    errors = [0 if int(row.prediction) == int(row.ground_truth) else 1 for row in rows]
    error_rate = sum(errors) / len(errors)
    accuracy = 1.0 - error_rate
    true_positive = sum(1 for row in rows if row.prediction == 1 and row.ground_truth == 1)
    false_positive = sum(1 for row in rows if row.prediction == 1 and row.ground_truth == 0)
    false_negative = sum(1 for row in rows if row.prediction == 0 and row.ground_truth == 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics.model_accuracy.set(float(accuracy))
    metrics.model_error_rate.set(float(error_rate))
    metrics.model_f1.set(float(f1))


def _refresh_drift(session: Session) -> None:
    alert = session.query(DriftAlert).order_by(desc(DriftAlert.id)).first()
    if alert is not None:
        metrics.drift_score.set(float(alert.overall_score))
        metrics.drift_severity.set(float(SEVERITY_VALUE.get(alert.severity, 0)))
        metrics.drifting_features.set(float(len(json.loads(alert.top_drifting_features or "[]"))))
    measurements = session.query(DriftMeasurement).order_by(desc(DriftMeasurement.id)).limit(100).all()
    seen: set[tuple[str, str | None]] = set()
    for measurement in measurements:
        key = (measurement.detector, measurement.feature)
        if key in seen or measurement.score is None:
            continue
        seen.add(key)
        if measurement.detector == "psi" and measurement.feature:
            metrics.psi_score.labels(feature=measurement.feature).set(float(measurement.score))
        elif measurement.detector == "ks" and measurement.feature:
            metrics.ks_statistic.labels(feature=measurement.feature).set(float(measurement.score))
        elif measurement.detector == "confidence":
            metrics.confidence_drift_score.set(float(measurement.score))


def _refresh_lifecycle(session: Session) -> None:
    decision = session.query(RetrainingDecision).order_by(desc(RetrainingDecision.id)).first()
    if decision is not None:
        metrics.estimated_drift_cost.labels(policy=decision.policy).set(float(decision.estimated_drift_cost))
        metrics.estimated_retraining_cost.labels(policy=decision.policy).set(float(decision.estimated_retraining_cost))
        metrics.retraining_net_benefit.labels(policy=decision.policy).set(float(decision.net_benefit))
    run = session.query(RetrainingRun).order_by(desc(RetrainingRun.id)).first()
    if run is not None and run.duration_seconds is not None:
        metrics.retraining_duration_seconds.labels(policy=run.policy, status=run.status).observe(float(run.duration_seconds))
    event = session.query(DeploymentEvent).order_by(desc(DeploymentEvent.id)).first()
    if event is not None and event.new_model_version and str(event.new_model_version).isdigit():
        metrics.active_model_version.set(float(event.new_model_version))
