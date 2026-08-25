from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from time import perf_counter
from uuid import uuid4

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import PredictionLog
from app.observability.metrics import record_prediction, record_prediction_error
from app.schemas.prediction import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    pass


class PredictionServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelBundle:
    model: object
    feature_columns: list[str]
    model_name: str
    model_version: str | None = None
    model_run_id: str | None = None


class PredictionService:
    def __init__(self, model_bundle: ModelBundle, session_factory: sessionmaker[Session]) -> None:
        self.model_bundle = model_bundle
        self.session_factory = session_factory

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        self._validate_features(payload.features)
        request_id = str(uuid4())
        started = perf_counter()

        try:
            input_frame = pd.DataFrame([[payload.features[col] for col in self.model_bundle.feature_columns]], columns=self.model_bundle.feature_columns)
            prediction = int(self.model_bundle.model.predict(input_frame)[0])
            probability = self._predict_probability(input_frame, prediction)
            latency_ms = (perf_counter() - started) * 1000
            self._log_prediction(request_id, payload, prediction, probability, latency_ms)
            record_prediction(self.model_bundle.model_version, prediction, latency_ms / 1000.0)
        except ValidationError:
            raise
        except Exception as exc:
            logger.exception("Prediction failed")
            record_prediction_error()
            raise PredictionServiceError("Prediction failed") from exc

        logger.info("Prediction succeeded request_id=%s", request_id)
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_name=self.model_bundle.model_name,
            model_version=self.model_bundle.model_version,
            request_id=request_id,
        )

    def _validate_features(self, features: dict[str, float]) -> None:
        expected = set(self.model_bundle.feature_columns)
        received = set(features)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing:
            raise ValidationError(f"Missing required features: {', '.join(missing)}")
        if unexpected:
            raise ValidationError(f"Unexpected features: {', '.join(unexpected)}")

    def _predict_probability(self, input_frame: pd.DataFrame, prediction: int) -> float | None:
        model = self.model_bundle.model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            if len(probabilities) == 2:
                return float(probabilities[1])
            return float(max(probabilities))
        return None

    def _log_prediction(
        self,
        request_id: str,
        payload: PredictionRequest,
        prediction: int,
        probability: float | None,
        latency_ms: float,
    ) -> None:
        with self.session_factory() as session:
            session.add(
                PredictionLog(
                    request_id=request_id,
                    input_features=json.dumps(payload.features, sort_keys=True),
                    prediction=prediction,
                    prediction_probability=probability,
                    ground_truth=payload.ground_truth,
                    model_name=self.model_bundle.model_name,
                    model_version=self.model_bundle.model_version,
                    model_run_id=self.model_bundle.model_run_id,
                    latency_ms=latency_ms,
                )
            )
            session.commit()
