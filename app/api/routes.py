from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.api.training import create_training_job, get_training_job, list_training_jobs
from app.db import database
from app.observability.metrics import CONTENT_TYPE_LATEST, metrics, refresh_metrics_from_database
from app.schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService, PredictionServiceError, ValidationError

router = APIRouter()


def get_prediction_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready",
        )
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "prediction_service", None)
    return HealthResponse(status="healthy" if service else "unhealthy", model_loaded=service is not None)


@router.get("/metrics")
def prometheus_metrics() -> Response:
    if database.SessionLocal is not None:
        with database.SessionLocal() as session:
            refresh_metrics_from_database(session)
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)


@router.post("/training/upload")
def upload_training_dataset(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    return create_training_job(request, background_tasks, file)


@router.get("/training/jobs")
def training_jobs() -> dict:
    return list_training_jobs()


@router.get("/training/jobs/{job_id}")
def training_job(job_id: str) -> dict:
    return get_training_job(job_id)


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    try:
        return service.predict(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except PredictionServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
