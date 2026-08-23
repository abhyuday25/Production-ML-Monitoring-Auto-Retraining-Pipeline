from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., min_length=1)
    ground_truth: int | None = None


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None
    model_name: str
    model_version: str | None
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

