from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.database import configure_database, create_tables
from app.ml.model_loader import load_model
from app.services.prediction_service import ModelBundle, PredictionService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, model_bundle: ModelBundle | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        session_factory = configure_database(settings.database_url)
        create_tables()
        bundle = model_bundle
        if bundle is None and not settings.skip_model_load:
            logger.info("Loading model %s@%s from MLflow", settings.mlflow_model_name, settings.mlflow_model_alias)
            bundle = load_model(settings)
        if bundle is not None:
            app.state.prediction_service = PredictionService(bundle, session_factory)
            logger.info("Prediction service ready")
        else:
            logger.warning("Prediction service started without a model")
        yield

    app = FastAPI(title="Production ML Monitoring API", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()

