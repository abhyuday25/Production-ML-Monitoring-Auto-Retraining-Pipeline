from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    database_url: str = "sqlite:///./predictions.db"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ml-monitoring-baseline"
    mlflow_model_name: str = "production-monitoring-model"
    mlflow_model_alias: str = "champion"
    data_dir: str = "data"
    log_level: str = "INFO"
    skip_model_load: bool = False


def _bool_from_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    return Settings(
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./predictions.db"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-monitoring-baseline"),
        mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", "production-monitoring-model"),
        mlflow_model_alias=os.getenv("MLFLOW_MODEL_ALIAS", "champion"),
        data_dir=os.getenv("DATA_DIR", "data"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        skip_model_load=_bool_from_env(os.getenv("SKIP_MODEL_LOAD"), False),
    )

