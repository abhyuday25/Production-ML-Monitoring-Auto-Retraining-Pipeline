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
    drift_window_size: int = 200
    min_drift_samples: int = 100
    psi_bins: int = 10
    psi_warning_threshold: float = 0.10
    psi_drift_threshold: float = 0.20
    ks_alpha: float = 0.05
    adwin_delta: float = 0.002
    min_adwin_samples: int = 30
    confidence_drift_threshold: float = 0.10
    drift_weight_psi: float = 0.30
    drift_weight_ks: float = 0.25
    drift_weight_adwin: float = 0.30
    drift_weight_confidence: float = 0.15
    drift_low_threshold: float = 0.20
    drift_medium_threshold: float = 0.40
    drift_high_threshold: float = 0.65
    drift_critical_threshold: float = 0.85
    top_drift_features: int = 5
    random_seed: int = 42


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
        drift_window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "200")),
        min_drift_samples=int(os.getenv("MIN_DRIFT_SAMPLES", "100")),
        psi_bins=int(os.getenv("PSI_BINS", "10")),
        psi_warning_threshold=float(os.getenv("PSI_WARNING_THRESHOLD", "0.10")),
        psi_drift_threshold=float(os.getenv("PSI_DRIFT_THRESHOLD", "0.20")),
        ks_alpha=float(os.getenv("KS_ALPHA", "0.05")),
        adwin_delta=float(os.getenv("ADWIN_DELTA", "0.002")),
        min_adwin_samples=int(os.getenv("MIN_ADWIN_SAMPLES", "30")),
        confidence_drift_threshold=float(os.getenv("CONFIDENCE_DRIFT_THRESHOLD", "0.10")),
        drift_weight_psi=float(os.getenv("DRIFT_WEIGHT_PSI", "0.30")),
        drift_weight_ks=float(os.getenv("DRIFT_WEIGHT_KS", "0.25")),
        drift_weight_adwin=float(os.getenv("DRIFT_WEIGHT_ADWIN", "0.30")),
        drift_weight_confidence=float(os.getenv("DRIFT_WEIGHT_CONFIDENCE", "0.15")),
        drift_low_threshold=float(os.getenv("DRIFT_LOW_THRESHOLD", "0.20")),
        drift_medium_threshold=float(os.getenv("DRIFT_MEDIUM_THRESHOLD", "0.40")),
        drift_high_threshold=float(os.getenv("DRIFT_HIGH_THRESHOLD", "0.65")),
        drift_critical_threshold=float(os.getenv("DRIFT_CRITICAL_THRESHOLD", "0.85")),
        top_drift_features=int(os.getenv("TOP_DRIFT_FEATURES", "5")),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )
