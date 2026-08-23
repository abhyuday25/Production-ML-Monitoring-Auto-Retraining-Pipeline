from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    input_features: Mapped[str] = mapped_column(Text, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_truth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)


class DriftMeasurement(Base):
    __tablename__ = "drift_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    detector: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    feature: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    drift_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="ok")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class DriftAlert(Base):
    __tablename__ = "drift_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    drift_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    triggered_detectors: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    top_drifting_features: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    window_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
