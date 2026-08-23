from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import DriftAlert, DriftMeasurement
from app.drift.persistence import persist_drift_results
from app.drift.schemas import DriftAssessment, DriftResult


def test_drift_results_are_persisted(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'drift.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        alert = persist_drift_results(
            session,
            [DriftResult("psi", 0.3, 0.2, True, 120, "feature_0")],
            DriftAssessment(0.5, "MEDIUM", True, ["psi"], ["feature_0"], datetime.now(), 120),
        )
        assert alert.id is not None
        assert session.query(DriftMeasurement).count() == 1
        assert session.query(DriftAlert).count() == 1
