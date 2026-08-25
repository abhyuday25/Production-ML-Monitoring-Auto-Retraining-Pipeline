from __future__ import annotations

try:
    from prefect import flow
except Exception:  # pragma: no cover - used when Prefect is not installed
    def flow(fn=None, **_kwargs):
        def decorator(func):
            return func

        return decorator(fn) if fn is not None else decorator

from app.core.config import Settings
from app.retraining.service import run_retraining_pipeline


@flow(name="week3-retraining-flow")
def retraining_flow(settings: Settings, session, policy_name: str) -> dict:
    return run_retraining_pipeline(settings, session, policy_name)
