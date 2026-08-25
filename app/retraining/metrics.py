from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate_classifier(model, frame: pd.DataFrame) -> dict[str, float]:
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].astype(int)
    predictions = model.predict(x)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
        probabilities = model.predict_proba(x)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
    return metrics
