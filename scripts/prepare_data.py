from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from app.core.logging import configure_logging
from app.ml.features import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


def prepare_datasets(data_dir: Path, rows: int, random_seed: int) -> None:
    rng = np.random.default_rng(random_seed)
    x, y = make_classification(
        n_samples=rows,
        n_features=len(FEATURE_COLUMNS),
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.03,
        random_state=random_seed,
    )
    frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    frame[TARGET_COLUMN] = y.astype(int)

    train, remaining = train_test_split(frame, test_size=0.40, stratify=frame[TARGET_COLUMN], random_state=random_seed)
    reference, remaining = train_test_split(remaining, test_size=0.625, stratify=remaining[TARGET_COLUMN], random_state=random_seed)
    production, holdout = train_test_split(remaining, test_size=0.40, stratify=remaining[TARGET_COLUMN], random_state=random_seed)

    paths = {
        "train": data_dir / "processed" / "train.csv",
        "reference": data_dir / "reference" / "reference.csv",
        "production": data_dir / "production" / "production.csv",
        "holdout": data_dir / "processed" / "holdout.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(paths["train"], index=False)
    reference.to_csv(paths["reference"], index=False)
    production.to_csv(paths["production"], index=False)
    holdout.to_csv(paths["holdout"], index=False)

    metadata = {
        "dataset": "synthetic_sklearn_classification",
        "random_seed": random_seed,
        "rows": rows,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "split_rows": {name: int(pd.read_csv(path).shape[0]) for name, path in paths.items()},
        "notes": "Deterministic Week 1 splits: 60% train, 15% reference, 15% production, 10% holdout.",
    }
    metadata_path = data_dir / "processed" / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Prepared datasets under %s", data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic Week 1 datasets.")
    parser.add_argument("--data-dir", default="data", help="Root data directory.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of synthetic rows to generate.")
    parser.add_argument("--random-seed", type=int, default=42, help="Deterministic random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    prepare_datasets(Path(args.data_dir), args.rows, args.random_seed)
