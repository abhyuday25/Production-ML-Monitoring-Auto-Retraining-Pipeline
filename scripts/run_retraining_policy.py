from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.database import configure_database, create_tables
from app.retraining.service import evaluate_policy_with_cost


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Week 3 retraining policy and cost-aware gate.")
    parser.add_argument("--policy", choices=["periodic", "error_threshold", "drift_triggered"], required=True)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    session_factory = configure_database(settings.database_url)
    create_tables()
    with session_factory() as session:
        decision, cost = evaluate_policy_with_cost(settings, session, args.policy)
    print(json.dumps({"decision": asdict(decision), "cost": asdict(cost)}, indent=2, default=str))


if __name__ == "__main__":
    main()
