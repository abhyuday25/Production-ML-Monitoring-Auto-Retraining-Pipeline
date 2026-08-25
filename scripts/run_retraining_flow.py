from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.database import configure_database, create_tables
from app.workflows.retraining_flow import retraining_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Week 3 cost-aware retraining workflow.")
    parser.add_argument("--policy", choices=["periodic", "error_threshold", "drift_triggered"], required=True)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    session_factory = configure_database(settings.database_url)
    create_tables()
    with session_factory() as session:
        result = retraining_flow(settings, session, args.policy)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
