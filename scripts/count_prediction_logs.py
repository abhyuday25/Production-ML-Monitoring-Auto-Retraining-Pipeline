from __future__ import annotations

import os

from sqlalchemy import create_engine, text


if __name__ == "__main__":
    database_url = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")
    engine = create_engine(database_url)
    with engine.connect() as connection:
        count = connection.execute(text("SELECT COUNT(*) FROM prediction_logs")).scalar_one()
    print(count)

