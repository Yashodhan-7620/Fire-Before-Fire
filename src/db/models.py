"""
SQLite schema + thin data-access layer.
Local deployment keeps this simple: SQLite is enough for one/few nodes.
Swap DB_PATH / the connection string for Postgres later without changing
the pipeline code much, since all access goes through this module.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "sensor_data.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    ts TEXT NOT NULL,              -- ISO8601 UTC timestamp
    temperature_c REAL,
    voltage_v REAL,
    current_a REAL,
    power_w REAL,
    kwh REAL
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,          -- 'realtime' or 'batch_forecast'
    reason TEXT NOT NULL,
    value REAL
);
"""


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def insert_reading(
    node_id: str, temperature_c: float, voltage_v: float,
    current_a: float, power_w: float, kwh: float,
):
    conn = get_conn()
    conn.execute(
        """INSERT INTO readings (node_id, ts, temperature_c, voltage_v, current_a, power_w, kwh)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (node_id, datetime.now(timezone.utc).isoformat(), temperature_c,
         voltage_v, current_a, power_w, kwh),
    )
    conn.commit()
    conn.close()


def insert_alert(node_id: str, source: str, reason: str, value: float):
    conn = get_conn()
    conn.execute(
        """INSERT INTO alerts (node_id, ts, source, reason, value)
           VALUES (?, ?, ?, ?, ?)""",
        (node_id, datetime.now(timezone.utc).isoformat(), source, reason, value),
    )
    conn.commit()
    conn.close()


def fetch_readings_df(since_iso: str | None = None):
    import pandas as pd
    conn = get_conn()
    query = "SELECT * FROM readings"
    params = ()
    if since_iso:
        query += " WHERE ts >= ?"
        params = (since_iso,)
    query += " ORDER BY ts ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")
