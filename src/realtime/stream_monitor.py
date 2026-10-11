"""
Real-time inference service (separate from the daily batch pipeline).

Goal: catch a SUDDEN rise/fall in temperature/current/power/voltage as
soon as it happens, without waiting for the next day's trained model.
This is a lightweight statistical detector (rate-of-change + rolling
z-score), not a heavy ML model, because it must react within seconds.

Run continuously:
    python -m src.realtime.stream_monitor
"""
import sys
import time
from collections import deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.db.models import get_conn, insert_alert  # noqa: E402
from src.alerts.notifier import send_alert  # noqa: E402

THRESHOLDS = {"temperature_c": 33.0, "power_w": 1.0, "current_a": 0.09, "voltage_v": 10.0}

POLL_SECONDS = 5
WINDOW = 12          # ~1 minute of history at 5s polling
Z_THRESHOLD = 3.0    # standard deviations considered a "sudden" jump
RATE_THRESHOLDS = {  # absolute change allowed between consecutive readings
    "temperature_c": 5.0,
    "current_a": 0.3,
    "power_w": 3.0,
    "voltage_v": 3.0,
}
# Minimum std used by the z-score so normal sensor noise on a very
# steady signal is not reported as an outlier.
MIN_STD = {
    "temperature_c": 0.5,
    "current_a": 0.02,
    "power_w": 0.15,
    "voltage_v": 0.5,
}

history = {col: deque(maxlen=WINDOW) for col in RATE_THRESHOLDS}
last_seen_id = 0


def zscore_flag(series, new_value, min_std=0.0):
    if len(series) < 4:
        return False
    mean = sum(series) / len(series)
    var = sum((x - mean) ** 2 for x in series) / len(series)
    std = max(var ** 0.5, min_std)
    if std == 0:
        return False
    return abs(new_value - mean) / std >= Z_THRESHOLD


def check_row(row):
    alerts = []  # (reason, value)
    for col, limit in RATE_THRESHOLDS.items():
        value = row[col]
        series = history[col]
        if series:
            delta = abs(value - series[-1])
            if delta >= limit:
                alerts.append((f"Sudden change in {col}: {series[-1]:.2f} -> {value:.2f}", value))
        if zscore_flag(series, value, MIN_STD[col]):
            alerts.append((f"Statistical outlier in {col}: {value:.2f}", value))
        if value >= THRESHOLDS[col]:
            alerts.append((f"{col}={value:.3f} crossed threshold {THRESHOLDS[col]}", value))
        series.append(value)
    return alerts


def poll_loop():
    global last_seen_id
    print("Real-time stream monitor started. Polling every", POLL_SECONDS, "s")
    conn = get_conn()
    latest = conn.execute("SELECT COALESCE(MAX(id), 0) FROM readings").fetchone()[0]
    conn.close()
    last_seen_id = max(0, latest - WINDOW)
    while True:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM readings WHERE id > ? ORDER BY id ASC", (last_seen_id,)
        ).fetchall()
        conn.close()

        for row in rows:
            last_seen_id = row["id"]
            alerts = check_row(row)
            for reason, val in alerts:
                print(f"[REALTIME ALERT] node={row['node_id']} {reason}")
                insert_alert(row["node_id"], "realtime", reason, value=float(val))
                send_alert(f"Realtime anomaly on {row['node_id']}: {reason}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    poll_loop()
