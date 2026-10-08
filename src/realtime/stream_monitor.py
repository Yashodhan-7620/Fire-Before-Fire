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

from src.db.models import get_conn, insert_alert
from src.alerts.notifier import send_alert

POLL_SECONDS = 5
WINDOW = 12          # ~1 minute of history at 5s polling
Z_THRESHOLD = 3.0    # standard deviations considered a "sudden" jump
RATE_THRESHOLDS = {  # absolute change allowed between consecutive readings
    "temperature_c": 5.0,
    "current_a": 3.0,
    "power_w": 300.0,
    "voltage_v": 20.0,
}

history = {col: deque(maxlen=WINDOW) for col in RATE_THRESHOLDS}
last_seen_id = 0


def zscore_flag(series, new_value):
    if len(series) < 4:
        return False
    mean = sum(series) / len(series)
    var = sum((x - mean) ** 2 for x in series) / len(series)
    std = var ** 0.5
    if std == 0:
        return False
    return abs(new_value - mean) / std >= Z_THRESHOLD


def check_row(row):
    alerts = []
    for col, limit in RATE_THRESHOLDS.items():
        value = row[col]
        series = history[col]
        if series:
            delta = abs(value - series[-1])
            if delta >= limit:
                alerts.append(f"Sudden change in {col}: {series[-1]:.2f} -> {value:.2f}")
        if zscore_flag(series, value):
            alerts.append(f"Statistical outlier in {col}: {value:.2f}")
        series.append(value)
    return alerts


def poll_loop():
    global last_seen_id
    print("Real-time stream monitor started. Polling every", POLL_SECONDS, "s")
    while True:
        conn = get_conn()
        rows = conn.execute(
            "SELECT * FROM readings WHERE id > ? ORDER BY id ASC", (last_seen_id,)
        ).fetchall()
        conn.close()

        for row in rows:
            last_seen_id = row["id"]
            alerts = check_row(row)
            for reason in alerts:
                print(f"[REALTIME ALERT] node={row['node_id']} {reason}")
                insert_alert(row["node_id"], "realtime", reason, value=row["power_w"])
                send_alert(f"Realtime anomaly on {row['node_id']}: {reason}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    poll_loop()
