"""
Preprocessing stage of the batch pipeline.
Turns raw per-reading rows into a daily/hourly feature table used for
both (a) training the next-day forecaster and (b) fitting the
anomaly-scoring model. Kept as a standalone, importable function so
Airflow (or any orchestrator) can call it as one task.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import pandas as pd
from src.db.models import fetch_readings_df

FEATURE_COLS = ["temperature_c", "voltage_v", "current_a", "power_w"]

# Production default is hourly buckets. For local testing with only a
# few minutes of ESP32 data, override before running, e.g. on Windows:
#   set RESAMPLE_FREQ=2min && python -m src.pipelines.preprocess
# or on Mac/Linux:
#   RESAMPLE_FREQ=2min python -m src.pipelines.preprocess
RESAMPLE_FREQ = os.environ.get("RESAMPLE_FREQ", "1h")

_RAW_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw_export.csv",
)


def load_raw() -> pd.DataFrame:
    """
    DVC's 'preprocess' stage lists data/raw_export.csv as its dependency
    (that's the file DVC hashes to decide whether to re-run this stage),
    so when that file exists, we must actually read FROM it -- otherwise
    DVC's caching decision and the data used here disagree, and dvc repro
    will wrongly reuse a stale cached result.

    Outside of `dvc repro` (plain manual runs), if no snapshot has been
    exported yet, fall back to reading live from the database.
    """
    source = os.environ.get("RAW_SOURCE", "auto")
    if source == "csv" or (source == "auto" and os.path.exists(_RAW_CSV_PATH)):
        df = pd.read_csv(_RAW_CSV_PATH)
    else:
        df = fetch_readings_df()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to hourly mean/std/max per node -- this is what the
    model actually trains on, not raw noisy per-reading values."""
    if df.empty:
        return df
    df = df.set_index("ts")
    grouped = df.groupby("node_id")

    frames = []
    for node_id, g in grouped:
        agg = g[FEATURE_COLS].resample(RESAMPLE_FREQ).agg(["mean", "std", "max", "min"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg["node_id"] = node_id
        agg = agg.dropna(subset=[f"{c}_mean" for c in FEATURE_COLS])
        frames.append(agg.reset_index())

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_next_period_targets(features: pd.DataFrame) -> pd.DataFrame:
    """Shift each node's own future mean back by one row so every row has
    a label: 'what were temp/voltage/current/power in the NEXT hour'."""
    if features.empty:
        return features
    features = features.sort_values(["node_id", "ts"])
    for col in FEATURE_COLS:
        features[f"target_{col}"] = features.groupby("node_id")[f"{col}_mean"].shift(-1)
    return features.dropna(subset=[f"target_{c}" for c in FEATURE_COLS])


def run(save_path: str = "data/features.parquet") -> pd.DataFrame:
    raw = load_raw()
    hourly = build_hourly_features(raw)
    labeled = make_next_period_targets(hourly)
    out_path = Path(__file__).resolve().parent.parent.parent / save_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out_path, index=False)
    print(f"Preprocessing done: {len(labeled)} labeled rows -> {out_path}")
    return labeled


if __name__ == "__main__":
    run()