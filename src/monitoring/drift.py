"""
Data drift detection.

Compares the distribution of each raw sensor feature in a recent window
of readings against a "baseline" snapshot captured the last time the
model was trained. Uses a two-sample Kolmogorov-Smirnov test per
feature (a real statistical test, unlike the naive mean-shift check in
src/serving/predict_api.py::drift_check, which this module supersedes
for the dashboard while that endpoint stays as a lightweight example).

The baseline is persisted to data/drift_baseline.json so it survives
process restarts, and is refreshed every time a (re)training run
completes -- see src/monitoring/retrain.py.
"""
import json
from pathlib import Path

import pandas as pd
from scipy import stats

from src.db.models import fetch_readings_df
from src.pipelines.preprocess import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = ROOT / "data" / "drift_baseline.json"

# p-value below this -> that single feature is considered drifted.
P_VALUE_THRESHOLD = 0.05
# How many of the 4 features must drift before we call it "drift detected"
# overall (and, in the scheduled job, trigger an automatic retrain).
MIN_DRIFTED_FEATURES = 2
# How far back "recent" looks when comparing against the baseline.
RECENT_WINDOW_HOURS = 24
# Only keep this many of the most recent baseline samples per feature,
# to keep the JSON file small.
MAX_BASELINE_SAMPLES = 1000


def save_baseline(df: pd.DataFrame) -> None:
    """Persist a sample of each feature's distribution as the new
    reference baseline. Called right after a successful training run."""
    if df.empty:
        return
    baseline = {
        col: df[col].dropna().tail(MAX_BASELINE_SAMPLES).round(4).tolist()
        for col in FEATURE_COLS if col in df.columns
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline))


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    try:
        return json.loads(BASELINE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check_drift(recent_hours: int = RECENT_WINDOW_HOURS) -> dict:
    """Returns a report describing whether each raw feature's recent
    distribution has drifted away from the training-time baseline."""
    df = fetch_readings_df()
    if df.empty:
        return {"status": "no_data", "drift_detected": False, "features": {}}

    df["ts"] = pd.to_datetime(df["ts"], format="mixed", utc=True).dt.tz_localize(None)
    recent = df[df["ts"] >= df["ts"].max() - pd.Timedelta(hours=recent_hours)]

    baseline = load_baseline()
    if baseline is None:
        # First run ever: nothing to compare against yet, so establish
        # the current data as the reference and report "no drift".
        save_baseline(df)
        return {"status": "baseline_established", "drift_detected": False, "features": {}}

    features = {}
    drifted_count = 0
    for col in FEATURE_COLS:
        baseline_vals = baseline.get(col, [])
        recent_vals = recent[col].dropna().tolist() if col in recent.columns else []
        if len(baseline_vals) < 10 or len(recent_vals) < 10:
            features[col] = {"status": "insufficient_data"}
            continue
        statistic, p_value = stats.ks_2samp(baseline_vals, recent_vals)
        drifted = bool(p_value < P_VALUE_THRESHOLD)
        if drifted:
            drifted_count += 1
        features[col] = {
            "status": "ok",
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted,
            "baseline_mean": round(sum(baseline_vals) / len(baseline_vals), 3),
            "recent_mean": round(sum(recent_vals) / len(recent_vals), 3),
            "baseline_samples": len(baseline_vals),
            "recent_samples": len(recent_vals),
        }

    return {
        "status": "checked",
        "drift_detected": drifted_count >= MIN_DRIFTED_FEATURES,
        "drifted_feature_count": drifted_count,
        "features": features,
        "recent_rows": int(len(recent)),
    }
