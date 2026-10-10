"""
Business logic for the web dashboard: loading the latest trained
forecasters straight from MLflow, and running batch / real-time
inference (prediction + threshold check + SHAP explanation +
recommendation) against them.
"""
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.db.models import get_conn
from src.monitoring import explain
from src.pipelines.preprocess import FEATURE_COLS
from src.pipelines.train import (
    EXPERIMENT_NAME,
    FEATURE_INPUT_COLS,
    MLFLOW_TRACKING_URI,
    THRESHOLDS,
)

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = ROOT / "data" / "features.parquet"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_cache = {"run_id": None, "models": {}, "trained_at": None}


def get_forecasters(force_reload: bool = False):
    """Loads the forecaster_<col> models logged by the most recent run
    in the training experiment (every parameter, not just the single
    one registered to the model registry), with a tiny in-process
    cache keyed by run id."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None, {}, None
        runs = client.search_runs(
            [experiment.experiment_id], order_by=["start_time DESC"], max_results=1,
        )
        if not runs:
            return None, {}, None
        run = runs[0]
        run_id = run.info.run_id
        if not force_reload and _cache["run_id"] == run_id:
            return run_id, _cache["models"], _cache["trained_at"]

        models = {}
        for col in FEATURE_COLS:
            try:
                models[col] = mlflow.sklearn.load_model(f"runs:/{run_id}/forecaster_{col}")
            except Exception:  # noqa: BLE001 - tolerate partially-logged runs
                continue
        trained_at = run.info.end_time or run.info.start_time
        _cache.update(run_id=run_id, models=models, trained_at=trained_at)
        return run_id, models, trained_at
    except Exception:  # noqa: BLE001 - no MLflow data yet is a valid state
        return None, {}, None


def _predict_row(models: dict, x_row: pd.DataFrame) -> dict:
    """Runs every available forecaster against one feature row and
    returns {col: {predicted, threshold, crossed}}, whether any
    threshold was crossed, and the top SHAP contributions per param."""
    predictions = {}
    contributions_by_param = {}
    emergency = False
    for col, model in models.items():
        pred = float(model.predict(x_row)[0])
        crossed = pred >= THRESHOLDS[col]
        predictions[col] = {
            "predicted": round(pred, 3),
            "threshold": THRESHOLDS[col],
            "crossed": crossed,
        }
        if crossed:
            emergency = True
            try:
                contributions_by_param[col] = explain.explain_row(model, x_row)[:3]
            except Exception:  # noqa: BLE001 - SHAP is best-effort for display
                contributions_by_param[col] = []
    recommendation = explain.build_recommendation(contributions_by_param) if emergency else None
    return {
        "predictions": predictions,
        "emergency": emergency,
        "recommendation": recommendation,
        "contributions": contributions_by_param,
    }


def list_nodes() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT node_id FROM readings ORDER BY node_id").fetchall()
    conn.close()
    return [r["node_id"] for r in rows]


def batch_inference() -> list:
    """Predicts the next period's values for every node using the
    latest hourly feature row produced by the committed batch
    pipeline (data/features.parquet)."""
    if not FEATURES_PATH.exists():
        return []
    df = pd.read_parquet(FEATURES_PATH)
    if df.empty:
        return []
    _, models, _ = get_forecasters()
    if not models:
        return []

    latest = df.sort_values("ts").groupby("node_id").tail(1)
    results = []
    for _, row in latest.iterrows():
        x_row = row[FEATURE_INPUT_COLS].fillna(0).to_frame().T.astype(float)
        outcome = _predict_row(models, x_row)
        results.append({
            "node_id": row["node_id"],
            "ts": str(row["ts"]),
            **outcome,
        })
    return results


def realtime_inference(node_id: str, window: int = 12) -> dict | None:
    """Builds an on-the-fly feature vector from the most recent raw
    readings for one node (mean/std/max over the last `window`
    readings, mirroring how the batch pipeline engineers features),
    then predicts the next value with the same trained forecasters."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM readings WHERE node_id = ? ORDER BY id DESC LIMIT ?",
        (node_id, window),
    ).fetchall()
    conn.close()
    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    feature_values = {}
    for col in FEATURE_COLS:
        feature_values[f"{col}_mean"] = df[col].mean()
        feature_values[f"{col}_std"] = df[col].std() if len(df) > 1 else 0.0
        feature_values[f"{col}_max"] = df[col].max()
    x_row = pd.DataFrame([feature_values])[FEATURE_INPUT_COLS].fillna(0).astype(float)

    _, models, _ = get_forecasters()
    if not models:
        return {
            "node_id": node_id, "latest_ts": str(df["ts"].iloc[0]),
            "predictions": {}, "emergency": False, "recommendation": None,
            "contributions": {}, "status": "no_model_trained_yet",
        }

    outcome = _predict_row(models, x_row)
    return {
        "node_id": node_id,
        "latest_ts": str(df["ts"].iloc[0]),
        "samples_used": len(df),
        "status": "ok",
        **outcome,
    }


def dashboard_summary() -> dict:
    conn = get_conn()
    reading_count = conn.execute("SELECT COUNT(*) AS c FROM readings").fetchone()["c"]
    node_count = conn.execute("SELECT COUNT(DISTINCT node_id) AS c FROM readings").fetchone()["c"]
    last_alert = conn.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    _, models, trained_at = get_forecasters()
    trained_at_str = (
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(trained_at / 1000))
        if trained_at else "no model trained yet"
    )
    return {
        "reading_count": reading_count,
        "node_count": node_count,
        "last_alert": dict(last_alert) if last_alert else None,
        "models_loaded": len(models),
        "last_trained_at": trained_at_str,
    }
