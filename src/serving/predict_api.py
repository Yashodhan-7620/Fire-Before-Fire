"""
Model deployment / serving stage (local deployment target).

Loads the latest version of "fire_risk_forecaster" from the MLflow
Model Registry (whatever the batch pipeline registered most recently)
and exposes it as a REST API, per the Unit-IV syllabus content
(Model Deployment, REST API, batch vs real-time inference).

Run:
    uvicorn src.serving.predict_api:app --host 0.0.0.0 --port 8001
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

from src.pipelines.train import MLFLOW_TRACKING_URI, MODEL_NAME, THRESHOLDS, FEATURE_INPUT_COLS

app = FastAPI(title="Fire Risk Forecaster - Serving API")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model = None


def get_model():
    global _model
    if _model is None:
        # "models:/<name>/latest" always points at the newest registered
        # version, so redeploying just means the batch job registering
        # a new version -- no code change here.
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    return _model


class FeatureVector(BaseModel):
    values: dict  # e.g. {"temperature_c_mean": 42.1, "temperature_c_std": 1.2, ...}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
def predict(payload: FeatureVector):
    model = get_model()
    row = [[payload.values.get(col, 0.0) for col in FEATURE_INPUT_COLS]]
    prediction = float(model.predict(row)[0])
    at_risk = prediction >= THRESHOLDS["temperature_c"]
    return {"predicted_next_period_temperature_c": prediction, "threshold_exceeded": at_risk}


@app.get("/drift-check")
def drift_check():
    """Very simple drift signal: compares the mean of the last day's
    readings to the historical mean. A real system would use a proper
    test (KS-test / PSI); this keeps the local-deployment scope simple
    while still demonstrating the concept from Unit-IV/V."""
    import pandas as pd
    from src.db.models import fetch_readings_df

    df = fetch_readings_df()
    if df.empty:
        return {"status": "no_data"}
    df["ts"] = pd.to_datetime(df["ts"])
    recent = df[df["ts"] >= df["ts"].max() - pd.Timedelta(days=1)]
    drift = {}
    for col in ["temperature_c", "voltage_v", "current_a", "power_w"]:
        overall_mean = df[col].mean()
        recent_mean = recent[col].mean()
        drift[col] = {
            "overall_mean": round(overall_mean, 3),
            "recent_mean": round(recent_mean, 3),
            "pct_shift": round(100 * (recent_mean - overall_mean) / overall_mean, 2) if overall_mean else None,
        }
    return drift
