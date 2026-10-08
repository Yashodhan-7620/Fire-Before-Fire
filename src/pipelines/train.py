"""
Batch training pipeline. Runs once a day (triggered by the Airflow DAG
in src/dags/batch_training_dag.py, or manually / via cron for local
testing without Airflow).

What it trains, and why two models:
  1. Forecaster (RandomForestRegressor, one per parameter): predicts
     next hour's temperature / voltage / current / power, so we can
     flag "tomorrow this node is predicted to cross the threshold"
     ahead of time and schedule maintenance.
  2. Anomaly scorer (IsolationForest): scores how unusual each hour's
     feature vector is, independent of thresholds, to catch abnormal
     *combinations* of readings, not just single crossed thresholds.

Everything is logged to MLflow (params, metrics, the model itself),
and the forecaster is registered to the MLflow Model Registry under
"fire_risk_forecaster" so the serving API always loads whatever the
registry marks as the current version -- that's the "model
deployment" hookup between training and serving.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.pipelines.preprocess import FEATURE_COLS
from src.db.models import insert_alert

FEATURES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "features.parquet"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "fire_risk_forecasting"
MODEL_NAME = "fire_risk_forecaster"

THRESHOLDS = {
    "temperature_c": 55.0,
    "power_w": 1500.0,
    "current_a": 10.0,
    "voltage_v": 280.0,
}

FEATURE_INPUT_COLS = [f"{c}_mean" for c in FEATURE_COLS] + \
                      [f"{c}_std" for c in FEATURE_COLS] + \
                      [f"{c}_max" for c in FEATURE_COLS]


def train_and_log(df: pd.DataFrame):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X = df[FEATURE_INPUT_COLS].fillna(0)

    with mlflow.start_run(run_name="daily_batch_train") as run:
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("features", FEATURE_INPUT_COLS)

        # --- Anomaly scorer ---
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        iso.fit(X)
        mlflow.sklearn.log_model(iso, "anomaly_model")

        # --- Per-parameter next-hour forecaster ---
        forecasters = {}
        for target_col in FEATURE_COLS:
            y = df[f"target_{target_col}"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
            model.fit(X_train, y_train)
            mae = mean_absolute_error(y_test, model.predict(X_test))
            mlflow.log_metric(f"mae_{target_col}", mae)
            forecasters[target_col] = model
            mlflow.sklearn.log_model(model, f"forecaster_{target_col}")

        # Register the temperature forecaster as the headline model version
        # (serving loads all four via mlflow artifacts of this run; see
        # src/serving/predict_api.py).
        model_uri = f"runs:/{run.info.run_id}/forecaster_temperature_c"
        mlflow.register_model(model_uri, MODEL_NAME)

        # --- Predict tomorrow and raise a maintenance alert if it crosses threshold ---
        latest = df.sort_values("ts").groupby("node_id").tail(1)
        latest_inputs = latest[FEATURE_INPUT_COLS].fillna(0).astype(float)
        for position, (_, row) in enumerate(latest.iterrows()):
            x_row = latest_inputs.iloc[[position]]
            for col, model in forecasters.items():
                pred = model.predict(x_row)[0]
                if pred >= THRESHOLDS[col]:
                    reason = f"Predicted next-period {col}={pred:.2f} exceeds threshold {THRESHOLDS[col]}"
                    print(f"[BATCH FORECAST ALERT] node={row['node_id']} {reason}")
                    insert_alert(row["node_id"], "batch_forecast", reason, value=float(pred))

        print(f"MLflow run: {run.info.run_id} — model registered as '{MODEL_NAME}'")
        return run.info.run_id


def main():
    if not FEATURES_PATH.exists():
        print(f"{FEATURES_PATH} not found — run `python -m src.pipelines.preprocess` first.")
        return
    df = pd.read_parquet(FEATURES_PATH)
    if df.empty or len(df) < 10:
        print(f"Not enough data yet to train ({len(df)} rows, need >= 10). Skipping.")
        return
    train_and_log(df)


if __name__ == "__main__":
    main()