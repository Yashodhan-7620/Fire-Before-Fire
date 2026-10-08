# Fire / Short-Circuit Early-Warning System — IoT + MLOps

Detects anomalies in **voltage, current, power, and heat** from a sensor
node to predict and prevent short circuits, fires, and to schedule
maintenance before failure.

Two detection paths, as specified:

1. **Batch path (end of day):** all readings from the day are used to
   retrain a forecasting model that predicts tomorrow's parameters; if
   a prediction crosses a threshold, a maintenance alert fires.
2. **Real-time path (continuous):** a lightweight statistical detector
   watches every incoming reading for a *sudden* rise or fall and
   alerts immediately, independent of the trained model.

## Architecture

```
ESP32 sensor node (firmware/sensor_node.ino)
   │  DS18B20 temp, EmonLib volt/current, LCD+RGB status
   │  HTTP POST every 5s
   ▼
Ingestion API (src/ingestion/api.py) ──► SQLite (src/db/models.py)
   │                                         │
   │                                         ├──► Real-time monitor (src/realtime/stream_monitor.py)
   │                                         │       polls new rows, z-score + rate-of-change → alerts
   │                                         │
   │                                         ▼
   │                              Preprocess (src/pipelines/preprocess.py)
   │                                         │  hourly features + next-period labels
   │                                         ▼
   │                              Train (src/pipelines/train.py)
   │                                 IsolationForest (anomaly) +
   │                                 RandomForestRegressor per parameter (forecast)
   │                                 → logged + registered in MLflow
   │                                         │
   │                                         ▼
   └────────────────────────────► Serving API (src/serving/predict_api.py)
                                       loads latest registered model, /predict, /drift-check

Orchestration: run_batch.bat (Windows Task Scheduler, daily) via `dvc repro`
               -- src/dags/batch_training_dag.py (Airflow) is an optional
                  alternative, not required and not used in CI/CD
Data versioning: dvc.yaml (DVC)
Containers: Dockerfile.api + docker-compose.yml
CI/CD: .github/workflows/ci-cd.yml -- lint + unit tests + a DVC batch-
       pipeline dry run (CI, Airflow-free), then a compose-validated,
       container-smoke-tested image build (CD, local deployment target)
```

## How this maps to the syllabus (for your report)

| Unit | Concept | Where it's used |
|---|---|---|
| I | ML lifecycle, versioning, reproducibility | Overall pipeline structure; DVC + MLflow |
| II | Data versioning, experiment tracking, feature engineering, model registry | `dvc.yaml`, `train.py` (MLflow tracking + registry) |
| III | Pipelines/DAGs, workflow automation | `src/dags/batch_training_dag.py` (Airflow) |
| IV | Model deployment, REST API, batch vs real-time inference, drift basics | `predict_api.py` (`/predict`, `/drift-check`), `stream_monitor.py` |
| V/VI | Monitoring, alerting, industry application (predictive maintenance) | `alerts/notifier.py`, threshold + forecast alerts |

## Repo layout

```
firmware/sensor_node.ino        ESP32 firmware (merged + debugged, LCD + RGB added)
src/ingestion/api.py            Receives sensor POSTs, writes to DB
src/db/models.py                SQLite schema + access helpers
src/pipelines/preprocess.py     Raw readings → hourly features + labels
src/pipelines/train.py          Trains + logs to MLflow, registers model, checks thresholds
src/serving/predict_api.py      Loads registered model, serves predictions
src/realtime/stream_monitor.py  Continuous sudden rise/fall detector
src/alerts/notifier.py          Alert delivery (console + optional webhook)
src/dags/batch_training_dag.py  Optional Airflow DAG for the daily batch job (not used in CI/CD)
dvc.yaml                        DVC pipeline for reproducible preprocess/train
Dockerfile.api / docker-compose.yml   Local containerized deployment
.github/workflows/ci-cd.yml     CI (lint + tests + DVC batch-pipeline dry run) +
                                 CD (compose validate + image build + container
                                 smoke test), local-deployment scope, Airflow-free
scripts/generate_synthetic_data.py  Synthetic sensor data generator used by CI
                                     to dry-run the batch pipeline (also usable
                                     for local testing)
tests/test_pipeline.py          Unit tests for preprocessing
```

## 1. Firmware setup

1. Open `firmware/sensor_node.ino` in Arduino IDE with the **ESP32
   board package** installed (this code needs Wi-Fi — see the note at
   the top of the file about why it can't run on a plain Uno).
2. Install libraries via Library Manager: `EmonLib`, `OneWire`,
   `DallasTemperature`, `Blynk`, `LiquidCrystal_I2C`.
3. Wiring (as coded):
   - DS18B20 data → GPIO 4 (with a 4.7kΩ pull-up to 3.3V)
   - Voltage sensor output → GPIO 35
   - Current sensor (SCT) output → GPIO 34
   - RGB LED → GPIO 25 (R), 26 (G), 27 (B), through current-limiting resistors
   - 16x2 I2C LCD → SDA=21, SCL=22
4. Set your real `ssid`/`pass`, Blynk template/auth values, and
   `INGEST_URL` (your PC's LAN IP + port 8000) in the sketch.
5. Flash it. Open Serial Monitor at 115200 baud to confirm readings.

## 2. Local pipeline setup (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Initialize the database
python -m src.db.models

# 2. Start the ingestion API (point the ESP32's INGEST_URL here)
uvicorn src.ingestion.api:app --host 127.0.0.1 --port 8000
uvicorn src.ingestion.api:app --host 0.0.0.0 --port 8000
/docs /health
# 3. In another terminal: start the real-time monitor
python -m src.realtime.stream_monitor

# 4. Let readings accumulate for a while, then run the batch pipeline manually
python -m src.pipelines.preprocess
set RESAMPLE_FREQ=2min && python -m src.pipelines.preprocess
python -m src.pipelines.train

# 5. Inspect experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000

# 6. Serve the latest registered model
uvicorn src.serving.predict_api:app --host 0.0.0.0 --port 8001
curl http://localhost:8001/health
curl http://localhost:8001/drift-check
```

## 3. Automate the daily batch job with Airflow

```bash
export AIRFLOW_HOME=~/airflow
airflow db init
cp src/dags/batch_training_dag.py ~/airflow/dags/
airflow webserver -p 8080 &
airflow scheduler &
```
The DAG (`fire_risk_batch_training`) runs `preprocess → train` daily
at 23:30 — this is the automated "end of day" batch cycle from the
brief. No manual re-running needed once it's registered.

## 4. Data versioning with DVC

```bash
dvc init
dvc remote add -d local_remote ./dvc_store
python -c "from src.db.models import fetch_readings_df; fetch_readings_df().to_csv('data/raw_export.csv', index=False)"
dvc add data/raw_export.csv
dvc repro       # runs preprocess -> train per dvc.yaml
dvc push
```

## 5. Full local deployment with Docker Compose

```bash
cp .env.example .env       # optionally set ALERT_WEBHOOK_URL
docker compose up --build
```
This starts, on your machine: ingestion API (`:8000`), serving API
(`:8001`), the real-time monitor, and the MLflow UI (`:5000`). Run the
batch job inside the compose network with:
```bash
docker compose run --rm serving-api python -m src.pipelines.train
```

## 6. CI/CD

Every push/PR to `main` triggers `.github/workflows/ci-cd.yml`, which
validates the project end to end — lint, unit tests, a real batch-pipeline
dry run, and a container smoke test — before anything is considered
"deployable." **Airflow is not used or required anywhere in this
pipeline**: CI/CD validates the same `run_batch.bat` / `dvc repro` batch
path described in section 2/4, not the optional Airflow DAG from section 3.

Two jobs run in parallel first:

- **`ci` (Lint & Test):** `flake8` on `src`, then `pytest` for the unit
  tests in `tests/`.
- **`batch-pipeline` (Validate batch pipeline):** generates a small,
  synthetic, fully offline sensor dataset
  (`scripts/generate_synthetic_data.py`), initializes the SQLite schema,
  then runs **`dvc repro`** — the real, committed `dvc.yaml` pipeline
  (`preprocess` → `train`), the same stages `run_batch.bat` runs daily —
  against that synthetic data, and asserts `data/features.parquet` and
  `mlruns/` come out the other end. This fails the build if the batch
  training path itself is broken, not just if a unit test is.

Only once **both** of those pass does the final job run:

- **`cd` (Build and smoke-test deployment image):** validates
  `docker-compose.yml` with `docker compose config`, builds the Docker
  image with `Dockerfile.api`, then actually brings up the
  `ingestion-api` and `serving-api` containers with `docker compose up`
  and polls both `/health` endpoints before tearing them down. Deployment
  is intentionally kept **local-only** for this project — CD stops once
  it has proven the image builds *and* runs correctly; it does not push
  to a registry or deploy to any server. Add a `docker push` step with
  registry secrets if you want to take it further.

## Alert thresholds

Defined in `src/pipelines/train.py::THRESHOLDS` (batch/forecast) and
`src/realtime/stream_monitor.py::RATE_THRESHOLDS` (real-time
rise/fall). Tune both to your actual sensor calibration before relying
on them — the shipped numbers are placeholders.

## Notes / assumptions made

- "RGB module + LCD" was implemented as a separate RGB status LED
  (traffic-light style: green/amber/red) plus a standard 16x2 I2C text
  LCD, since that's the combination that best matches the "alert
  humans" goal — swap in a true single RGB-backlit LCD driver if
  that's what you actually have.
- Local deployment uses SQLite + local MLflow (SQLite backend) to keep
  everything runnable without any cloud account; swapping to
  Postgres/S3-backed MLflow later only touches `src/db/models.py` and
  `MLFLOW_TRACKING_URI`.
