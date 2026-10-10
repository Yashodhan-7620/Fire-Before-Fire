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
   ├────────────────────────────► Serving API (src/serving/predict_api.py)
   │                                   loads latest registered model, /predict, /drift-check
   │
   └────────────────────────────► Web dashboard (src/webapp/app.py, FastAPI + Jinja2)
                                       Ingestion view · drift detection (KS-test, src/monitoring/drift.py)
                                       → automatic retrain on drift (src/monitoring/retrain.py: fetch all
                                       history → CSV → `dvc repro` → best-effort git push) · batch +
                                       real-time inference with SHAP explanations (src/monitoring/explain.py)
                                       and emergency/recommendation banners · /metrics for Prometheus,
                                       visualized in Grafana (monitoring/)

Orchestration: run_batch.bat (Windows Task Scheduler, daily) via `dvc repro`,
               and the web dashboard's own drift-triggered auto-retrain
               -- src/dags/batch_training_dag.py (Airflow) is an optional
                  alternative, not required and not used in CI/CD
Data versioning: dvc.yaml (DVC)
Containers: Dockerfile.api + docker-compose.yml (ingestion/serving/webapp
            APIs, realtime monitor, mlflow UI, Prometheus, Grafana)
CI/CD: .github/workflows/ci-cd.yml -- lint + unit tests (incl. the web
       dashboard) + a DVC batch-pipeline dry run (CI, Airflow-free), then a
       compose-validated, container-smoke-tested image build (CD, local
       deployment target)
```

## How this maps to the syllabus (for your report)

| Unit | Concept | Where it's used |
|---|---|---|
| I | ML lifecycle, versioning, reproducibility | Overall pipeline structure; DVC + MLflow |
| II | Data versioning, experiment tracking, feature engineering, model registry | `dvc.yaml`, `train.py` (MLflow tracking + registry) |
| III | Pipelines/DAGs, workflow automation | `src/dags/batch_training_dag.py` (Airflow) |
| IV | Model deployment, REST API, batch vs real-time inference, drift basics | `predict_api.py` (`/predict`, `/drift-check`), `stream_monitor.py`, `src/webapp/app.py` |
| V/VI | Monitoring, alerting, explainability, industry application (predictive maintenance) | `alerts/notifier.py`, threshold + forecast alerts, `src/monitoring/` (drift, auto-retrain, SHAP), Prometheus + Grafana (`monitoring/`) |

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
src/monitoring/drift.py         KS-test drift detection (recent readings vs. training baseline)
src/monitoring/retrain.py       Auto-retrain: fetch all history → CSV → `dvc repro` → best-effort git push
src/monitoring/explain.py       SHAP explanations → human-readable "what to focus on" recommendations
src/monitoring/metrics.py       Prometheus counters/gauges/histograms for the dashboard
src/webapp/app.py               FastAPI dashboard: ingestion view, drift detection, batch/real-time
                                 inference, /metrics; background scheduler auto-retrains on drift
src/webapp/services.py          Loads the latest MLflow forecasters, runs batch/real-time inference
src/webapp/templates/           Jinja2 templates (navbar + pages), styled with the project color palette
dvc.yaml                        DVC pipeline for reproducible preprocess/train
Dockerfile.api / docker-compose.yml   Local containerized deployment (adds webapp, prometheus, grafana)
monitoring/prometheus.yml       Prometheus scrape config (targets the webapp's /metrics)
monitoring/grafana/             Provisioned Grafana datasource + starter dashboard
.github/workflows/ci-cd.yml     CI (lint + tests, incl. the dashboard + DVC batch-pipeline dry run) +
                                 CD (compose validate + image build + container
                                 smoke test), local-deployment scope, Airflow-free
scripts/generate_synthetic_data.py  Synthetic sensor data generator used by CI
                                     to dry-run the batch pipeline (also usable
                                     for local testing)
tests/test_pipeline.py          Unit tests for preprocessing
tests/test_webapp.py            Smoke tests for every dashboard page + the drift/metrics APIs
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
(`:8001`), the **web dashboard** (`:8080`, see section 6), the
real-time monitor, the MLflow UI (`:5000`), **Prometheus** (`:9090`)
and **Grafana** (`:3000`, default login `admin` / `admin`, with the
dashboard in `monitoring/grafana/dashboards/` provisioned automatically).
Run the batch job directly inside the compose network with:
```bash
docker compose run --rm serving-api python -m src.pipelines.train
```

## 6. Web dashboard — ingestion, drift detection, auto-retrain, inference, SHAP, monitoring

A single FastAPI app (`src/webapp/app.py`) ties every feature above
together behind one navbar, styled with the project's color palette
(maroon `#6D0808` / near-black `#2D0000` / sage `#757D6F` / cream
`#EEEAD7`):

```bash
uvicorn src.webapp.app:app --host 0.0.0.0 --port 8080   # or: docker compose up webapp
```

| Page | What it does |
|---|---|
| **Dashboard** (`/`) | Readings/node counts, last alert, models loaded, last trained time. |
| **Ingestion** (`/ingestion`) | Per-node reading counts/date-range and the latest 50 raw readings. |
| **Drift Detection** (`/drift`) | Runs a **two-sample Kolmogorov-Smirnov test** (`src/monitoring/drift.py`) per raw feature, comparing a recent window of readings to the distribution saved as the baseline the last time the model trained. "Run drift check now" and "Retrain now" buttons trigger the same checks/pipeline the background scheduler runs automatically every `DRIFT_CHECK_INTERVAL_MINUTES` (default 15) — **when drift is detected, the system retrains itself with no manual step**: it fetches *all* historical readings from the database, saves them to `data/raw_export.csv`, runs `dvc repro` (the exact `preprocess → train` path `run_batch.bat` runs), refreshes the drift baseline, and then best-effort commits + pushes the updated `dvc.lock`/baseline to GitHub (`src/monitoring/retrain.py`; set `AUTO_GIT_PUSH=false` to disable the push, e.g. in CI/sandboxes without git credentials). Every run is logged to `data/retrain_log.json` and shown as history on this page. |
| **Batch Inference** (`/batch`) | Next-period predictions for every node from the latest committed features (`data/features.parquet`), using every `forecaster_<param>` model from the most recent MLflow run. Crossing a threshold (`src/pipelines/train.py::THRESHOLDS`) shows an **emergency banner**. |
| **Real-time Inference** (`/realtime`) | Same prediction/threshold/emergency flow, but the feature vector is built on-the-fly from each node's most recent live readings instead of the precomputed feature table — so you get a live "predicted next value" per node. |

**Explainability:** whenever a threshold is crossed (batch or
real-time), `src/monitoring/explain.py` runs `shap.TreeExplainer` on
the forecaster that crossed it, picks the strongest contributing
feature, and turns it into a plain-English **recommendation** ("Inspect
wiring, connectors and the connected load...", etc.) shown right under
the emergency banner — "what aspect to focus on."

**Monitoring:** the app exposes Prometheus metrics at `/metrics`
(`src/monitoring/metrics.py`) — drift checks/events, retrain
runs/failures, batch/real-time predictions served, emergency alerts,
last predicted value per node/parameter, inference latency, and last
training time. `monitoring/prometheus.yml` scrapes it, and Grafana is
pre-provisioned (`monitoring/grafana/`) with a starter dashboard
visualizing all of the above — open it from the dashboard's top-right
**📊 Grafana** link (`http://localhost:3000`).

## 7. CI/CD

Every push/PR to `main` triggers `.github/workflows/ci-cd.yml`, which
validates the project end to end — lint, unit tests, a real batch-pipeline
dry run, and a container smoke test — before anything is considered
"deployable." **Airflow is not used or required anywhere in this
pipeline**: CI/CD validates the same `run_batch.bat` / `dvc repro` batch
path described in section 2/4, not the optional Airflow DAG from section 3.

Two jobs run in parallel first:

- **`ci` (Lint & Test):** `flake8` on `src`, then `pytest` for the unit
  tests in `tests/` (including `tests/test_webapp.py`, which exercises
  every dashboard page plus the drift-check and `/metrics` APIs).
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
  `ingestion-api`, `serving-api` **and `webapp`** containers with
  `docker compose up` and polls all three `/health` endpoints before
  tearing them down. This runs on every push/PR (not just after
  merging), so "deployability" is checked before code lands on `main`,
  not only after. Deployment is intentionally kept **local-only** for
  this project — CD stops once it has proven every image builds *and*
  runs correctly; it does not push to
  a registry or deploy to any server. Add a `docker push` step with
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
- Web dashboard env vars (all optional, sensible defaults): `GRAFANA_URL`
  (link in the navbar, default `http://localhost:3000`), `AUTO_GIT_PUSH`
  (`true`/`false`, default `true` — auto-retrain's commit+push step is
  always best-effort and never crashes the app if git isn't configured),
  `DRIFT_CHECK_INTERVAL_MINUTES` (default `15`), `DISABLE_SCHEDULER=1`
  to turn off the background drift-check job entirely (it's already
  disabled automatically under `pytest`).
