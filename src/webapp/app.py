"""
Fire-risk MLOps dashboard.

Single FastAPI app, run with:
    uvicorn src.webapp.app:app --host 0.0.0.0 --port 8080

Navbar pages:
  /            Dashboard overview
  /ingestion   Recent raw readings per node
  /drift       Drift detection report + manual/auto retrain
  /batch       Batch inference (next predicted values per node)
  /realtime    Real-time inference (single node, latest readings)
  /metrics     Prometheus scrape target
  Grafana / Prometheus are separate services (docker-compose.yml) that
  read /metrics; this app only exposes the data, it doesn't render
  dashboards itself.

A background scheduler periodically runs the same drift check the
dashboard button triggers and, when drift is detected, automatically
fires the retrain pipeline -- "after the system has detected data
drift it should train automatically on its own" per the project brief.
The scheduler is disabled under pytest (PYTEST_CURRENT_TEST is always
set by pytest while tests run) so test runs stay fast and deterministic.
"""
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: E402

from src.db.models import init_db  # noqa: E402
from src.monitoring import drift, retrain  # noqa: E402
from src.monitoring import metrics as m  # noqa: E402
from src.webapp import services  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Fire Risk MLOps Dashboard")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

NAV_LINKS = [
    ("/", "Dashboard"),
    ("/ingestion", "Ingestion"),
    ("/drift", "Drift Detection"),
    ("/batch", "Batch Inference"),
    ("/realtime", "Real-time Inference"),
]
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")


def _ctx(request: Request, **extra):
    return {"request": request, "nav_links": NAV_LINKS, "grafana_url": GRAFANA_URL, **extra}


@app.on_event("startup")
def on_startup():
    init_db()
    _maybe_start_scheduler()


def _maybe_start_scheduler():
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("DISABLE_SCHEDULER") == "1":
        return
    from apscheduler.schedulers.background import BackgroundScheduler

    interval_minutes = int(os.environ.get("DRIFT_CHECK_INTERVAL_MINUTES", "15"))
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_drift_and_retrain, "interval", minutes=interval_minutes,
        id="drift_check", replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler


def scheduled_drift_and_retrain():
    report = drift.check_drift()
    m.DRIFT_CHECKS.inc()
    if report.get("drift_detected"):
        m.DRIFT_DETECTED.inc()
        entry = retrain.run_retrain_pipeline(trigger="auto_drift")
        m.RETRAIN_RUNS.labels(trigger="auto_drift").inc()
        if entry.get("success"):
            m.MODEL_LAST_TRAINED_TIMESTAMP.set(time.time())
        else:
            m.RETRAIN_FAILURES.inc()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(m.REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------- Dashboard
@app.get("/")
def dashboard(request: Request):
    summary = services.dashboard_summary()
    return templates.TemplateResponse(
        "dashboard.html", _ctx(request, summary=summary, title="Dashboard"),
    )


# ---------------------------------------------------------------- Ingestion
@app.get("/ingestion")
def ingestion(request: Request):
    from src.db.models import get_conn

    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM readings ORDER BY id DESC LIMIT 50"
    ).fetchall()
    node_counts = conn.execute(
        "SELECT node_id, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts "
        "FROM readings GROUP BY node_id"
    ).fetchall()
    conn.close()
    return templates.TemplateResponse(
        "ingestion.html",
        _ctx(request, readings=[dict(r) for r in rows],
             node_counts=[dict(r) for r in node_counts], title="Ingestion"),
    )


# ------------------------------------------------------------ Drift + retrain
@app.get("/drift")
def drift_page(request: Request):
    report = drift.check_drift()
    history = retrain.get_retrain_history()[::-1][:10]
    return templates.TemplateResponse(
        "drift.html", _ctx(request, report=report, history=history, title="Drift Detection"),
    )


@app.post("/api/drift/check")
def api_drift_check():
    report = drift.check_drift()
    m.DRIFT_CHECKS.inc()
    if report.get("drift_detected"):
        m.DRIFT_DETECTED.inc()
    return report


@app.post("/api/drift/retrain")
def api_retrain(trigger: str = "manual"):
    entry = retrain.run_retrain_pipeline(trigger=trigger)
    m.RETRAIN_RUNS.labels(trigger=trigger).inc()
    if entry.get("success"):
        m.MODEL_LAST_TRAINED_TIMESTAMP.set(time.time())
    else:
        m.RETRAIN_FAILURES.inc()
    return entry


# ---------------------------------------------------------------- Batch
@app.get("/batch")
def batch_page(request: Request):
    with m.INFERENCE_LATENCY_SECONDS.labels(mode="batch").time():
        results = services.batch_inference()
    for row in results:
        m.BATCH_PREDICTIONS.inc()
        if row["emergency"]:
            m.EMERGENCY_ALERTS.labels(mode="batch").inc()
        for col, pred in row["predictions"].items():
            m.LAST_PREDICTED_VALUE.labels(node_id=row["node_id"], parameter=col, mode="batch") \
                .set(pred["predicted"])
    return templates.TemplateResponse(
        "batch.html", _ctx(request, results=results, title="Batch Inference"),
    )


# ---------------------------------------------------------------- Realtime
@app.get("/realtime")
def realtime_page(request: Request, node_id: str | None = None):
    nodes = services.list_nodes()
    selected = node_id or (nodes[0] if nodes else None)
    result = None
    if selected:
        with m.INFERENCE_LATENCY_SECONDS.labels(mode="realtime").time():
            result = services.realtime_inference(selected)
        if result:
            m.REALTIME_PREDICTIONS.inc()
            if result.get("emergency"):
                m.EMERGENCY_ALERTS.labels(mode="realtime").inc()
            for col, pred in result.get("predictions", {}).items():
                m.LAST_PREDICTED_VALUE.labels(node_id=selected, parameter=col, mode="realtime") \
                    .set(pred["predicted"])
    return templates.TemplateResponse(
        "realtime.html",
        _ctx(request, nodes=nodes, selected=selected, result=result, title="Real-time Inference"),
    )
