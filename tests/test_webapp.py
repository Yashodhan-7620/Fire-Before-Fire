import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.webapp.app import app

# Use the client as a context manager so startup events (init_db) run.
client = TestClient(app)
client.__enter__()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_dashboard_page_renders():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


def test_ingestion_page_renders():
    resp = client.get("/ingestion")
    assert resp.status_code == 200
    assert "Ingestion" in resp.text


def test_drift_page_renders():
    resp = client.get("/drift")
    assert resp.status_code == 200
    assert "Drift Detection" in resp.text


def test_batch_page_renders():
    resp = client.get("/batch")
    assert resp.status_code == 200
    assert "Batch Inference" in resp.text


def test_realtime_page_renders():
    resp = client.get("/realtime")
    assert resp.status_code == 200
    assert "Real-time Inference" in resp.text


def test_drift_check_api():
    resp = client.post("/api/drift/check")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_metrics_endpoint_exposes_prometheus_format():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "drift_checks_total" in resp.text
