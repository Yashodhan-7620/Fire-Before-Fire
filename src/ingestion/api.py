"""
Data intake service.
The ESP32 firmware POSTs a JSON reading here every ~5s. This is the
"data collection" stage of the MLOps lifecycle: every reading is
persisted immediately and untouched (raw), so preprocessing always
starts from a full, replayable history.

Run:
    uvicorn src.ingestion.api:app --host 0.0.0.0 --port 8000
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from src.db.models import init_db, insert_reading

app = FastAPI(title="Sensor Ingestion API")
init_db()


class Reading(BaseModel):
    node_id: str
    temperature_c: float
    voltage_v: float
    current_a: float
    power_w: float
    kwh: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(reading: Reading):
    insert_reading(
        node_id=reading.node_id,
        temperature_c=reading.temperature_c,
        voltage_v=reading.voltage_v,
        current_a=reading.current_a,
        power_w=reading.power_w,
        kwh=reading.kwh,
    )
    return {"status": "stored"}
