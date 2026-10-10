"""
Auto-retrain orchestration.

When drift is detected (automatically by the scheduled job in
src/webapp/app.py, or manually via the dashboard's "Retrain now"
button), this module:

  1. Fetches ALL historical readings from the node database and saves
     them to data/raw_export.csv (the same file dvc.yaml's `preprocess`
     stage depends on).
  2. Runs `dvc repro` (preprocess -> train), i.e. exactly the batch
     path run_batch.bat performs -- no Airflow involved.
  3. Refreshes the drift baseline from the freshly exported data.
  4. Best-effort commits + pushes the updated data/model-tracking
     files (dvc.lock, the .dvc pointer, the new baseline) to GitHub.
     If no git remote/credentials are configured (e.g. in a sandbox),
     this step fails gracefully and is simply logged -- it never
     crashes the retrain run.

Every run (success or failure) is appended to data/retrain_log.json so
the dashboard can show a history of past retrains.
"""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.db.models import fetch_readings_df
from src.monitoring import drift as drift_mod

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CSV = ROOT / "data" / "raw_export.csv"
LOG_PATH = ROOT / "data" / "retrain_log.json"
MAX_LOG_ENTRIES = 50


def _append_log(entry: dict) -> None:
    log = []
    if LOG_PATH.exists():
        try:
            log = json.loads(LOG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            log = []
    log.append(entry)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log[-MAX_LOG_ENTRIES:], indent=2, default=str))


def get_retrain_history() -> list:
    if not LOG_PATH.exists():
        return []
    try:
        return json.loads(LOG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def fetch_all_and_save_csv() -> int:
    """Pull every reading ever recorded for every node and snapshot it
    to the CSV the batch pipeline reads from."""
    df = fetch_readings_df()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    return len(df)


def git_push_changes(message: str) -> dict:
    """Best-effort commit + push of the retrain outputs. Disabled by
    setting AUTO_GIT_PUSH=false; always non-fatal on failure."""
    if os.environ.get("AUTO_GIT_PUSH", "true").lower() != "true":
        return {"pushed": False, "reason": "AUTO_GIT_PUSH disabled"}
    try:
        subprocess.run(
            ["git", "add", "data/raw_export.csv.dvc", "dvc.lock", "data/drift_baseline.json"],
            cwd=ROOT, capture_output=True, text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
        )
        if not status.stdout.strip():
            return {"pushed": False, "reason": "no tracked changes to commit"}
        commit = subprocess.run(
            ["git", "commit", "-m", message], cwd=ROOT, capture_output=True, text=True,
        )
        if commit.returncode != 0:
            return {"pushed": False, "reason": commit.stderr.strip()[-500:]}
        push = subprocess.run(["git", "push"], cwd=ROOT, capture_output=True, text=True)
        if push.returncode != 0:
            return {"pushed": False, "reason": push.stderr.strip()[-500:]}
        return {"pushed": True}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"pushed": False, "reason": str(exc)}


def run_retrain_pipeline(trigger: str = "manual") -> dict:
    """Fetch -> CSV -> preprocess -> train (via dvc repro) -> refresh
    drift baseline -> best-effort push. Returns a log entry dict."""
    started_at = datetime.now(timezone.utc).isoformat()
    n_rows = fetch_all_and_save_csv()

    env = os.environ.copy()
    env["RAW_SOURCE"] = "csv"
    env.setdefault("RESAMPLE_FREQ", os.environ.get("RESAMPLE_FREQ", "1h"))

    result = subprocess.run(
        ["dvc", "repro", "--no-commit"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    success = result.returncode == 0

    if success:
        try:
            raw_df = pd.read_csv(RAW_CSV)
            drift_mod.save_baseline(raw_df)
        except (OSError, pd.errors.ParserError):
            pass

    git_result = (
        git_push_changes(f"Auto-retrain ({trigger}) at {started_at}")
        if success else {"pushed": False, "reason": "training failed, nothing to push"}
    )

    entry = {
        "timestamp": started_at,
        "trigger": trigger,
        "rows_used": n_rows,
        "success": success,
        "git": git_result,
        "stdout_tail": result.stdout[-1500:],
        "stderr_tail": "" if success else result.stderr[-1500:],
    }
    _append_log(entry)
    return entry
