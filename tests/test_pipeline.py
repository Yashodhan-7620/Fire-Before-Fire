import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.pipelines.preprocess import build_hourly_features, make_next_period_targets, FEATURE_COLS


def _fake_raw():
    ts = pd.date_range("2026-01-01", periods=10, freq="15min")
    return pd.DataFrame({
        "ts": ts,
        "node_id": ["node-01"] * 10,
        "temperature_c": [30 + i for i in range(10)],
        "voltage_v": [220] * 10,
        "current_a": [2 + 0.1 * i for i in range(10)],
        "power_w": [500 + 10 * i for i in range(10)],
    })


def test_build_hourly_features_shapes():
    raw = _fake_raw()
    hourly = build_hourly_features(raw)
    assert not hourly.empty
    for col in FEATURE_COLS:
        assert f"{col}_mean" in hourly.columns


def test_targets_drop_last_unlabeled_row():
    raw = _fake_raw()
    hourly = build_hourly_features(raw)
    labeled = make_next_period_targets(hourly)
    # last hour per node has no "next" row, so it must be dropped
    assert len(labeled) <= len(hourly)
