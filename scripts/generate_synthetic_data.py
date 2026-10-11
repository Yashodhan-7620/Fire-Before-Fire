"""
Generates a small synthetic `data/raw_export.csv` with the same schema
the real pipeline expects (see src/pipelines/preprocess.py / fetch_readings_df):
    ts, node_id, temperature_c, voltage_v, current_a, power_w

Used by CI (.github/workflows/ci-cd.yml) to validate the real batch
path (DVC's `preprocess` -> `train` stages, the same ones `run_batch.bat`
drives) end-to-end on fast, deterministic, offline data -- no network
calls, no dependency on a live SQLite DB of real sensor readings.

Also runnable standalone for local dry runs:
    python scripts/generate_synthetic_data.py
    set RESAMPLE_FREQ=5min && python -m src.pipelines.preprocess
    python -m src.pipelines.train
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = REPO_ROOT / "data" / "raw_export.csv"


def generate(nodes: int = 2, hours: int = 3, readings_per_minute: int = 1,
             seed: int = 42) -> pd.DataFrame:
    """One row per node per minute, for `hours` hours -- enough readings
    for several 5-minute resample buckets per node once fed through
    build_hourly_features(), which is what the batch pipeline needs to
    produce labeled training rows."""
    rng = np.random.default_rng(seed)
    periods = hours * 60 * readings_per_minute
    ts = pd.date_range("2026-01-01", periods=periods, freq="1min")

    frames = []
    for n in range(nodes):
        node_id = f"node-{n + 1:02d}"
        # Smooth-ish synthetic signals with a little noise so mean/std/max
        # aren't degenerate, but nothing extreme enough to be flaky.
        base_temp = 30 + n * 2
        base_current = 2 + n * 0.5
        temperature_c = base_temp + 3 * np.sin(np.linspace(0, 6, periods)) + rng.normal(0, 0.3, periods)
        voltage_v = 220 + rng.normal(0, 1.0, periods)
        current_a = base_current + 0.5 * np.sin(np.linspace(0, 4, periods)) + rng.normal(0, 0.05, periods)
        power_w = voltage_v * current_a

        frames.append(pd.DataFrame({
            "ts": ts,
            "node_id": node_id,
            "temperature_c": temperature_c,
            "voltage_v": voltage_v,
            "current_a": current_a,
            "power_w": power_w,
        }))

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--hours", type=int, default=3)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH))
    args = parser.parse_args()

    df = generate(nodes=args.nodes, hours=args.hours)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rows ({args.nodes} nodes x {args.hours}h) -> {out_path}")


if __name__ == "__main__":
    main()
