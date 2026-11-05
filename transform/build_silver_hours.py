import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOURS_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "hours"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
OUT_PATH = SILVER_DIR / "pm25_hourly.parquet"
SUMMARY_PATH = SILVER_DIR / "pm25_sensor_summary.csv"


def gap_runs(is_missing):
    """Lengths of each run of consecutive True values."""
    runs, run = [], 0
    for m in is_missing:
        if m:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


# ---- 1. Flatten bronze JSON into one table ----
rows = []
window_from = window_to = None
for path in sorted(HOURS_DIR.glob("city=*/sensor_id=*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    if window_from is None:
        window_from = pd.Timestamp(record["window"]["from"])
        window_to = pd.Timestamp(record["window"]["to"])
    for r in record["response"].get("results", []):
        coverage = r.get("coverage") or {}
        rows.append({
            "city": record["city"],
            "location_id": record["location_id"],
            "sensor_id": record["sensor_id"],
            "hour_start_utc": r["period"]["datetimeFrom"]["utc"],
            "value": r["value"],
            "pct_complete": coverage.get("percentComplete"),
            "observed_count": coverage.get("observedCount"),
            "expected_count": coverage.get("expectedCount"),
            "has_flags": (r.get("flagInfo") or {}).get("hasFlags"),
        })

raw = pd.DataFrame(rows)
raw["hour_start_utc"] = pd.to_datetime(raw["hour_start_utc"], utc=True)

# ---- 2. Basic cleaning ----
n_raw = len(raw)
dups = int(raw.duplicated(["sensor_id", "hour_start_utc"]).sum())
raw = raw.drop_duplicates(["sensor_id", "hour_start_utc"], keep="last")
negatives = int((raw["value"] < 0).sum())
raw["is_negative"] = raw["value"] < 0

# ---- 3. Expand each sensor onto a full hourly grid ----
grids, summary, outside, boundary_rows = [], [], 0, 0
all_internal_gaps = []

for sensor_id, g in raw.groupby("sensor_id"):
    # Indian sensors put hourly buckets at :30 past the hour in UTC, so use each
    # sensor's own minute offset instead of assuming :00
    offset = int(g["hour_start_utc"].dt.minute.mode().iloc[0])
    grid = pd.date_range(
        window_from + pd.Timedelta(minutes=offset),
        window_to + pd.Timedelta(minutes=offset) - pd.Timedelta(hours=1),
        freq="h",
    )
    g = g.set_index("hour_start_utc")
    # The API returns one bucket that straddles the start of the window
    # (it begins one hour before the grid). Trim it deliberately and count it.
    boundary = g.index == (grid[0] - pd.Timedelta(hours=1))
    boundary_rows += int(boundary.sum())
    g = g[~boundary]
    outside += len(g.index.difference(grid))

    full = g.reindex(grid)
    full.index.name = "hour_start_utc"
    full["city"] = g["city"].iloc[0]
    full["location_id"] = g["location_id"].iloc[0]
    full["sensor_id"] = sensor_id
    full["is_missing"] = full["value"].isna()
    full["is_negative"] = full["is_negative"].fillna(False).astype(bool)
    grids.append(full.reset_index())

    missing = full["is_missing"].to_numpy()
    present_idx = np.flatnonzero(~missing)
    if len(present_idx):
        leading, trailing = present_idx[0], len(missing) - 1 - present_idx[-1]
        internal = gap_runs(missing[present_idx[0]: present_idx[-1] + 1])
    else:
        leading = trailing = len(missing)
        internal = []
    all_internal_gaps.extend(internal)

    summary.append({
        "city": g["city"].iloc[0],
        "location_id": g["location_id"].iloc[0],
        "sensor_id": sensor_id,
        "expected_hours": len(grid),
        "present_hours": int((~missing).sum()),
        "pct_present": round(100 * (~missing).sum() / len(grid), 1),
        "leading_gap_hours": int(leading),
        "trailing_gap_hours": int(trailing),
        "internal_gaps": len(internal),
        "longest_internal_gap": max(internal) if internal else 0,
    })
# Fail loudly if anything other than the known boundary bucket is off-grid
assert outside == 0, f"{outside} rows fall outside the hourly grid"
silver = pd.concat(grids, ignore_index=True)
summary = pd.DataFrame(summary)

SILVER_DIR.mkdir(parents=True, exist_ok=True)
silver.to_parquet(OUT_PATH, index=False)
summary.to_csv(SUMMARY_PATH, index=False)

# ---- 4. Report ----
print("Rows read from bronze:", n_raw)
print("Duplicate (sensor, hour) rows dropped:", dups)
print("Rows outside the expected hourly grid:", outside)
print("Boundary rows trimmed (bucket starts before the window):", boundary_rows)
print("Negative PM2.5 values:", negatives)
print("Working sensors:", len(summary))
print("Working sensors per station (max):",
      raw.groupby("location_id")["sensor_id"].nunique().max())
print()
print(summary.groupby("city").agg(
    sensors=("sensor_id", "count"),
    median_pct_present=("pct_present", "median"),
    sensors_under_50pct=("pct_present", lambda s: int((s < 50).sum())),
))
print()
print("Internal gaps by length (hours):")
buckets = pd.cut(all_internal_gaps, [0, 2, 6, 24, 10_000],
                 labels=["1-2 h", "3-6 h", "7-24 h", "over 24 h"])
print(pd.Series(buckets).value_counts().reindex(["1-2 h", "3-6 h", "7-24 h", "over 24 h"]))
print()
print("Silver table:", silver.shape, "->", OUT_PATH.name)