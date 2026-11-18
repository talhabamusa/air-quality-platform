import gzip
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOURS_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "hours_long"

rows, window = [], None
for path in sorted(HOURS_DIR.glob("city=*/sensor_id=*.json.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        record = json.load(f)
    window = record["window"]
    for r in record["response"].get("results", []):
        rows.append({
            "city": record["city"],
            "sensor_id": record["sensor_id"],
            "hour_start_utc": r["period"]["datetimeFrom"]["utc"],
            "hour_end_utc": r["period"]["datetimeTo"]["utc"],
            "value": r["value"],
        })

raw = pd.DataFrame(rows)
raw["hour_start_utc"] = pd.to_datetime(raw["hour_start_utc"], utc=True)
raw["hour_end_utc"] = pd.to_datetime(raw["hour_end_utc"], utc=True)
window_from, window_to = pd.Timestamp(window["from"]), pd.Timestamp(window["to"])

n_read = len(raw)
dups = int(raw.duplicated(["sensor_id", "hour_start_utc"]).sum())
raw = raw.drop_duplicates(["sensor_id", "hour_start_utc"], keep="last")

# Same rules as the silver builder
raw["minute"] = raw["hour_start_utc"].dt.minute
offsets = raw.groupby("sensor_id")["minute"].agg(lambda s: int(s.mode().iloc[0]))
raw["offset"] = raw["sensor_id"].map(offsets)
delta = pd.to_timedelta(raw["offset"], unit="m")
grid_start = window_from + delta
grid_end = window_to + delta - pd.Timedelta(hours=1)

before = raw["hour_start_utc"] < grid_start
after = raw["hour_start_utc"] > grid_end
off_minute = raw["minute"] != raw["offset"]
boundary = raw["hour_start_utc"] == grid_start - pd.Timedelta(hours=1)
outside = raw[(before | after | off_minute) & ~boundary]

print("Window:", window)
print("Rows read:", n_read, "  duplicate (sensor, hour) rows dropped:", dups)
print("Sensors:", raw["sensor_id"].nunique())
print("Sensor minute offsets (sensors per offset):", offsets.value_counts().to_dict())
print()
print("Boundary rows (bucket starts one hour before the grid):", int(boundary.sum()))
print("Rows outside the grid, not counting boundary rows:", len(outside))
print("  on a different minute than the sensor's usual:", int((off_minute & ~boundary).sum()))
print("  before the grid start:", int((before & ~boundary).sum()))
print("  after the grid end:   ", int((after & ~boundary).sum()))
print()
print(outside[["city", "sensor_id", "hour_start_utc", "hour_end_utc", "offset", "value"]]
      .head(20).to_string(index=False))