import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOURS_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "hours"

rows = []
window_from = window_to = None
for path in sorted(HOURS_DIR.glob("city=*/sensor_id=*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    if window_from is None:
        window_from = pd.Timestamp(record["window"]["from"])
        window_to = pd.Timestamp(record["window"]["to"])
    for r in record["response"].get("results", []):
        rows.append({
            "city": record["city"],
            "sensor_id": record["sensor_id"],
            "hour_start_utc": r["period"]["datetimeFrom"]["utc"],
        })

raw = pd.DataFrame(rows)
raw["hour_start_utc"] = pd.to_datetime(raw["hour_start_utc"], utc=True)
raw["minute"] = raw["hour_start_utc"].dt.minute

# Each sensor's usual minute offset, same rule as the silver script
offsets = raw.groupby("sensor_id")["minute"].agg(lambda s: int(s.mode().iloc[0]))
raw["offset"] = raw["sensor_id"].map(offsets)

delta = pd.to_timedelta(raw["offset"], unit="m")
raw["grid_start"] = window_from + delta
raw["grid_end"] = window_to + delta - pd.Timedelta(hours=1)

outside = raw[(raw["hour_start_utc"] < raw["grid_start"]) | (raw["hour_start_utc"] > raw["grid_end"])]

print("Window:", window_from, "->", window_to)
print()
print("Sensor minute offsets (sensors per offset):")
print(offsets.value_counts().to_dict())
print()
print("Rows outside the grid:", len(outside))
print("  before grid start:", int((outside["hour_start_utc"] < outside["grid_start"]).sum()))
print("  after grid end:   ", int((outside["hour_start_utc"] > outside["grid_end"]).sum()))
print("  sensors affected: ", outside["sensor_id"].nunique())
print("  rows whose minute differs from the sensor's usual offset:",
      int((outside["minute"] != outside["offset"]).sum()))
print()
print("Outside rows by city:")
print(outside["city"].value_counts().to_dict())
print()
print(outside[["city", "sensor_id", "hour_start_utc", "offset", "grid_start"]].head(8).to_string(index=False))