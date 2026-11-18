import gzip
import json
from pathlib import Path

import pandas as pd

path = (Path(__file__).resolve().parents[1] / "data" / "bronze" / "openaq" / "hours_long"
        / "city=delhi" / "sensor_id=15289075.json.gz")
with gzip.open(path, "rt", encoding="utf-8") as f:
    record = json.load(f)

rows = [{"from": r["period"]["datetimeFrom"]["utc"], "to": r["period"]["datetimeTo"]["utc"],
         "value": r["value"], "pct_complete": (r.get("coverage") or {}).get("percentComplete")}
        for r in record["response"]["results"]]
df = pd.DataFrame(rows)
df["from"] = pd.to_datetime(df["from"], utc=True)
lo, hi = pd.Timestamp("2026-01-28 07:00", tz="UTC"), pd.Timestamp("2026-01-28 15:00", tz="UTC")
print(df[(df["from"] >= lo) & (df["from"] <= hi)].sort_values("from").to_string(index=False))