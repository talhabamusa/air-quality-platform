import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "locations"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "pm25_sensors.csv"


def is_recent(location, days=7):
    last = location.get("datetimeLast")
    if not last or not last.get("utc"):
        return False
    last_ts = datetime.fromisoformat(last["utc"].replace("Z", "+00:00"))
    return last_ts > datetime.now(timezone.utc) - timedelta(days=days)


rows = []
for path in sorted(BRONZE_DIR.glob("city=*/locations.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    city = record["city"]
    for loc in record["response"].get("results", []):
        if not is_recent(loc):
            continue
        for sensor in loc.get("sensors", []):
            param = sensor.get("parameter", {})
            if param.get("name") == "pm25":
                rows.append({
                    "city": city,
                    "location_id": loc["id"],
                    "location_name": loc.get("name"),
                    "sensor_id": sensor["id"],
                    "units": param.get("units"),
                })

manifest = pd.DataFrame(rows).drop_duplicates(subset="sensor_id")
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(MANIFEST_PATH, index=False)

print("PM2.5 sensors found:", len(manifest))
print()
print(manifest.groupby("city").size())
print()
print("Units seen:", manifest["units"].value_counts().to_dict())
print("Sensors per station (max):", manifest.groupby("location_id").size().max())
print()
print(manifest.head())