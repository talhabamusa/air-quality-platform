import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "bronze" / "open_meteo" / "archive_long"

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
             "precipitation", "wind_speed_10m", "wind_direction_10m"]

CITIES = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "kolkata": (22.5726, 88.3639),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
}

START_DATE = "2025-11-05"
END_DATE = "2026-11-05"
expected_rows = ((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1) * 24


def fetch(lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": ",".join(VARIABLES),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    for attempt in range(1, 5):
        response = requests.get(URL, params=params, timeout=120)
        if response.status_code == 200:
            return params, response.json()
        if response.status_code == 429 or response.status_code >= 500:
            wait = 15 * attempt
            print(f"    status {response.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        print("    API said:", response.text[:300])
        response.raise_for_status()
    raise RuntimeError("gave up after 4 attempts")


print(f"Window: {START_DATE} -> {END_DATE}  (expecting {expected_rows} hourly rows per city)")
summary = []

for city, (lat, lon) in CITIES.items():
    params, payload = fetch(lat, lon)

    # Bronze: save the response untouched, wrapped with when/what we asked
    record = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "request": params,
        "response": payload,
    }
    out_dir = OUT_DIR / f"city={city}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weather.json").write_text(json.dumps(record), encoding="utf-8")

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    with_data = hourly.loc[hourly["temperature_2m"].notna(), "time"]
    summary.append({
        "city": city,
        "rows": len(hourly),
        "first_hour": hourly["time"].min(),
        "last_hour_with_data": with_data.max(),
        "most_nulls_in_a_variable": int(hourly[VARIABLES].isna().sum().max()),
    })
    time.sleep(2)

print()
print(pd.DataFrame(summary).to_string(index=False))