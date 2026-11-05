import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "pm25_sensors.csv"
LOG_PATH = PROJECT_ROOT / "data" / "manifest" / "download_log.csv"
HOURS_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "hours"

DAYS = 30
PAUSE_SECONDS = 1.2
LIMIT = 1000
FMT = "%Y-%m-%dT%H:%M:%SZ"


def fetch_hours(sensor_id, start, end):
    """Download hourly data for one sensor, retrying on rate limits and server errors."""
    for attempt in range(1, 6):
        try:
            response = requests.get(
                f"{BASE_URL}/sensors/{sensor_id}/hours",
                headers={"X-API-Key": API_KEY},
                params={
                    "datetime_from": start.strftime(FMT),
                    "datetime_to": end.strftime(FMT),
                    "limit": LIMIT,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"    network error ({type(exc).__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            try:
                wait = int(float(response.headers.get("Retry-After", 30)))
            except ValueError:
                wait = 30
            print(f"    rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            wait = 2 ** attempt
            print(f"    server error {response.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()  # other 4xx: give up on this sensor

    raise RuntimeError("gave up after 5 attempts")


end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=DAYS)
print(f"Window: {start.strftime(FMT)} -> {end.strftime(FMT)}")

manifest = pd.read_csv(MANIFEST_PATH)
total = len(manifest)
log = []

for i, row in enumerate(manifest.itertuples(), start=1):
    out_dir = HOURS_DIR / f"city={row.city}"
    out_path = out_dir / f"sensor_id={row.sensor_id}.json"

    # Safe to rerun: sensors already saved are not downloaded again
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        n = len(existing["response"].get("results", []))
        log.append({"city": row.city, "location_id": row.location_id,
                    "sensor_id": row.sensor_id, "status": "ok" if n else "empty",
                    "rows": n, "from_cache": True})
        continue

    try:
        payload = fetch_hours(row.sensor_id, start, end)
        n = len(payload.get("results", []))
        status = "ok" if n else "empty"
        if n >= LIMIT:
            print("    warning: hit the row limit, data may be truncated")

        # Bronze: save the response untouched, wrapped with when/what we asked
        record = {
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "city": row.city,
            "location_id": int(row.location_id),
            "sensor_id": int(row.sensor_id),
            "window": {"from": start.strftime(FMT), "to": end.strftime(FMT)},
            "response": payload,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record), encoding="utf-8")
    except Exception as exc:
        n, status = 0, "failed"
        print(f"    failed: {exc}")

    log.append({"city": row.city, "location_id": row.location_id,
                "sensor_id": row.sensor_id, "status": status,
                "rows": n, "from_cache": False})
    print(f"[{i}/{total}] {row.city:10} sensor {row.sensor_id:>9}  {status:7} rows={n}")
    time.sleep(PAUSE_SECONDS)

log_df = pd.DataFrame(log)
log_df.to_csv(LOG_PATH, index=False)

print()
print(log_df.groupby(["city", "status"]).size().unstack(fill_value=0))
print()
print("Total hourly rows:", int(log_df["rows"].sum()))
ok = log_df[log_df["status"] == "ok"]
print("Median rows per working sensor:", int(ok["rows"].median()), "(max possible: ~720)")