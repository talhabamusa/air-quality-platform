import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "pm25_sensors.csv"
LOG_PATH = PROJECT_ROOT / "data" / "manifest" / "download_log_long.csv"
HOURS_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "hours_long"

# Same end as the 30-day window, so the existing month sits inside the new window.
WINDOW_FROM = "2025-11-05T07:00:00Z"
WINDOW_TO = "2026-11-05T07:00:00Z"

PAUSE_SECONDS = 1.2          # about 50 requests a minute, under the 60/min limit
LIMIT = 1000                 # rows per page
MAX_PAGES = 20               # safety cap per sensor
MAX_CONSECUTIVE_FAILURES = 5
WINDOW = {"from": WINDOW_FROM, "to": WINDOW_TO}


def get_page(sensor_id, page, date_from=WINDOW_FROM, date_to=WINDOW_TO):
    """One page of hourly data, retrying on rate limits, server errors and network errors."""
    for attempt in range(1, 6):
        try:
            response = requests.get(
                f"{BASE_URL}/sensors/{sensor_id}/hours",
                headers={"X-API-Key": API_KEY},
                params={"datetime_from": date_from, "datetime_to": date_to,
                        "limit": LIMIT, "page": page},
                timeout=60,
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
                wait = int(float(response.headers.get("x-ratelimit-reset", 60)))
            except ValueError:
                wait = 60
            wait = min(max(wait, 5), 120)
            print(f"    rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            wait = 2 ** attempt
            print(f"    server error {response.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        print("    API said:", response.text[:300])
        response.raise_for_status()
    raise RuntimeError("gave up after 5 attempts")


def fetch_all(sensor_id, date_from=WINDOW_FROM, date_to=WINDOW_TO):
    """All pages for one sensor: keep asking until a page comes back short."""
    results, pages = [], 0
    for page in range(1, MAX_PAGES + 1):
        payload = get_page(sensor_id, page, date_from, date_to)
        batch = payload.get("results", [])
        results.extend(batch)
        pages += 1
        time.sleep(PAUSE_SECONDS)
        if len(batch) < LIMIT:
            return {"results": results}, pages
    raise RuntimeError(f"still getting full pages after {MAX_PAGES} pages")

def fetch_chunked(sensor_id, n_chunks=4):
    """Fallback: ask for the window in smaller time frames and join the results."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    edges = pd.date_range(pd.Timestamp(WINDOW_FROM), pd.Timestamp(WINDOW_TO), periods=n_chunks + 1)
    results, pages = [], 0
    for start, end in zip(edges[:-1], edges[1:]):
        payload, p = fetch_all(sensor_id, start.strftime(fmt), end.strftime(fmt))
        results.extend(payload["results"])
        pages += p
    return {"results": results}, pages

manifest = pd.read_csv(MANIFEST_PATH)
if len(sys.argv) > 1:                       # test mode: python ...py 3
    manifest = manifest.head(int(sys.argv[1]))
total = len(manifest)
expected_hours = int((pd.Timestamp(WINDOW_TO) - pd.Timestamp(WINDOW_FROM)).total_seconds() // 3600)
print(f"Window: {WINDOW_FROM} -> {WINDOW_TO}  ({expected_hours} hours)  sensors: {total}")

log, failures_in_a_row, started = [], 0, time.time()

for i, row in enumerate(manifest.itertuples(), start=1):
    out_dir = HOURS_DIR / f"city={row.city}"
    out_path = out_dir / f"sensor_id={row.sensor_id}.json.gz"

    # Safe to rerun: a sensor already saved for this exact window is not downloaded again
    cached = None
    if out_path.exists():
        with gzip.open(out_path, "rt", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("window") != WINDOW:
            cached = None
    if cached is not None:
        n = len(cached["response"]["results"])
        log.append({"city": row.city, "location_id": row.location_id, "sensor_id": row.sensor_id,
                    "status": "ok" if n else "empty", "rows": n,
                    "pages": cached.get("pages", 0), "from_cache": True})
        continue

    try:
        payload, pages = fetch_all(row.sensor_id)
    except Exception as first_error:
        print(f"    full-window request failed ({first_error}); retrying in smaller chunks")
        payload, pages = fetch_chunked(row.sensor_id)
        n = len(payload["results"])
        status = "ok" if n else "empty"

        # Bronze: save the responses untouched, wrapped with when/what we asked
        record = {
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "city": row.city,
            "location_id": int(row.location_id),
            "sensor_id": int(row.sensor_id),
            "window": WINDOW,
            "pages": pages,
            "response": payload,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(record, f)
        failures_in_a_row = 0
    except Exception as exc:
        n, pages, status = 0, 0, "failed"
        failures_in_a_row += 1
        print(f"    failed: {exc}")

    log.append({"city": row.city, "location_id": row.location_id, "sensor_id": row.sensor_id,
                "status": status, "rows": n, "pages": pages, "from_cache": False})
    print(f"[{i}/{total}] {row.city:10} sensor {row.sensor_id:>9}  {status:7} rows={n:<5} pages={pages}")

    if failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
        print("Stopping: too many failures in a row. Fix the error above and rerun; "
              "sensors already saved are kept.")
        break

log_df = pd.DataFrame(log)
log_df.to_csv(LOG_PATH, index=False)

print()
print(log_df.groupby(["city", "status"]).size().unstack(fill_value=0))
print()
ok = log_df[log_df["status"] == "ok"]
print("Total hourly rows:", int(log_df["rows"].sum()))
if len(ok):
    print(f"Median rows per working sensor: {int(ok['rows'].median())} (max possible: {expected_hours})")
    print("Most pages used by one sensor:", int(ok["pages"].max()))
print(f"Minutes elapsed this run: {(time.time() - started) / 60:.1f}")