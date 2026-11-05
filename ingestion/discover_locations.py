import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "locations"

CITIES = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "kolkata": (22.5726, 88.3639),
    "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
}


def fetch_locations(lat, lon):
    response = requests.get(
        f"{BASE_URL}/locations",
        headers={"X-API-Key": API_KEY},
        params={"coordinates": f"{lat},{lon}", "radius": 25000, "limit": 100},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def is_recent(location, days=7):
    last = location.get("datetimeLast")
    if not last or not last.get("utc"):
        return False
    last_ts = datetime.fromisoformat(last["utc"].replace("Z", "+00:00"))
    return last_ts > datetime.now(timezone.utc) - timedelta(days=days)


def has_pm25(location):
    return any(s.get("parameter", {}).get("name") == "pm25" for s in location.get("sensors", []))


for city, (lat, lon) in CITIES.items():
    payload = fetch_locations(lat, lon)

    # Bronze: save the response untouched, wrapped with when/what we asked
    record = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "request": {"lat": lat, "lon": lon, "radius": 25000},
        "response": payload,
    }
    out_dir = BRONZE_DIR / f"city={city}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "locations.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    results = payload.get("results", [])
    recent = [loc for loc in results if is_recent(loc)]
    recent_pm25 = [loc for loc in recent if has_pm25(loc)]
    print(
        f"{city:10} stations: {len(results):3}  "
        f"active last 7d: {len(recent):3}  "
        f"active with PM2.5: {len(recent_pm25):3}"
    )

    time.sleep(1.5)  # stay well under the API rate limit