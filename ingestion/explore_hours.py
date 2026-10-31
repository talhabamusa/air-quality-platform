import os
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAQ_API_KEY")

SENSOR_IDS = [12234787, 35]  # two PM2.5 sensors at the same station

end = datetime.now(timezone.utc)
start = end - timedelta(days=7)
fmt = "%Y-%m-%dT%H:%M:%SZ"

for sensor_id in SENSOR_IDS:
    response = requests.get(
        f"https://api.openaq.org/v3/sensors/{sensor_id}/hours",
        headers={"X-API-Key": api_key},
        params={
            "datetime_from": start.strftime(fmt),
            "datetime_to": end.strftime(fmt),
            "limit": 200,
        },
        timeout=30,
    )
    print("Sensor", sensor_id, "status:", response.status_code)
    results = response.json().get("results", [])
    print("Rows returned:", len(results), "(168 expected for 7 days)")
    for row in results[:3]:
        print(row)
    print("-" * 40)