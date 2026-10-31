import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAQ_API_KEY")

SENSOR_ID = 12234787

end = datetime.now(timezone.utc)
start = end - timedelta(days=7)
fmt = "%Y-%m-%dT%H:%M:%SZ"

response = requests.get(
    f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/hours",
    headers={"X-API-Key": api_key},
    params={
        "datetime_from": start.strftime(fmt),
        "datetime_to": end.strftime(fmt),
        "limit": 200,
    },
    timeout=30,
)
rows = response.json().get("results", [])

df = pd.DataFrame({
    "hour": [pd.Timestamp(r["period"]["datetimeFrom"]["utc"]) for r in rows],
    "value": [r["value"] for r in rows],
    "pct_complete": [r["coverage"]["percentComplete"] for r in rows],
}).set_index("hour").sort_index()

# Build the expected hours from the sensor's own first and last reading,
# so the grid lines up with its hour boundaries (:30 past in UTC for India)
first, last = df.index.min(), df.index.max()
full = pd.date_range(first, last, freq="h")
missing = full.difference(df.index)

# Length of each run of consecutive missing hours
gap_lengths = []
run = 0
prev = None
for ts in missing:
    if prev is not None and ts - prev == pd.Timedelta(hours=1):
        run += 1
    else:
        if run:
            gap_lengths.append(run)
        run = 1
    prev = ts
if run:
    gap_lengths.append(run)

hours_before_first = int((first - pd.Timestamp(start)).total_seconds() // 3600)

print("Hours before the sensor's first reading in the window:", hours_before_first)
print("Expected hours (first to last reading):", len(full))
print("Hours present:", len(df))
print("Hours missing inside that range:", len(missing), f"({len(missing) / len(full):.0%})")
print("Number of gaps:", len(gap_lengths))
print("Gap lengths (hours):", gap_lengths)
print("Hours with incomplete coverage (<100%):", int((df['pct_complete'] < 100).sum()))
print("First hour:", first)
print("Last hour:", last)