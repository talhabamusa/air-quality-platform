import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAQ_API_KEY")

LOCATION_ID = 17  # R K Puram, Delhi - DPCC

response = requests.get(
    f"https://api.openaq.org/v3/locations/{LOCATION_ID}/sensors",
    headers={"X-API-Key": api_key},
    timeout=30,
)

print("Status:", response.status_code)
data = response.json()
for sensor in data.get("results", []):
    param = sensor.get("parameter", {})
    print(sensor["id"], param.get("name"), param.get("units"))