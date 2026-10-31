import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAQ_API_KEY")

response = requests.get(
    "https://api.openaq.org/v3/locations",
    headers={"X-API-Key": api_key},
    params={"coordinates": "28.6139,77.2090", "radius": 25000, "limit": 5},
    timeout=30,
)

print("Status:", response.status_code)
data = response.json()
for loc in data.get("results", []):
    print(loc["id"], loc["name"])