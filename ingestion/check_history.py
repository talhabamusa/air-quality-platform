import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "openaq" / "locations"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "pm25_sensors.csv"

manifest = pd.read_csv(MANIFEST_PATH)

rows = []
for path in sorted(BRONZE_DIR.glob("city=*/locations.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    for loc in record["response"].get("results", []):
        first = (loc.get("datetimeFirst") or {}).get("utc")
        rows.append({"city": record["city"], "location_id": loc["id"], "first": first})

locs = pd.DataFrame(rows)
locs = locs[locs["location_id"].isin(manifest["location_id"])].drop_duplicates("location_id")
locs["first"] = pd.to_datetime(locs["first"], utc=True)

cutoffs = {
    "12_months": pd.Timestamp("2025-11-06", tz="UTC"),
    "6_months": pd.Timestamp("2026-05-06", tz="UTC"),
    "3_months": pd.Timestamp("2026-08-06", tz="UTC"),
}

table = pd.DataFrame({
    name: locs.groupby("city")["first"].apply(lambda s, c=cut: int((s <= c).sum()))
    for name, cut in cutoffs.items()
})
table.insert(0, "stations", locs.groupby("city").size())

print("Stations whose first reading is on or before each cutoff:")
print(table.to_string())
print()
print("Earliest first reading:", locs["first"].min())
print("Median first reading:  ", locs["first"].median())
print("Stations with no first-reading date:", int(locs["first"].isna().sum()))