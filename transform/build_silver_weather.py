import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "open_meteo" / "archive"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
OUT_PATH = SILVER_DIR / "weather_hourly.parquet"

RENAME = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precip_mm",
    "wind_speed_10m": "wind_ms",
    "wind_direction_10m": "wind_dir_deg",
}
RANGES = {   # physically plausible limits for these cities
    "temp_c": (-10, 55),
    "humidity_pct": (0, 100),
    "pressure_hpa": (800, 1100),
    "precip_mm": (0, 200),
    "wind_ms": (0, 50),
    "wind_dir_deg": (0, 360),
}

frames = []
for path in sorted(BRONZE_DIR.glob("city=*/weather.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    hourly = pd.DataFrame(record["response"]["hourly"])
    hourly["hour_utc"] = pd.to_datetime(hourly.pop("time"), utc=True)
    hourly.insert(0, "city", record["city"])
    frames.append(hourly.rename(columns=RENAME))

weather = pd.concat(frames, ignore_index=True)

# ---- Checks ----
dups = int(weather.duplicated(["city", "hour_utc"]).sum())
rows_per_city = weather.groupby("city").size()
nulls = int(weather[list(RENAME.values())].isna().sum().sum())
out_of_range = {col: int(((weather[col] < lo) | (weather[col] > hi)).sum())
                for col, (lo, hi) in RANGES.items()}

print("Rows:", len(weather), " cities:", weather["city"].nunique())
print("Rows per city:", rows_per_city.to_dict())
print("Duplicate (city, hour) rows:", dups)
print("Null values:", nulls)
print("Out-of-range values by column:", out_of_range)

assert dups == 0, "duplicate weather rows"
assert rows_per_city.nunique() == 1, "cities have different row counts"
assert nulls == 0, "unexpected nulls in weather"
assert sum(out_of_range.values()) == 0, "weather values outside plausible ranges"

SILVER_DIR.mkdir(parents=True, exist_ok=True)
weather.to_parquet(OUT_PATH, index=False)

print()
print(weather.groupby("city").agg(
    temp_min=("temp_c", "min"), temp_max=("temp_c", "max"),
    humidity_mean=("humidity_pct", "mean"),
    wind_mean=("wind_ms", "mean"),
    rain_total_mm=("precip_mm", "sum"),
).round(1).to_string())
print()
print("Hours:", weather["hour_utc"].min(), "->", weather["hour_utc"].max())
print("Saved:", OUT_PATH.name)