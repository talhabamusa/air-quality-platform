from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
VARS = ["temp_c", "humidity_pct", "pressure_hpa", "precip_mm", "wind_ms", "wind_dir_deg"]

pm = pd.read_parquet(SILVER_DIR / "pm25_hourly_imputed_clean.parquet")
wx = pd.read_parquet(SILVER_DIR / "weather_hourly.parquet")
pm["hour_start_utc"] = pm["hour_start_utc"].astype("datetime64[ns, UTC]")
wx["hour_utc"] = wx["hour_utc"].astype("datetime64[ns, UTC]")

# Every PM2.5 hour must start at :30, so its midpoint lands on a weather timestamp
assert (pm["hour_start_utc"].dt.minute == 30).all(), "some PM2.5 hours do not start at :30"
pm["weather_hour_utc"] = pm["hour_start_utc"] + pd.Timedelta(minutes=30)

merged = pm.merge(
    wx, left_on=["city", "weather_hour_utc"], right_on=["city", "hour_utc"],
    how="left", validate="many_to_one",
)

print("PM2.5 rows:", len(pm), "  after join:", len(merged))
print("Rows without weather:", int(merged["temp_c"].isna().sum()))
assert len(merged) == len(pm), "join changed the row count"

# First look: Spearman correlation of observed PM2.5 with each weather variable
observed = merged[merged["impute_method"] == "observed"]
corr = {}
for city, g in observed.groupby("city"):
    corr[city] = g[["value"] + VARS].corr(method="spearman")["value"].drop("value")

print()
print("Observed readings per city:", observed.groupby("city").size().to_dict())
print()
print("Spearman correlation of PM2.5 with weather (observed readings only):")
print(pd.DataFrame(corr).T.round(2).to_string())