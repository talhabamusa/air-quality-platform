import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "ml"))
from features import BASE, load_rows  # noqa: E402

load_dotenv()
OUT = PROJECT_ROOT / "dashboard" / "app_data"      # not called "data", which .gitignore skips
OUT.mkdir(parents=True, exist_ok=True)

engine = create_engine(URL.create(
    "postgresql+psycopg",
    username=os.getenv("POSTGRES_USER"), password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost", port=5432, database=os.getenv("POSTGRES_DB"),
))


def q(sql):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# 1. Daily city table
mart = q("select * from gold.mart_city_daily order by city_key, date_local")
for c in ["n_sensors", "pm25_mean_ugm3", "pm25_median_ugm3", "weather_hours",
          "temp_mean_c", "humidity_mean_pct", "precip_total_mm", "wind_mean_ms"]:
    mart[c] = mart[c].astype(float)
mart.to_csv(OUT / "mart_city_daily.csv", index=False)

# 2. Latest next-day forecast per city, from the saved quantile models
rows, daily, data_window = load_rows(engine)
bundle = joblib.load(PROJECT_ROOT / "ml" / "models" / "pm25_daily_quantile_models.joblib")
latest = (daily.dropna(subset=["pm25_lag0"]).sort_values("date")
          .groupby("city_key", observed=True).tail(1).copy())
raw = np.column_stack([bundle["models"][qq].predict(latest[BASE]) for qq in bundle["quantiles"]])
raw = np.sort(raw, axis=1)
base = np.log1p(latest["pm25_lag0"].to_numpy())


def level(delta):
    return np.clip(np.expm1(base + delta), 0, None)


forecast = pd.DataFrame({
    "city_key": latest["city_key"].astype(str).to_numpy(),
    "as_of_date": latest["date"].dt.date.to_numpy(),
    "forecast_date": latest["target_date"].dt.date.to_numpy(),
    "today_pm25": latest["pm25_lag0"].to_numpy(),
    "persistence": latest["pm25_lag0"].to_numpy(),
    "forecast_p10": level(raw[:, 0]),
    "forecast_median": level(raw[:, 1]),
    "forecast_p90": level(raw[:, 2]),
})
forecast.to_csv(OUT / "forecast_latest.csv", index=False)

# 3. Data quality tiers and the daily cycle
tiers = q("select city_key, quality_tier, count(*) as hours from gold.fact_air_quality group by 1, 2 order by 1, 2")
tiers["hours"] = tiers["hours"].astype(int)
tiers.to_csv(OUT / "quality_tiers.csv", index=False)

profile = q("""
select f.city_key, h.hour_of_day_local as hour, avg(f.pm25_ugm3) as pm25
from gold.fact_air_quality f
join gold.dim_hour h on h.hour_key = f.hour_key
where f.quality_tier in ('observed', 'imputed')
group by 1, 2 order by 1, 2
""")
profile["pm25"] = profile["pm25"].astype(float)
profile.to_csv(OUT / "hourly_profile.csv", index=False)

# 4. Model results logged in MLflow (latest run of each name)
mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'ml' / 'mlflow.db').as_posix()}")
frames = []
for name in ["pm25-next-day-forecast", "pm25-next-day-intervals"]:
    r = mlflow.search_runs(experiment_names=[name])
    if len(r):
        r.insert(0, "experiment", name)
        frames.append(r)
runs = pd.concat(frames, ignore_index=True)
cols = ["experiment", "tags.mlflow.runName"] + [c for c in runs.columns if c.startswith("metrics.")]
runs = runs[cols].rename(columns=lambda c: c.replace("tags.mlflow.runName", "run").replace("metrics.", ""))
runs = runs.drop_duplicates(subset=["experiment", "run"], keep="first")
runs.to_csv(OUT / "model_runs.csv", index=False)

# 5. Test-window predictions, saved by train_intervals.py
backtest = PROJECT_ROOT / "ml" / "backtest_predictions.csv"
if backtest.exists():
    shutil.copy(backtest, OUT / "backtest_predictions.csv")
else:
    print("WARNING: ml/backtest_predictions.csv not found. Rerun ml/train_intervals.py first.")

(OUT / "meta.json").write_text(json.dumps({
    "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "data_window": data_window,
    "cities": sorted(mart["city_key"].unique().tolist()),
}, indent=2), encoding="utf-8")

print("Files written to dashboard/app_data:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:28} {f.stat().st_size / 1024:8.1f} KB")
print()
print("Latest forecasts:")
print(forecast.round(1).to_string(index=False))