import numpy as np
import pandas as pd
from sqlalchemy import text

BASE = ["city_key", "pm25_lag0", "pm25_lag1", "pm25_lag2", "pm25_lag6", "pm25_roll7",
        "chg_1d", "vs_mean7", "temp_mean_c", "humidity_mean_pct", "precip_total_mm",
        "wind_mean_ms", "dow_target"]
PARAMS = dict(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=30,
              l2_regularization=2.0, random_state=42)


def load_rows(engine):
    """Read the daily mart and build the next-day modelling table."""
    with engine.connect() as conn:
        mart = pd.read_sql(text("select * from gold.mart_city_daily"), conn)
    assert len(mart) > 500, f"mart_city_daily has only {len(mart)} rows: reload the 12-month data first."

    mart["date_local"] = pd.to_datetime(mart["date_local"])
    for c in ["n_sensors", "pm25_mean_ugm3", "temp_mean_c", "humidity_mean_pct",
              "precip_total_mm", "wind_mean_ms"]:
        mart[c] = mart[c].astype(float)

    frames = []
    for city, g in mart.groupby("city_key"):
        g = g.set_index("date_local").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        g["city_key"] = city
        g.index.name = "date"
        frames.append(g.reset_index())
    daily = pd.concat(frames, ignore_index=True).sort_values(["city_key", "date"]).reset_index(drop=True)

    by_city = daily.groupby("city_key")
    daily["pm25_lag0"] = daily["pm25_mean_ugm3"]
    for k in (1, 2, 6):
        daily[f"pm25_lag{k}"] = by_city["pm25_mean_ugm3"].shift(k)
    daily["pm25_roll7"] = by_city["pm25_mean_ugm3"].transform(lambda s: s.rolling(7, min_periods=4).mean())
    daily["target"] = by_city["pm25_mean_ugm3"].shift(-1)
    daily["target_date"] = daily["date"] + pd.Timedelta(days=1)
    daily["dow_target"] = daily["target_date"].dt.day_of_week
    daily["chg_1d"] = np.log1p(daily["pm25_lag0"]) - np.log1p(daily["pm25_lag1"])
    daily["vs_mean7"] = np.log1p(daily["pm25_lag0"]) - np.log1p(daily["pm25_roll7"])
    daily["city_key"] = pd.Categorical(daily["city_key"])

    rows = daily.dropna(subset=["target", "pm25_lag0", "pm25_lag6", "pm25_roll7"]).copy()
    data_window = f"{mart['date_local'].min().date()} to {mart['date_local'].max().date()}"
    return rows, daily, data_window