import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sqlalchemy import URL, create_engine, text

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
DOCS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FOLD_DAYS, N_FOLDS = 30, 4
FEATURES = ["city_key", "pm25_lag0", "pm25_lag1", "pm25_lag2", "pm25_lag6", "pm25_roll7",
            "temp_mean_c", "humidity_mean_pct", "precip_total_mm", "wind_mean_ms", "dow_target"]
METHODS = ["persistence", "seasonal_naive", "mean7", "model"]

engine = create_engine(URL.create(
    "postgresql+psycopg",
    username=os.getenv("POSTGRES_USER"), password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost", port=5432, database=os.getenv("POSTGRES_DB"),
))
with engine.connect() as conn:
    mart = pd.read_sql(text("select * from gold.mart_city_daily"), conn)

assert len(mart) > 500, (f"mart_city_daily has only {len(mart)} rows. The database probably still "
                         "holds the 30-day data: run load_staging.py and dbt build first.")

mart["date_local"] = pd.to_datetime(mart["date_local"])
for c in ["n_sensors", "pm25_mean_ugm3", "temp_mean_c", "humidity_mean_pct",
          "precip_total_mm", "wind_mean_ms"]:
    mart[c] = mart[c].astype(float)

print("Mart rows:", len(mart), " cities:", mart["city_key"].nunique(),
      " dates:", mart["date_local"].min().date(), "to", mart["date_local"].max().date())

# One row per city and calendar day (days without a city value stay empty)
frames = []
for city, g in mart.groupby("city_key"):
    g = g.set_index("date_local").sort_index()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    g["city_key"] = city
    g.index.name = "date"
    frames.append(g.reset_index())
daily = pd.concat(frames, ignore_index=True).sort_values(["city_key", "date"]).reset_index(drop=True)

# Features known on day t, for predicting day t+1
by_city = daily.groupby("city_key")["pm25_mean_ugm3"]
daily["pm25_lag0"] = daily["pm25_mean_ugm3"]
for k in (1, 2, 6):
    daily[f"pm25_lag{k}"] = by_city.shift(k)
daily["pm25_roll7"] = by_city.transform(lambda s: s.rolling(7, min_periods=4).mean())
daily["target"] = by_city.shift(-1)
daily["target_date"] = daily["date"] + pd.Timedelta(days=1)
daily["dow_target"] = daily["target_date"].dt.day_of_week
daily["city_key"] = pd.Categorical(daily["city_key"])

# Evaluate only on rows where every baseline can also make a forecast
rows = daily.dropna(subset=["target", "pm25_lag0", "pm25_lag6", "pm25_roll7"]).copy()
print("Rows usable for modelling:", len(rows))
print("Rows per city:", rows.groupby("city_key", observed=True).size().to_dict())


def make_model():
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=42,
        categorical_features="from_dtype")


# Rolling-origin evaluation: each fold trains on everything before its test window
last_day = rows["target_date"].max()
folds = []
for i in range(N_FOLDS, 0, -1):
    test_end = last_day - pd.Timedelta(days=FOLD_DAYS * (i - 1))
    folds.append((test_end - pd.Timedelta(days=FOLD_DAYS - 1), test_end))

results, last_model, last_test = [], None, None
print()
for k, (start, end) in enumerate(folds, start=1):
    train = rows[rows["target_date"] < start]
    test = rows[(rows["target_date"] >= start) & (rows["target_date"] <= end)]
    model = make_model().fit(train[FEATURES], np.log1p(train["target"]))
    out = test[["city_key", "target_date", "target"]].copy()
    out["fold"] = k
    out["model"] = np.clip(np.expm1(model.predict(test[FEATURES])), 0, None)
    out["persistence"] = test["pm25_lag0"]
    out["seasonal_naive"] = test["pm25_lag6"]
    out["mean7"] = test["pm25_roll7"]
    results.append(out)
    last_model, last_test = model, test
    print(f"Fold {k}: test {start.date()} to {end.date()}   train rows {len(train)}   test rows {len(test)}")

res = pd.concat(results, ignore_index=True)

mae = pd.Series({m: (res[m] - res["target"]).abs().mean() for m in METHODS})
rmse = pd.Series({m: float(np.sqrt(((res[m] - res["target"]) ** 2).mean())) for m in METHODS})
summary = pd.DataFrame({"MAE": mae, "RMSE": rmse})
summary["MAE_vs_persistence_%"] = 100 * (summary["MAE"] / mae["persistence"] - 1)
print()
print("Mean target in the test windows (ug/m3): %.1f" % res["target"].mean())
print("All test windows pooled (lower is better):")
print(summary.round(1).to_string())

per_fold = pd.DataFrame({
    f"fold {k}": {m: (d[m] - d["target"]).abs().mean() for m in METHODS}
    for k, d in res.groupby("fold")
})
print()
print("MAE per test window:")
print(per_fold.round(1).to_string())

per_city = pd.DataFrame({
    city: {**{m: (d[m] - d["target"]).abs().mean() for m in METHODS}, "mean_target": d["target"].mean()}
    for city, d in res.groupby("city_key", observed=True)
}).T
print()
print("MAE per city (and the city's mean PM2.5 in the test windows):")
print(per_city.round(1).to_string())

# Which inputs matter most (shuffle one input at a time, look at how much the error grows)
imp = permutation_importance(last_model, last_test[FEATURES], np.log1p(last_test["target"]),
                             scoring="neg_mean_absolute_error", n_repeats=10, random_state=42)
imp = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
print()
print("Permutation importance on the last test window (error increase in log units):")
print(imp.round(3).to_string())

# Final model: trained on all available rows
final = make_model().fit(rows[FEATURES], np.log1p(rows["target"]))
joblib.dump({"model": final, "features": FEATURES, "cities": list(daily["city_key"].cat.categories)},
            MODEL_DIR / "pm25_daily_hgb.joblib")
metrics = {
    "test_windows": [[str(a.date()), str(b.date())] for a, b in folds],
    "pooled": summary.round(2).to_dict(),
    "mean_target": round(float(res["target"].mean()), 2),
}
(PROJECT_ROOT / "ml" / "metrics_daily.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# Chart: Delhi, all test windows
d = res[res["city_key"] == "delhi"].sort_values("target_date")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(d["target_date"], d["target"], label="Actual", linewidth=1.8)
ax.plot(d["target_date"], d["model"], label="Gradient boosting", linewidth=1.2)
ax.plot(d["target_date"], d["persistence"], label="Persistence (today's value)", linewidth=1, alpha=0.7)
ax.set_ylabel("Daily mean PM2.5 (ug/m3)")
ax.set_title("Delhi: next-day PM2.5 forecast on the test windows")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(DOCS_DIR / "forecast_delhi.png", dpi=150)

print()
print("Saved: ml/models/pm25_daily_hgb.joblib, ml/metrics_daily.json, docs/forecast_delhi.png")