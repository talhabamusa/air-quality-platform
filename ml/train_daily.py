import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sqlalchemy import URL, create_engine, text

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
MODEL_DIR = ML_DIR / "models"
ARTIFACT_DIR = ML_DIR / "mlartifacts"
DB_PATH = ML_DIR / "mlflow.db"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT = "pm25-next-day-forecast"

FOLD_DAYS, N_FOLDS = 30, 4
BASE = ["city_key", "pm25_lag0", "pm25_lag1", "pm25_lag2", "pm25_lag6", "pm25_roll7",
        "chg_1d", "vs_mean7", "temp_mean_c", "humidity_mean_pct", "precip_total_mm",
        "wind_mean_ms", "dow_target"]
NEXT_WEATHER = ["temp_next", "humidity_next", "precip_next", "wind_next"]
VARIANTS = {
    "level": {"features": BASE, "delta": False},
    "change": {"features": BASE, "delta": True},
    "change_oracle_weather": {"features": BASE + NEXT_WEATHER, "delta": True},
}
BASELINES = ["persistence", "mean7"]
METHODS = BASELINES + list(VARIANTS)
DEPLOYABLE = "change"
PARAMS = dict(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=30,
              l2_regularization=2.0, random_state=42)

engine = create_engine(URL.create(
    "postgresql+psycopg",
    username=os.getenv("POSTGRES_USER"), password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost", port=5432, database=os.getenv("POSTGRES_DB"),
))
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
for src, dst in {"temp_mean_c": "temp_next", "humidity_mean_pct": "humidity_next",
                 "precip_total_mm": "precip_next", "wind_mean_ms": "wind_next"}.items():
    daily[dst] = by_city[src].shift(-1)
daily["target_date"] = daily["date"] + pd.Timedelta(days=1)
daily["dow_target"] = daily["target_date"].dt.day_of_week
daily["chg_1d"] = np.log1p(daily["pm25_lag0"]) - np.log1p(daily["pm25_lag1"])
daily["vs_mean7"] = np.log1p(daily["pm25_lag0"]) - np.log1p(daily["pm25_roll7"])
daily["city_key"] = pd.Categorical(daily["city_key"])

rows = daily.dropna(subset=["target", "pm25_lag0", "pm25_lag6", "pm25_roll7"]).copy()
data_window = f"{mart['date_local'].min().date()} to {mart['date_local'].max().date()}"
print("Mart rows:", len(mart), "  usable rows:", len(rows), "  data:", data_window)


def make_model():
    return HistGradientBoostingRegressor(early_stopping=False, categorical_features="from_dtype",
                                         **PARAMS)


def target_for(variant, df):
    y = np.log1p(df["target"])
    return y - np.log1p(df["pm25_lag0"]) if VARIANTS[variant]["delta"] else y


def fit_predict(variant, train, test):
    feats = VARIANTS[variant]["features"]
    model = make_model().fit(train[feats], target_for(variant, train))
    raw = model.predict(test[feats])
    if VARIANTS[variant]["delta"]:
        pred = np.expm1(np.log1p(test["pm25_lag0"]).to_numpy() + raw)
    else:
        pred = np.expm1(raw)
    return model, np.clip(pred, 0, None)


def paired_bootstrap(res, method, n=2000, seed=42):
    """95% interval for (method MAE - persistence MAE), resampling whole days."""
    rng = np.random.default_rng(seed)
    diff = (res[method] - res["target"]).abs() - (res["persistence"] - res["target"]).abs()
    by_day = diff.groupby(res["target_date"]).agg(["sum", "count"])
    sums, counts = by_day["sum"].to_numpy(), by_day["count"].to_numpy()
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return np.percentile(boot, [2.5, 97.5])


last_day = rows["target_date"].max()
folds = []
for i in range(N_FOLDS, 0, -1):
    test_end = last_day - pd.Timedelta(days=FOLD_DAYS * (i - 1))
    folds.append((test_end - pd.Timedelta(days=FOLD_DAYS - 1), test_end))

results, last = [], {}
for k, (start, end) in enumerate(folds, start=1):
    train = rows[rows["target_date"] < start]
    test = rows[(rows["target_date"] >= start) & (rows["target_date"] <= end)]
    out = test[["city_key", "target_date", "target"]].copy()
    out["fold"] = k
    out["persistence"] = test["pm25_lag0"]
    out["mean7"] = test["pm25_roll7"]
    for name in VARIANTS:
        model, out[name] = fit_predict(name, train, test)
        last[name] = (model, test)
    results.append(out)
    print(f"Fold {k}: test {start.date()} to {end.date()}   train rows {len(train)}   test rows {len(test)}")

res = pd.concat(results, ignore_index=True)
err = {m: (res[m] - res["target"]).abs() for m in METHODS}
table = {}
for m in METHODS:
    table[m] = {
        "mae": err[m].mean(),
        "rmse": float(np.sqrt(((res[m] - res["target"]) ** 2).mean())),
        "win_rate_vs_persistence_pct": 100 * (err[m] < err["persistence"]).mean(),
        "mae_vs_persistence_pct": 100 * (err[m].mean() / err["persistence"].mean() - 1),
    }
    if m != "persistence":
        lo, hi = paired_bootstrap(res, m)
        table[m]["diff_ci_low"], table[m]["diff_ci_high"] = float(lo), float(hi)
summary = pd.DataFrame(table).T

print()
print("Mean target in the test windows (ug/m3): %.1f" % res["target"].mean())
print("All test windows pooled. diff_ci = 95% interval for (method MAE - persistence MAE);")
print("if it contains 0, the method cannot be told apart from persistence:")
print(summary.round(2).to_string())
print("(change_oracle_weather uses tomorrow's actual weather: an upper bound, not deployable)")

per_fold = {m: {int(k): (d[m] - d["target"]).abs().mean() for k, d in res.groupby("fold")}
            for m in METHODS}
print()
print("MAE per test window:")
print(pd.DataFrame(per_fold).T.rename(columns=lambda k: f"fold {k}").round(2).to_string())

print()
print("MAE per city:")
print(pd.DataFrame({city: {m: (d[m] - d["target"]).abs().mean() for m in METHODS}
                    for city, d in res.groupby("city_key", observed=True)}).T.round(1).to_string())

importances = {}
for name in ["change", "change_oracle_weather"]:
    model, test = last[name]
    feats = VARIANTS[name]["features"]
    imp = permutation_importance(model, test[feats], target_for(name, test),
                                 scoring="neg_mean_absolute_error", n_repeats=10, random_state=42)
    importances[name] = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    print()
    print(f"Permutation importance, {name}, last test window (top 6):")
    print(importances[name].head(6).round(3).to_string())

# The deployable model: trained on all rows
feats = VARIANTS[DEPLOYABLE]["features"]
final = make_model().fit(rows[feats], target_for(DEPLOYABLE, rows))
model_path = MODEL_DIR / "pm25_daily_change_model.joblib"
joblib.dump({"model": final, "features": feats, "predicts_change": True,
             "cities": list(daily["city_key"].cat.categories), "data_window": data_window}, model_path)

# Log everything to MLflow
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
mlflow.set_experiment(EXPERIMENT)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for m in METHODS:
        with mlflow.start_run(run_name=m):
            is_baseline = m in BASELINES
            mlflow.set_tags({"model_type": "baseline" if is_baseline else "hist_gradient_boosting",
                             "deployable": str(m == DEPLOYABLE), "data_window": data_window})
            mlflow.log_params({"n_folds": N_FOLDS, "fold_days": FOLD_DAYS, "horizon_days": 1,
                               "n_rows": len(rows)})
            if not is_baseline:
                mlflow.log_params({**PARAMS,
                                   "predicts": "change from today" if VARIANTS[m]["delta"] else "level",
                                   "features": ",".join(VARIANTS[m]["features"])})
            mlflow.log_metrics({k: float(v) for k, v in summary.loc[m].dropna().items()})
            for fold, v in per_fold[m].items():
                mlflow.log_metric("mae_by_fold", float(v), step=fold)
            pred_file = tmp / f"predictions_{m}.csv"
            res[["fold", "city_key", "target_date", "target", m]].to_csv(pred_file, index=False)
            mlflow.log_artifact(str(pred_file))
            if m in importances:
                imp_file = tmp / f"permutation_importance_{m}.csv"
                importances[m].rename("importance").to_csv(imp_file)
                mlflow.log_artifact(str(imp_file))
            if m == DEPLOYABLE:
                mlflow.log_artifact(str(model_path))

print()
print(f"Logged {len(METHODS)} runs to the MLflow experiment '{EXPERIMENT}'.")
print("Model saved:", model_path.name)