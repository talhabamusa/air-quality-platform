import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sqlalchemy import URL, create_engine

from features import BASE, PARAMS, load_rows

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
MODEL_DIR = ML_DIR / "models"
ARTIFACT_DIR = ML_DIR / "mlartifacts"
DB_PATH = ML_DIR / "mlflow.db"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT = "pm25-next-day-intervals"

FOLD_DAYS, N_FOLDS = 30, 4
QUANTILES = (0.1, 0.5, 0.9)
ALPHA = 0.2                      # 80% interval

engine = create_engine(URL.create(
    "postgresql+psycopg",
    username=os.getenv("POSTGRES_USER"), password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost", port=5432, database=os.getenv("POSTGRES_DB"),
))
rows, daily, data_window = load_rows(engine)
print("Usable rows:", len(rows), "  data:", data_window)


def delta_target(df):
    """Change in log(1 + PM2.5) from today to tomorrow."""
    return np.log1p(df["target"]) - np.log1p(df["pm25_lag0"])


def make_quantile_model(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, early_stopping=False,
                                         categorical_features="from_dtype", **PARAMS)


def to_level(df, log_delta):
    return np.clip(np.expm1(np.log1p(df["pm25_lag0"]).to_numpy() + log_delta), 0, None)


def interval_metrics(y, lo, hi):
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    width = hi - lo
    score = width + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)
    return {"coverage_pct": 100 * np.mean((y >= lo) & (y <= hi)),
            "below_pct": 100 * np.mean(y < lo), "above_pct": 100 * np.mean(y > hi),
            "mean_width": width.mean(), "interval_score": score.mean()}


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

results = []
for k, (start, end) in enumerate(folds, start=1):
    train = rows[rows["target_date"] < start]
    test = rows[(rows["target_date"] >= start) & (rows["target_date"] <= end)]
    d_train = delta_target(train)

    # Model: three quantile regressions on the change from today
    raw = np.column_stack([make_quantile_model(q).fit(train[BASE], d_train).predict(test[BASE])
                           for q in QUANTILES])
    raw = np.sort(raw, axis=1)                      # keep the quantiles in order

    # Baseline: today's value plus each city's past quantiles of the change
    cq = (train.assign(d=d_train).groupby("city_key", observed=True)["d"]
          .quantile(list(QUANTILES)).unstack())
    cq.index = cq.index.astype(str)
    band = cq.reindex(test["city_key"].astype(str)).to_numpy()

    out = test[["city_key", "target_date", "target"]].copy()
    out["fold"] = k
    out["persistence"] = test["pm25_lag0"]
    out["model_lo"], out["model_med"], out["model_hi"] = (to_level(test, raw[:, i]) for i in range(3))
    out["band_lo"], out["band_med"], out["band_hi"] = (to_level(test, band[:, i]) for i in range(3))
    results.append(out)
    print(f"Fold {k}: test {start.date()} to {end.date()}   train rows {len(train)}   test rows {len(test)}")

res = pd.concat(results, ignore_index=True)
INTERVALS = {"band_persistence": ("band_lo", "band_hi"), "model_quantile": ("model_lo", "model_hi")}

pooled = pd.DataFrame({name: interval_metrics(res["target"], res[lo], res[hi])
                       for name, (lo, hi) in INTERVALS.items()}).T
print()
print("Mean target in the test windows (ug/m3): %.1f" % res["target"].mean())
print("80% intervals, all test windows pooled (coverage should be near 80; lower width and score is better):")
print(pooled.round(1).to_string())

print()
print("Coverage (%) per test window:")
print(pd.DataFrame({name: {int(k): interval_metrics(d["target"], d[lo], d[hi])["coverage_pct"]
                           for k, d in res.groupby("fold")}
                    for name, (lo, hi) in INTERVALS.items()}).T.rename(columns=lambda k: f"fold {k}")
      .round(1).to_string())

city_rows = {}
for city, d in res.groupby("city_key", observed=True):
    for name, (lo, hi) in INTERVALS.items():
        m = interval_metrics(d["target"], d[lo], d[hi])
        city_rows[(city, name)] = {"coverage_pct": m["coverage_pct"], "mean_width": m["mean_width"]}
print()
print("Coverage and mean width per city:")
print(pd.DataFrame(city_rows).T.round(1).to_string())

# The median forecast as a point forecast, against persistence
point = {}
for name, col in {"persistence": "persistence", "model_median": "model_med"}.items():
    e = (res[col] - res["target"]).abs()
    point[name] = {"mae": e.mean(), "rmse": float(np.sqrt(((res[col] - res["target"]) ** 2).mean())),
                   "win_rate_vs_persistence_pct": 100 * (e < (res["persistence"] - res["target"]).abs()).mean()}
lo_ci, hi_ci = paired_bootstrap(res, "model_med")
point["model_median"]["diff_ci_low"], point["model_median"]["diff_ci_high"] = float(lo_ci), float(hi_ci)
print()
print("Median forecast vs persistence (diff_ci = 95% interval for MAE difference; contains 0 = no clear difference):")
print(pd.DataFrame(point).T.round(2).to_string())

# Models trained on all rows, for later use
d_all = delta_target(rows)
final = {q: make_quantile_model(q).fit(rows[BASE], d_all) for q in QUANTILES}
model_path = MODEL_DIR / "pm25_daily_quantile_models.joblib"
joblib.dump({"models": final, "features": BASE, "quantiles": QUANTILES,
             "cities": list(daily["city_key"].cat.categories), "data_window": data_window}, model_path)

# Log to MLflow
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
mlflow.set_experiment(EXPERIMENT)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for name, (lo, hi) in INTERVALS.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"data_window": data_window, "interval": "80%"})
            mlflow.log_params({"n_folds": N_FOLDS, "fold_days": FOLD_DAYS, "n_rows": len(rows),
                               "quantiles": ",".join(map(str, QUANTILES))})
            if name == "model_quantile":
                mlflow.log_params(PARAMS)
                mlflow.log_metrics({f"median_{k}": float(v) for k, v in point["model_median"].items()})
            mlflow.log_metrics({k: float(v) for k, v in pooled.loc[name].items()})
            for k, d in res.groupby("fold"):
                mlflow.log_metric("coverage_by_fold", float(interval_metrics(d["target"], d[lo], d[hi])["coverage_pct"]),
                                  step=int(k))
            f = tmp / f"predictions_{name}.csv"
            res[["fold", "city_key", "target_date", "target", lo, hi]].to_csv(f, index=False)
            mlflow.log_artifact(str(f))
            if name == "model_quantile":
                mlflow.log_artifact(str(model_path))

print()
print(f"Logged 2 runs to the MLflow experiment '{EXPERIMENT}'.")
print("Models saved:", model_path.name)