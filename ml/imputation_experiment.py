from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "pm25_hourly.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "experiments" / "imputation_results.csv"

GAP_LENGTHS = [1, 2, 3, 6, 9, 12, 24]
TRIALS_PER_SENSOR = 6      # hidden blocks per sensor, per gap length
MIN_OTHER_SENSORS = 3      # neighbours needed for the city-based methods
rng = np.random.default_rng(42)  # fixed seed so results are reproducible

silver = pd.read_parquet(SILVER_PATH)
wide = silver.pivot(index="hour_start_utc", columns="sensor_id", values="value").sort_index()
city_of = silver.drop_duplicates("sensor_id").set_index("sensor_id")["city"]
hours = wide.index.hour.to_numpy()
n = len(wide)

records = []
for city in city_of.unique():
    sensor_ids = city_of[city_of == city].index
    block = wide[sensor_ids]
    city_sum = block.sum(axis=1)
    city_cnt = block.notna().sum(axis=1)

    for sid in sensor_ids:
        x = block[sid].to_numpy()
        present = ~np.isnan(x)

        # Mean of the OTHER sensors in this city, hour by hour
        others_cnt = (city_cnt - block[sid].notna().astype(int)).to_numpy()
        others_sum = (city_sum - block[sid].fillna(0)).to_numpy()
        c = np.where(others_cnt >= 1, others_sum / np.maximum(others_cnt, 1), np.nan)
        c_ok = others_cnt >= MIN_OTHER_SENSORS

        # Sums used by the hour-of-day profile and the city scaling ratio
        prof_sum = np.bincount(hours[present], weights=x[present], minlength=24)
        prof_cnt = np.bincount(hours[present], minlength=24).astype(float)
        both = present & c_ok
        sum_s, sum_c = x[both].sum(), c[both].sum()

        block_ok = present & c_ok
        cs = np.concatenate([[0], np.cumsum(block_ok)])

        for L in GAP_LENGTHS:
            starts = np.arange(1, n - L)
            ok = (
                (cs[starts + L] - cs[starts] == L)   # every hidden hour is known + has neighbours
                & present[starts - 1]                # known reading just before
                & present[starts + L]                # known reading just after
                & c_ok[starts - 1]                   # city mean usable just before
                & c_ok[starts + L]                   # city mean usable just after
            )
            candidates = starts[ok]
            if len(candidates) == 0:
                continue
            chosen = rng.choice(candidates, size=min(TRIALS_PER_SENSOR, len(candidates)), replace=False)

            for s in chosen:
                y = x[s:s + L]
                a, b = x[s - 1], x[s + L]
                k = np.arange(1, L + 1)
                hh = hours[s:s + L]
                cm = c[s:s + L]

                da = a - c[s - 1]   # sensor minus city mean, just before the gap
                db = b - c[s + L]   # sensor minus city mean, just after the gap
                preds = {
                    "linear": a + (b - a) * k / (L + 1),
                    "ffill": np.full(L, a),
                    # leave the hidden values out of the average they are predicted from
                    "hour_profile": (prof_sum[hh] - y) / np.maximum(prof_cnt[hh] - 1, 1),
                    "city_mean": cm,
                    "city_scaled": cm * ((sum_s - y.sum()) / (sum_c - cm.sum())),
                    "city_anchored": np.maximum(cm + da + (db - da) * k / (L + 1), 0),
                }
                for method, p in preds.items():
                    err = p - y
                    records.append({
                        "city": city, "sensor_id": sid, "gap_hours": L, "method": method,
                        "mae": np.mean(np.abs(err)), "mse": np.mean(err ** 2),
                    })

results = pd.DataFrame(records)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUT_PATH, index=False)

order = ["linear", "ffill", "hour_profile", "city_mean", "city_scaled", "city_anchored"]
mae = results.pivot_table(index="method", columns="gap_hours", values="mae", aggfunc="mean").reindex(order)
rmse = results.pivot_table(index="method", columns="gap_hours", values="mse", aggfunc="mean").reindex(order) ** 0.5
trials = results[results["method"] == "linear"].groupby("gap_hours").size()

print("Mean PM2.5 over the whole table (ug/m3):", round(float(np.nanmean(wide.to_numpy())), 1))
print()
print("Hidden blocks tested per gap length:", trials.to_dict())
print()
print("MAE (ug/m3), lower is better:")
print(mae.round(1).to_string())
print()
print("RMSE (ug/m3), lower is better:")
print(rmse.round(1).to_string())
print()
print("Best method by MAE for each gap length:")
print(mae.idxmin().to_string())
print()
print("Saved:", OUT_PATH.name)