from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
IN_PATH = SILVER_DIR / "pm25_hourly.parquet"
OUT_PATH = SILVER_DIR / "pm25_hourly_imputed.parquet"
GAPS_PATH = SILVER_DIR / "pm25_gaps.csv"

# Rules chosen from the imputation experiment (gap lengths in hours)
LINEAR_MAX = 2             # 1-2 h: linear interpolation
ANCHORED_MAX = 12          # 3-12 h: city_anchored
SCALED_MAX = 24            # 13-24 h: city_scaled; longer gaps stay empty
LINEAR_FALLBACK_MAX = 6    # 3-6 h with too few neighbours: linear (not benchmarked)
MIN_OTHER_SENSORS = 3      # neighbours needed at every hour a city method uses

silver = pd.read_parquet(IN_PATH)
wide = silver.pivot(index="hour_start_utc", columns="sensor_id", values="value").sort_index()
info = silver.drop_duplicates("sensor_id").set_index("sensor_id")[["city", "location_id"]]
n = len(wide)

frames, gap_records = [], []

for city in info["city"].unique():
    sensor_ids = list(info.index[info["city"] == city])
    block = wide[sensor_ids]
    city_sum = block.sum(axis=1)
    city_cnt = block.notna().sum(axis=1)

    for sid in sensor_ids:
        x = block[sid].to_numpy(dtype=float)
        present = ~np.isnan(x)
        idx = np.flatnonzero(present)
        if len(idx) == 0:
            continue
        first, last = idx[0], idx[-1]

        # Mean of the OTHER sensors in this city, hour by hour (observed values only)
        others_cnt = (city_cnt - block[sid].notna().astype(int)).to_numpy()
        others_sum = (city_sum - block[sid].fillna(0)).to_numpy()
        c = np.where(others_cnt >= 1, others_sum / np.maximum(others_cnt, 1), np.nan)
        c_ok = others_cnt >= MIN_OTHER_SENSORS

        # How this sensor compares to its neighbours overall (used by city_scaled)
        both = present & c_ok
        ratio = x[both].sum() / c[both].sum() if both.any() and c[both].sum() > 0 else np.nan

        out = x.copy()
        method = np.full(n, "observed", dtype=object)
        gap_hours = np.zeros(n, dtype=int)
        method[:first] = "unfilled_edge"
        method[last + 1:] = "unfilled_edge"

        i = first
        while i <= last:
            if present[i]:
                i += 1
                continue
            s = i
            while not present[i]:
                i += 1
            e = i                      # first reading after the gap
            L = e - s
            a, b = x[s - 1], x[e]
            k = np.arange(1, L + 1)
            # neighbours must be available at both edges and inside the gap
            have_city = bool(c_ok[s - 1:e + 1].all())

            vals, used = None, None
            if L > SCALED_MAX:
                used = "unfilled_long"
            elif L <= LINEAR_MAX:
                vals, used = a + (b - a) * k / (L + 1), "linear"
            elif L <= ANCHORED_MAX:
                if have_city:
                    da, db = a - c[s - 1], b - c[e]
                    vals = np.maximum(c[s:e] + da + (db - da) * k / (L + 1), 0)
                    used = "city_anchored"
                elif L <= LINEAR_FALLBACK_MAX:
                    vals, used = a + (b - a) * k / (L + 1), "linear_fallback"
                else:
                    used = "unfilled_no_neighbours"
            else:
                if have_city and not np.isnan(ratio):
                    vals, used = c[s:e] * ratio, "city_scaled"
                else:
                    used = "unfilled_no_neighbours"

            if vals is not None:
                out[s:e] = vals
            method[s:e] = used
            gap_hours[s:e] = L
            gap_records.append({
                "city": city, "sensor_id": sid,
                "gap_start_utc": wide.index[s], "gap_hours": L, "method": used,
            })

        # Safety checks: observed readings untouched, no negative values
        assert np.array_equal(out[present], x[present]), "observed values were changed"
        assert not (out[~np.isnan(out)] < 0).any(), "negative PM2.5 after imputation"

        frames.append(pd.DataFrame({
            "hour_start_utc": wide.index,
            "sensor_id": sid,
            "city": city,
            "location_id": info.loc[sid, "location_id"],
            "value_raw": x,
            "value": out,
            "is_imputed": (~present) & (~np.isnan(out)),
            "impute_method": method,
            "gap_hours": gap_hours,
        }))

final = pd.concat(frames, ignore_index=True)
final = final.merge(
    silver[["sensor_id", "hour_start_utc", "pct_complete"]],
    on=["sensor_id", "hour_start_utc"], how="left",
)
gaps = pd.DataFrame(gap_records)

final.to_parquet(OUT_PATH, index=False)
gaps.to_csv(GAPS_PATH, index=False)

# ---- Report ----
total = len(final)
missing_before = int(final["value_raw"].isna().sum())
filled = int(final["is_imputed"].sum())
still_missing = missing_before - filled

print("Rows:", total)
print(f"Hours missing before imputation: {missing_before} ({missing_before / total:.1%})")
print(f"Hours filled: {filled} ({filled / missing_before:.1%} of the missing hours)")
print(f"Hours still missing: {still_missing} ({still_missing / total:.1%} of all hours)")
print()
print("Internal gaps by outcome:")
print(gaps.groupby("method").agg(gaps=("gap_hours", "size"), hours=("gap_hours", "sum")).to_string())
print()
print("Rows by status:")
print(final["impute_method"].value_counts().to_string())
print()
print("Completeness by city (% of hours with a value):")
print(final.groupby("city").agg(
    before=("value_raw", lambda s: round(100 * s.notna().mean(), 1)),
    after=("value", lambda s: round(100 * s.notna().mean(), 1)),
).to_string())
print()
print("Saved:", OUT_PATH.name, "and", GAPS_PATH.name)