from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"

df = pd.read_parquet(SILVER_DIR / "pm25_hourly_imputed_clean.parquet")
df = df.sort_values(["sensor_id", "hour_start_utc"]).reset_index(drop=True)
gaps = pd.read_csv(SILVER_DIR / "pm25_gaps_clean.csv")
gaps["gap_start_utc"] = pd.to_datetime(gaps["gap_start_utc"], utc=True)

observed = df[df["impute_method"] == "observed"]
by_sensor = {sid: g.reset_index(drop=True) for sid, g in df.groupby("sensor_id")}

# ---- 1. Are the city_scaled fills consistent with each sensor's own readings? ----
rows = []
for gp in gaps[gaps["method"] == "city_scaled"].itertuples():
    g = by_sensor[gp.sensor_id]
    s = int(np.flatnonzero((g["hour_start_utc"] == gp.gap_start_utc).to_numpy())[0])
    L = int(gp.gap_hours)
    rows.append({
        "city": gp.city,
        "start": gp.gap_start_utc,
        "hours": L,
        "edge_mean": np.mean([g["value_raw"].iloc[s - 1], g["value_raw"].iloc[s + L]]),
        "fill_mean": g["value"].iloc[s:s + L].mean(),
    })
cs = pd.DataFrame(rows)

print("city_scaled gaps:", len(cs))
print("Mean of the fills: %.1f   mean of the two readings at the gap edges: %.1f"
      % (cs["fill_mean"].mean(), cs["edge_mean"].mean()))
print("Gaps where the fill is more than 2x the edge readings:",
      int((cs["fill_mean"] > 2 * cs["edge_mean"]).sum()))
print("Gaps where the fill is less than half the edge readings:",
      int((cs["fill_mean"] < 0.5 * cs["edge_mean"]).sum()))
print()

by_city = cs.groupby("city").agg(
    gaps=("hours", "size"), fill_mean=("fill_mean", "mean"), edge_mean=("edge_mean", "mean"))
by_city["observed_mean"] = observed.groupby("city")["value"].mean()
print("By city:")
print(by_city.round(1).to_string())
print()


def week_of(ts):
    return ts.dt.tz_localize(None).dt.to_period("W").dt.start_time


cs["week"] = week_of(cs["start"])
obs_week = observed.assign(week=week_of(observed["hour_start_utc"])).groupby("week")["value"].mean()
by_week = cs.groupby("week").agg(gaps=("hours", "size"), fill_mean=("fill_mean", "mean"))
by_week["observed_mean"] = obs_week
print("By week (week starting):")
print(by_week.round(1).to_string())
print()


# ---- 2. Stuck sensors: long runs of the exact same reading ----
def longest_flat_run(x):
    best_len, best_start, run_start = 1, 0, 0
    for i in range(1, len(x)):
        if not x[i] == x[i - 1]:          # False for NaN, so a gap ends a run
            run_start = i
        elif i - run_start + 1 > best_len:
            best_len, best_start = i - run_start + 1, run_start
    return best_len, best_start


stuck = []
for sid, g in by_sensor.items():
    x = g["value_raw"].to_numpy()
    n_run, start = longest_flat_run(x)
    if n_run >= 6:
        stuck.append({
            "sensor_id": sid,
            "city": g["city"].iloc[0],
            "run_hours": n_run,
            "repeated_value": x[start],
            "from": g["hour_start_utc"].iloc[start].strftime("%m-%d %H:%M"),
            "to": g["hour_start_utc"].iloc[start + n_run - 1].strftime("%m-%d %H:%M"),
        })
print("Sensors with a run of 6+ identical readings:")
if stuck:
    print(pd.DataFrame(stuck).sort_values("run_hours", ascending=False).to_string(index=False))
else:
    print("none")
print()

# ---- 3. Zeros and very high readings ----
obs = observed["value"]
print("Observed readings exactly 0:", int((obs == 0).sum()))
high = observed[observed["value"] > 500].copy()
print("Observed hours above 500:", len(high), "  by city:", high["city"].value_counts().to_dict())
print("Sensors with the most of them:")
print(high["sensor_id"].value_counts().head(5).to_string())
city_median = observed.groupby(["city", "hour_start_utc"])["value"].median().rename("city_median")
high = high.join(city_median, on=["city", "hour_start_utc"])
print("Of those, hours where the city median is at least half the reading:",
      int((high["city_median"] >= 0.5 * high["value"]).sum()))