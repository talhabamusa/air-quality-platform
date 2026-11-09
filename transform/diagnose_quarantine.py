from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"

df = pd.read_parquet(SILVER_DIR / "pm25_hourly.parquet")
df = df.sort_values(["sensor_id", "hour_start_utc"]).reset_index(drop=True)
q = pd.read_csv(SILVER_DIR / "pm25_quarantine.csv")
q["hour_start_utc"] = pd.to_datetime(q["hour_start_utc"], utc=True)


def runs(x, min_len=2):
    """(start index, length, value) of every run of identical consecutive readings."""
    out, start = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or not x[i] == x[start]:
            if x[start] == x[start] and i - start >= min_len:
                out.append((start, i - start, x[start]))
            start = i
    return out


# ---- 1. Run lengths across all sensors ----
rows = []
for sid, g in df.groupby("sensor_id"):
    x = g["value"].to_numpy()
    for s, n, v in runs(x):
        rows.append({"sensor_id": sid, "city": g["city"].iloc[0], "length": n, "value": v,
                     "start": g["hour_start_utc"].iloc[s]})
r = pd.DataFrame(rows)

print("Runs of identical consecutive readings, by length (all sensors):")
bins = pd.cut(r["length"], [1, 2, 3, 4, 5, 8, 12, 24, 1000],
              labels=["2", "3", "4", "5", "6-8", "9-12", "13-24", "25+"])
print(bins.value_counts().sort_index().to_string())
print()
print("Sensors with at least one run of 6+:", r[r["length"] >= 6]["sensor_id"].nunique())
print()

# ---- 2. Share of each sensor's readings that were quarantined ----
obs_n = df.groupby("sensor_id")["value"].count()
flag_n = q.groupby("sensor_id").size()
city_of = df.drop_duplicates("sensor_id").set_index("sensor_id")["city"]
share = pd.DataFrame({
    "city": city_of[flag_n.index],
    "quarantined": flag_n,
    "observed": obs_n[flag_n.index],
})
share["pct_quarantined"] = (100 * share["quarantined"] / share["observed"]).round(1)
print("Sensors with quarantined readings:")
print(share.sort_values("pct_quarantined", ascending=False).head(10).to_string())
print()

# ---- 3. The worst flat sensor: what do its runs look like? ----
worst = r[r["sensor_id"] == 12235531].sort_values("start")
print("Sensor 12235531, runs of 6+ identical readings:")
print(worst[worst["length"] >= 6][["start", "length", "value"]].to_string(index=False))
print()

# ---- 4. Hours above 500 that were NOT quarantined ----
obs = df[df["value"].notna()].copy()
g = obs.groupby(["city", "hour_start_utc"])["value"]
obs["city_median"] = g.transform("median")
obs["n_reporting"] = g.transform("count")
keys = set(zip(q["sensor_id"], q["hour_start_utc"]))
hi = obs[obs["value"] > 500]
survivors = hi[[(s, t) not in keys for s, t in zip(hi["sensor_id"], hi["hour_start_utc"])]]
print("Hours above 500 not quarantined:", len(survivors))
print(survivors[["city", "sensor_id", "hour_start_utc", "value", "city_median", "n_reporting"]]
      .to_string(index=False))
print()

# ---- 5. What the spike rule caught ----
spk = q[q["reasons"].str.contains("spike")]
print("Spike-flagged readings:", len(spk))
print("Value: min %.0f  median %.0f  max %.0f" % (spk["value"].min(), spk["value"].median(), spk["value"].max()))
print("By sensor:")
print(spk["sensor_id"].value_counts().head(6).to_string())