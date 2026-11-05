from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "silver" / "pm25_hourly_imputed.parquet"

df = pd.read_parquet(IN_PATH).sort_values(["sensor_id", "hour_start_utc"])
observed = df[df["impute_method"] == "observed"]
filled = df[df["is_imputed"]]

# 1. Do filled values look like observed ones?
obs = observed["value"]
print("Observed values: mean %.1f  median %.1f  min %.1f  max %.1f"
      % (obs.mean(), obs.median(), obs.min(), obs.max()))
print()
print("Filled values by method:")
print(filled.groupby("impute_method")["value"]
      .agg(["count", "mean", "median", "min", "max"]).round(1).to_string())
print()

# 2. Any filled value above the highest observed value in its city?
city_max = observed.groupby("city")["value"].max()
above = filled[filled["value"] > filled["city"].map(city_max)]
print("Filled values above their city's highest observed value:", len(above))
print()

# 3. Extreme observed values
print("Observed percentiles: 99%% = %.0f   99.9%% = %.0f   max = %.0f"
      % (obs.quantile(0.99), obs.quantile(0.999), obs.max()))
print("Observed hours above 500:", int((obs > 500).sum()),
      "  above 1000:", int((obs > 1000).sum()))
print()


# 4. Stuck sensors: long runs of the exact same reading
def longest_flat_run(x):
    best = run = 0
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:      # NaN == NaN is False, so gaps break a run
            run += 1
            best = max(best, run + 1)
        else:
            run = 0
    return best


flat = df.groupby("sensor_id")["value_raw"].apply(lambda s: longest_flat_run(s.to_numpy()))
print("Sensors with 6+ identical consecutive readings:", int((flat >= 6).sum()), "of", len(flat))
print("Longest identical run (hours):", int(flat.max()))
print()
print("Five longest runs (sensor_id, hours):")
print(flat.sort_values(ascending=False).head(5).to_string())