from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
IN_PATH = SILVER_DIR / "pm25_hourly.parquet"
CLEAN_PATH = SILVER_DIR / "pm25_hourly_clean.parquet"
QUARANTINE_PATH = SILVER_DIR / "pm25_quarantine.csv"

MAX_VALID = 1000        # ug/m3, above this is treated as a fault
FLAT_RUN_MIN = 6        # identical consecutive hourly values
SPIKE_MIN_VALUE = 300   # ug/m3
SPIKE_RATIO = 4         # times the city median for the same hour
MIN_CITY_SENSORS = 3    # sensors reporting that hour, needed for the spike rule

RULES = ["range", "zero", "flat", "spike"]


def flat_mask(x, min_run):
    """True for every hour that belongs to a run of min_run or more identical values."""
    mask = np.zeros(len(x), dtype=bool)
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or not x[i] == x[start]:
            if x[start] == x[start] and i - start >= min_run:   # skips NaN runs
                mask[start:i] = True
            start = i
    return mask


df = pd.read_parquet(IN_PATH).sort_values(["sensor_id", "hour_start_utc"]).reset_index(drop=True)
observed = df["value"].notna()

flags = pd.DataFrame(index=df.index)
flags["range"] = (df["value"] < 0) | (df["value"] > MAX_VALID)
flags["zero"] = df["value"] == 0
flags["flat"] = df.groupby("sensor_id")["value"].transform(
    lambda s: pd.Series(flat_mask(s.to_numpy(), FLAT_RUN_MIN), index=s.index)
).astype(bool)

grp = df.groupby(["city", "hour_start_utc"])["value"]
city_median = grp.transform("median")
city_n = grp.transform("count")
flags["spike"] = (
    (df["value"] > SPIKE_MIN_VALUE)
    & (city_n >= MIN_CITY_SENSORS)
    & (df["value"] > SPIKE_RATIO * city_median)
)

flags = flags.fillna(False).astype(bool)
flags = flags.mul(observed, axis=0).astype(bool)       # only observed readings can be flagged
quarantined = flags.any(axis=1)

reasons = pd.Series("", index=df.index)
for rule in RULES:
    reasons[flags[rule]] += rule + ";"

# ---- Cleaned copy of silver: quarantined readings become missing ----
clean = df.copy()
clean["value_original"] = clean["value"]
clean.loc[quarantined, "value"] = np.nan
clean["is_missing"] = clean["value"].isna()
clean["quarantine_reason"] = reasons.where(quarantined, "")

quarantine = df.loc[quarantined, ["city", "sensor_id", "hour_start_utc", "value"]].copy()
quarantine["reasons"] = reasons[quarantined]

clean.to_parquet(CLEAN_PATH, index=False)
quarantine.to_csv(QUARANTINE_PATH, index=False)

# ---- Safety checks ----
assert int(quarantined.sum()) == len(quarantine)
assert int(observed.sum()) - int(quarantined.sum()) == int(clean["value"].notna().sum())
assert (clean.loc[~quarantined, "value"].fillna(-1) == df.loc[~quarantined, "value"].fillna(-1)).all()

# ---- Report ----
n_obs = int(observed.sum())
print("Observed readings:", n_obs)
print("Quarantined:", int(quarantined.sum()), f"({quarantined.sum() / n_obs:.2%} of observed)")
print()
print("Flagged by each rule (a reading can break several):")
print(flags.sum().to_string())
print()
print("Flagged by only this rule:")
for rule in RULES:
    others = [r for r in RULES if r != rule]
    print(f"  {rule:6}", int((flags[rule] & ~flags[others].any(axis=1)).sum()))
print()
print("Quarantined by city:")
print(df.loc[quarantined, "city"].value_counts().to_string())
print()
print("Sensors with the most quarantined hours:")
top = df.loc[quarantined].groupby(["sensor_id", "city"]).size().sort_values(ascending=False).head(8)
print(top.to_string())
print()
hi = df["value"] > 500
print("Hours above 500:", int(hi.sum()), "  of those quarantined:", int((hi & quarantined).sum()))
print()
print("Observed mean before: %.1f   after: %.1f"
      % (df["value"].mean(), clean["value"].mean()))
print("Completeness before: %.1f%%   after: %.1f%%"
      % (100 * df["value"].notna().mean(), 100 * clean["value"].notna().mean()))
print()
print("Saved:", CLEAN_PATH.name, "and", QUARANTINE_PATH.name)