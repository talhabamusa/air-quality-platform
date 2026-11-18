from pathlib import Path

import pandas as pd

silver = Path(__file__).resolve().parents[1] / "data" / "silver"

q = pd.read_csv(silver / "pm25_quarantine.csv")
q["hour_start_utc"] = pd.to_datetime(q["hour_start_utc"], utc=True)
q["month"] = q["hour_start_utc"].dt.strftime("%Y-%m")

df = pd.read_parquet(silver / "pm25_hourly.parquet", columns=["hour_start_utc", "value"])
df = df[df["value"].notna()].copy()
df["month"] = pd.to_datetime(df["hour_start_utc"], utc=True).dt.strftime("%Y-%m")

table = pd.concat([df.groupby("month").size().rename("observed"),
                   q.groupby("month").size().rename("quarantined")], axis=1).fillna(0).astype(int)
table["pct"] = (100 * table["quarantined"] / table["observed"]).round(2)
print("Quarantined readings by month:")
print(table.to_string())

rules = ["zero", "flat", "spike", "range", "dead_sensor"]
for rule in rules:
    q[rule] = q["reasons"].str.contains(rule)
print()
print("Readings flagged by each rule, by month:")
print(q.groupby("month")[rules].sum().to_string())