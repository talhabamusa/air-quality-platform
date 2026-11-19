import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs"
OUT_DIR.mkdir(exist_ok=True)

engine = create_engine(URL.create(
    "postgresql+psycopg",
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost", port=5432,
    database=os.getenv("POSTGRES_DB"),
))


def q(sql):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150)
    plt.close(fig)


# ---------- Load the daily mart ----------
daily = q("select * from gold.mart_city_daily")
daily["date_local"] = pd.to_datetime(daily["date_local"])
daily["month"] = daily["date_local"].dt.to_period("M").dt.to_timestamp()
for c in daily.columns:
    if c not in ("city_key", "date_local", "month"):
        daily[c] = daily[c].astype(float)

n_fact = int(q("select count(*) as n from gold.fact_air_quality")["n"].iloc[0])
print("Rows in fact_air_quality:", n_fact)
print("Rows in mart_city_daily:", len(daily), " from", daily["date_local"].min().date(),
      "to", daily["date_local"].max().date())
print("Days per city:", daily.groupby("city_key").size().to_dict())

# ---------- 1. Monthly PM2.5 by city ----------
monthly = daily.groupby(["city_key", "month"]).agg(
    pm25=("pm25_mean_ugm3", "mean"), days=("pm25_mean_ugm3", "count")).reset_index()
monthly = monthly[monthly["days"] >= 15]            # skip months with too few days
pivot = monthly.pivot(index="month", columns="city_key", values="pm25")

fig, ax = plt.subplots(figsize=(10, 5))
for city in pivot.columns:
    ax.plot(pivot.index, pivot[city], marker="o", label=city.title())
ax.set_ylabel("PM2.5, monthly mean of daily city means (ug/m3)")
ax.set_title("Monthly PM2.5 by city")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
save(fig, "monthly_pm25.png")

print()
print("1. Monthly mean PM2.5 (ug/m3), months with at least 15 days:")
print(pivot.round(1).to_string())

# ---------- 2. Daily cycle by local hour ----------
hourly = q("""
select f.city_key, h.hour_of_day_local as hour, avg(f.pm25_ugm3) as pm25
from gold.fact_air_quality f
join gold.dim_hour h on h.hour_key = f.hour_key
where f.quality_tier in ('observed', 'imputed')
group by 1, 2
""")
prof = hourly.pivot(index="hour", columns="city_key", values="pm25")
rel = prof / prof.mean()

fig, ax = plt.subplots(figsize=(10, 5))
for city in rel.columns:
    ax.plot(rel.index, rel[city], marker="o", label=city.title())
ax.axhline(1.0, color="grey", linewidth=0.8)
ax.set_xlabel("Hour of day (India Standard Time, start of the hour)")
ax.set_ylabel("PM2.5 relative to the city's own average")
ax.set_title("Daily cycle of PM2.5")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "hourly_profile.png")

print()
print("2. Daily cycle (local hour of the highest and lowest PM2.5, and how far from the city average):")
print(pd.DataFrame({
    "peak_hour": prof.idxmax(), "peak_vs_avg": rel.max().round(2),
    "trough_hour": prof.idxmin(), "trough_vs_avg": rel.min().round(2),
}).to_string())

# ---------- 3. Weekday vs weekend ----------
wk = q("""
select f.city_key, h.is_weekend, avg(f.pm25_ugm3) as pm25
from gold.fact_air_quality f
join gold.dim_hour h on h.hour_key = f.hour_key
where f.quality_tier in ('observed', 'imputed')
group by 1, 2
""")
wk = wk.pivot(index="city_key", columns="is_weekend", values="pm25")
wk = wk.rename(columns={False: "weekday", True: "weekend"})
wk["weekend_vs_weekday_pct"] = (100 * (wk["weekend"] / wk["weekday"] - 1)).round(1)
print()
print("3. Weekday vs weekend PM2.5 (ug/m3):")
print(wk.round(1).to_string())

# ---------- 4. Rainy vs dry days ----------
daily["rainy"] = daily["precip_total_mm"] >= 1
mean_by = daily.pivot_table(index="city_key", columns="rainy", values="pm25_mean_ugm3",
                            aggfunc="mean").rename(columns={False: "dry", True: "rainy"})
days_by = daily.pivot_table(index="city_key", columns="rainy", values="pm25_mean_ugm3",
                            aggfunc="count").rename(columns={False: "dry_days", True: "rainy_days"})
rain = mean_by.join(days_by)
rain["rainy_minus_dry"] = (rain["rainy"] - rain["dry"]).round(1)

fig, ax = plt.subplots(figsize=(8, 5))
mean_by[["dry", "rainy"]].plot(kind="bar", ax=ax)
ax.set_ylabel("Mean daily PM2.5 (ug/m3)")
ax.set_title("PM2.5 on dry days vs rainy days (1 mm or more)")
ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=0)
ax.grid(axis="y", alpha=0.3)
save(fig, "rain_effect.png")

diffs = []
for (city, month), g in daily.groupby(["city_key", "month"]):
    wet, dry = g[g["rainy"]], g[~g["rainy"]]
    if len(wet) >= 3 and len(dry) >= 3:
        diffs.append({"city_key": city,
                      "diff": wet["pm25_mean_ugm3"].mean() - dry["pm25_mean_ugm3"].mean()})
print()
print("4. Rainy (1 mm or more) vs dry days, whole year:")
print(rain.round(1).to_string())
if diffs:
    within = pd.DataFrame(diffs).groupby("city_key")["diff"].agg(["mean", "count"])
    within.columns = ["rainy_minus_dry_within_month", "months_used"]
    print()
    print("   Same comparison inside each month (needs 3+ rainy and 3+ dry days), averaged:")
    print(within.round(1).to_string())

# ---------- 5. Correlations with weather ----------
cols = ["temp_mean_c", "humidity_mean_pct", "precip_total_mm", "wind_mean_ms"]
corr = pd.DataFrame({
    city: g[["pm25_mean_ugm3"] + cols].corr(method="spearman")["pm25_mean_ugm3"].drop("pm25_mean_ugm3")
    for city, g in daily.groupby("city_key")
}).T
print()
print("5. Spearman correlation of daily PM2.5 with daily weather:")
print(corr.round(2).to_string())

# ---------- 6. How much of the data is observed vs filled vs missing ----------
tiers = q("select city_key, quality_tier, count(*) as n from gold.fact_air_quality group by 1, 2")
share = tiers.pivot(index="city_key", columns="quality_tier", values="n").fillna(0)
share = share.div(share.sum(axis=1), axis=0) * 100
order = [t for t in ["observed", "imputed", "imputed_low_confidence", "missing"] if t in share.columns]

fig, ax = plt.subplots(figsize=(8, 5))
share[order].plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("% of sensor-hours")
ax.set_title("Data quality tiers by city")
ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, "data_quality.png")
print()
print("6. Share of sensor-hours by quality tier (%):")
print(share[order].round(1).to_string())

print()
print("Charts saved to docs: monthly_pm25.png, hourly_profile.png, rain_effect.png, data_quality.png")