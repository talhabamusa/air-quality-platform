import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest" / "pm25_sensors.csv"

url = URL.create(
    "postgresql+psycopg",
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost",
    port=5432,
    database=os.getenv("POSTGRES_DB"),
)
engine = create_engine(url)

quarantine = pd.read_csv(SILVER_DIR / "pm25_quarantine.csv")
quarantine["hour_start_utc"] = pd.to_datetime(quarantine["hour_start_utc"], utc=True)

tables = {
    "stg_pm25_hourly": pd.read_parquet(SILVER_DIR / "pm25_hourly_imputed_clean.parquet"),
    "stg_weather_hourly": pd.read_parquet(SILVER_DIR / "weather_hourly.parquet"),
    "stg_quarantine": quarantine,
    "stg_sensors": pd.read_csv(MANIFEST_PATH),
}

with engine.begin() as conn:
    print(conn.execute(text("select version()")).scalar())
    conn.execute(text("create schema if not exists staging"))
print()

for name, df in tables.items():
    df.to_sql(name, engine, schema="staging", if_exists="replace",
              index=False, chunksize=2000, method="multi")
    with engine.connect() as conn:
        n = conn.execute(text(f"select count(*) from staging.{name}")).scalar()
    assert n == len(df), f"{name}: loaded {n} rows, expected {len(df)}"
    print(f"{name:20} {n:>7} rows")

print()
print("stg_pm25_hourly by status:")
with engine.connect() as conn:
    rows = conn.execute(text(
        "select impute_method, count(*) from staging.stg_pm25_hourly "
        "group by 1 order by 2 desc"
    )).all()
for method, n in rows:
    print(f"  {method:24} {n:>7}")