from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "models" / "pm25_daily_quantile_models.joblib"
bundle = joblib.load(MODEL_PATH)
MODELS, FEATURES = bundle["models"], bundle["features"]
QUANTILES, CITIES = list(bundle["quantiles"]), list(bundle["cities"])
DATA_WINDOW = bundle["data_window"]

app = FastAPI(
    title="PM2.5 next-day forecast",
    description=("Forecasts tomorrow's daily mean PM2.5 for five Indian cities, with an 80% range. "
                 "In backtests the model did not clearly beat 'tomorrow equals today' on average error; "
                 "its value is the prediction range."),
    version="0.1.0",
)


def nan_if_none(x):
    return float("nan") if x is None else float(x)


class ForecastRequest(BaseModel):
    city: str = Field(description="One of the cities listed by /health")
    forecast_date: date = Field(description="The day being forecast (its day of week is used)")
    pm25_today: float = Field(ge=0, le=1000, description="Today's daily mean PM2.5 (ug/m3)")
    pm25_6_days_ago: float = Field(ge=0, le=1000, description="Daily mean PM2.5 six days before today")
    pm25_7day_mean: float = Field(ge=0, le=1000, description="Mean of the last 7 daily values, including today")
    pm25_yesterday: float | None = Field(default=None, ge=0, le=1000)
    pm25_2_days_ago: float | None = Field(default=None, ge=0, le=1000)
    temp_c: float | None = Field(default=None, description="Today's mean temperature")
    humidity_pct: float | None = Field(default=None, ge=0, le=100)
    precip_mm: float | None = Field(default=None, ge=0, description="Today's total precipitation")
    wind_ms: float | None = Field(default=None, ge=0, description="Today's mean wind speed")

    model_config = {"json_schema_extra": {"example": {
        "city": "delhi", "forecast_date": "2026-11-05", "pm25_today": 292.3,
        "pm25_yesterday": 270.0, "pm25_2_days_ago": 250.0, "pm25_6_days_ago": 180.0,
        "pm25_7day_mean": 230.0, "temp_c": 21.0, "humidity_pct": 60.0, "precip_mm": 0.0, "wind_ms": 1.5,
    }}}


class ForecastResponse(BaseModel):
    city: str
    forecast_date: date
    persistence_forecast: float
    forecast_p10: float
    forecast_median: float
    forecast_p90: float
    model_data_window: str
    note: str


@app.get("/health")
def health():
    return {"status": "ok", "model_data_window": DATA_WINDOW, "cities": CITIES, "quantiles": QUANTILES}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.city not in CITIES:
        raise HTTPException(status_code=422, detail=f"Unknown city '{req.city}'. Use one of {CITIES}.")
    lag0, lag1 = req.pm25_today, nan_if_none(req.pm25_yesterday)
    row = {
        "city_key": req.city,
        "pm25_lag0": lag0,
        "pm25_lag1": lag1,
        "pm25_lag2": nan_if_none(req.pm25_2_days_ago),
        "pm25_lag6": req.pm25_6_days_ago,
        "pm25_roll7": req.pm25_7day_mean,
        "chg_1d": np.log1p(lag0) - np.log1p(lag1),
        "vs_mean7": np.log1p(lag0) - np.log1p(req.pm25_7day_mean),
        "temp_mean_c": nan_if_none(req.temp_c),
        "humidity_mean_pct": nan_if_none(req.humidity_pct),
        "precip_total_mm": nan_if_none(req.precip_mm),
        "wind_mean_ms": nan_if_none(req.wind_ms),
        "dow_target": req.forecast_date.weekday(),
    }
    X = pd.DataFrame([row])[FEATURES]
    X["city_key"] = pd.Categorical(X["city_key"], categories=CITIES)
    deltas = np.sort([MODELS[q].predict(X)[0] for q in QUANTILES])
    p10, median, p90 = np.clip(np.expm1(np.log1p(lag0) + deltas), 0, None)
    return ForecastResponse(
        city=req.city, forecast_date=req.forecast_date,
        persistence_forecast=round(lag0, 1),
        forecast_p10=round(float(p10), 1), forecast_median=round(float(median), 1),
        forecast_p90=round(float(p90), 1),
        model_data_window=DATA_WINDOW,
        note="80% range from quantile regression on the change from today's value.",
    )