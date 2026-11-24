from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "dashboard" / "app_data"
TIERS = {"observed", "imputed", "imputed_low_confidence", "missing"}


def read(name):
    return pd.read_csv(DATA / name)


def test_all_snapshot_files_exist():
    for name in ["mart_city_daily.csv", "forecast_latest.csv", "quality_tiers.csv",
                 "hourly_profile.csv", "model_runs.csv", "backtest_predictions.csv", "meta.json"]:
        assert (DATA / name).exists(), name


def test_mart_respects_the_sensor_minimum_and_has_unique_city_days():
    mart = read("mart_city_daily.csv")
    assert not mart.duplicated(["city_key", "date_local"]).any()
    assert (mart["n_sensors"] >= 3).all()
    assert mart["pm25_mean_ugm3"].between(0, 1000).all()
    assert (mart["weather_hours"] == 24).all()


def test_forecast_ranges_are_ordered():
    fc = read("forecast_latest.csv")
    assert len(fc) == 5
    assert (fc["forecast_p10"] <= fc["forecast_median"]).all()
    assert (fc["forecast_median"] <= fc["forecast_p90"]).all()


def test_quality_tiers_are_the_known_ones():
    tiers = read("quality_tiers.csv")
    assert set(tiers["quality_tier"]) <= TIERS


def test_backtest_intervals_are_ordered():
    bt = read("backtest_predictions.csv")
    model = bt.dropna(subset=["model_lo", "model_hi"])
    band = bt.dropna(subset=["band_lo", "band_hi"])
    assert (model["model_lo"] <= model["model_hi"]).all()
    assert (band["band_lo"] <= band["band_hi"]).all()