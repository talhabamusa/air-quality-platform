import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "ml" / "models" / "pm25_daily_quantile_models.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="the trained model is not in the repository")
sys.path.insert(0, str(ROOT))

PAYLOAD = {"city": "delhi", "forecast_date": "2026-11-05", "pm25_today": 292.3,
           "pm25_6_days_ago": 180.0, "pm25_7day_mean": 230.0}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and "delhi" in r.json()["cities"]


def test_forecast_is_ordered_and_keeps_the_persistence_value(client):
    r = client.post("/forecast", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["forecast_p10"] <= body["forecast_median"] <= body["forecast_p90"]
    assert body["persistence_forecast"] == 292.3


def test_unknown_city_is_rejected(client):
    assert client.post("/forecast", json={**PAYLOAD, "city": "paris"}).status_code == 422


def test_impossible_value_is_rejected(client):
    assert client.post("/forecast", json={**PAYLOAD, "pm25_today": -5}).status_code == 422