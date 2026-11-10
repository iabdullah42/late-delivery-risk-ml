import pytest

from src.train import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")

ORDER = {"order_datetime": "2018-01-15T15:30:00", "shipping_mode": "Same Day",
         "market": "Europe", "region": "Western Europe", "country": "Irlanda",
         "segment": "Consumer", "payment_type": "DEBIT", "main_department": "Fan Shop",
         "n_items": 2, "n_departments": 1, "total_qty": 3, "order_value": 250.0,
         "avg_discount_rate": 0.1}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_afternoon_same_day_is_flagged(client):
    body = client.post("/predict", json=ORDER).json()
    assert body["flag"] is True and body["late_probability"] > 0.9


def test_morning_same_day_is_not_flagged(client):
    body = client.post("/predict", json={**ORDER, "order_datetime": "2018-01-15T09:30:00"}).json()
    assert body["flag"] is False and body["late_probability"] < 0.2


def test_bad_shipping_mode_is_rejected(client):
    assert client.post("/predict", json={**ORDER, "shipping_mode": "Overnight"}).status_code == 422