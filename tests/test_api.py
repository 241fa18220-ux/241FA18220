import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
SAMPLE = {"country": "Testland", "confirmed": 50000, "deaths": 1200, "new_cases": 400, "new_deaths": 10,
          "new_recovered": 300, "deaths_per_100_cases": 2.4, "confirmed_last_week": 47000, "who_region": "Europe"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_recovered"] >= 0 and body["country"] == "Testland"


def test_bigger_outbreak_predicts_more_recoveries():
    small = client.post("/predict", json=SAMPLE).json()["predicted_recovered"]
    big = client.post("/predict", json={**SAMPLE, "confirmed": 900000, "deaths": 30000, "new_cases": 20000,
                                        "new_recovered": 15000, "confirmed_last_week": 850000}).json()["predicted_recovered"]
    assert big > small


def test_batch():
    r = client.post("/predict_batch", json={"countries": [SAMPLE, {**SAMPLE, "country": "B", "who_region": "Africa"}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


@pytest.mark.parametrize("bad", [{**SAMPLE, "confirmed": -1}, {**SAMPLE, "who_region": "Atlantis"},
                                 {k: v for k, v in SAMPLE.items() if k != "deaths"}])
def test_invalid_payload_rejected(bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_empty_batch_rejected():
    assert client.post("/predict_batch", json={"countries": []}).status_code == 422
