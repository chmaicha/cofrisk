from fastapi.testclient import TestClient

from cofrisk.api.main import app
from cofrisk.data.validation import EXPECTED_FEATURES


client = TestClient(app)


def make_valid_features():
    return {
        feature: 0.1
        for feature in EXPECTED_FEATURES
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post(
        "/predict",
        json={"features": make_valid_features()},
    )

    assert response.status_code == 200

    data = response.json()

    assert "default_probability" in data
    assert "risk_class" in data

    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["risk_class"] in [0, 1]

def test_predict_rejects_missing_features():

    features = make_valid_features()

    del features["Attr27"]

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"]["error"] == "Missing features"
    assert "Attr27" in data["detail"]["features"]

def test_predict_rejects_unknown_feature():

    features = make_valid_features()

    features["Attr999"] = 0.1

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422