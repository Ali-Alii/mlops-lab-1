import numpy as np
from fastapi.testclient import TestClient

from src.food11 import service


class FakeModel:
    def predict(self, x):
        assert x.shape == (1, 3, 128, 128)
        return np.array([[5.0] + [0.0] * 10])


def test_health_without_model(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(service.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_version"] == "skipped"
        assert (
            client.post(
                "/predict", files={"file": ("x.jpg", b"abc", "image/jpeg")}
            ).status_code
            == 400
        )


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(service.app) as client:
        monkeypatch.setattr(service, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as image:
            response = client.post(
                "/predict", files={"file": ("bread.jpg", image, "image/jpeg")}
            )
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_category"] == "Bread"
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 3


def test_predict_without_model_is_unavailable(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with (
        TestClient(service.app) as client,
        open("tests/fixtures/bread.jpg", "rb") as image,
    ):
        response = client.post(
            "/predict", files={"file": ("bread.jpg", image, "image/jpeg")}
        )
    assert response.status_code == 503
