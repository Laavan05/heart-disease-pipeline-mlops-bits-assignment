"""Endpoint tests for the FastAPI prediction service."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as api
from app.main import app

SAMPLE_REQUEST = {
    "age": 63.0,
    "sex": 1,
    "cp": 1,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0.0,
    "thal": 6.0,
}
PREDICTION_KEYS = {"prediction", "prediction_label", "probability"}


@pytest.fixture
def client(
    model_artifact_paths: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """Create a client with model artifacts loaded by the app lifespan."""
    model_path, preprocessor_path = model_artifact_paths
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    monkeypatch.setattr(api, "PREPROCESSOR_PATH", preprocessor_path)
    with TestClient(app) as test_client:
        yield test_client


def test_get_root_returns_api_information(client: TestClient) -> None:
    """Return API metadata from the root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Heart Disease Prediction API"
    assert body["health"] == "/health"
    assert body["prediction"] == "/predict"


def test_get_health_reports_loaded_artifacts(client: TestClient) -> None:
    """Report a healthy service after application startup."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_http_requests_are_logged(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log method, path, status, and duration for handled requests."""
    caplog.set_level(logging.INFO, logger=api.LOGGER.name)

    response = client.get("/health")

    assert response.status_code == 200
    assert any(
        "HTTP GET /health status=200 duration_ms=" in record.message
        for record in caplog.records
    )


def test_get_metrics_returns_prometheus_text(client: TestClient) -> None:
    """Expose request instrumentation in Prometheus text format."""
    client.get("/")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text


def test_post_predict_returns_valid_prediction(client: TestClient) -> None:
    """Return class, label, and probability for a valid request."""
    response = client.post("/predict", json=SAMPLE_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert PREDICTION_KEYS <= body.keys()
    assert body["prediction"] in {0, 1}
    assert body["prediction_label"] in {
        "No Heart Disease",
        "Heart Disease",
    }
    assert 0.0 <= body["probability"] <= 1.0
