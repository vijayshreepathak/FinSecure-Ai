import os
import json
from typing import Dict, Any
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test_fraud.db"

from app.main import app, model_service  # type: ignore
from app.db import init_db


class FakeModelService:
    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "probability": 0.9,
            "risk_score": 0.85,
            "label": 1,
            "threshold": 0.6,
            "iso_score_raw": 1.0,
            "iso_norm": 0.8,
            "ae_error": 0.2,
            "ae_norm": 0.3,
            "explanations": [
                {"feature": "Amount", "value": float(data.get("Amount", 0.0)), "contribution": 0.5}
            ],
        }


def override_model_service() -> None:
    from app import main as main_module
    main_module.model_service = FakeModelService()


override_model_service()
init_db()
client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def sample_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"Time": 0.0, "Amount": 100.0}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    payload["Class"] = 1
    return payload


def test_predict_single() -> None:
    resp = client.post("/predict", json=sample_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data
    assert "risk_score" in data
    assert "label" in data
    assert "explanations" in data
    assert isinstance(data["explanations"], list)


def test_predict_batch_json() -> None:
    items = [sample_payload(), sample_payload()]
    resp = client.post("/predict_batch", json=items)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_alerts_and_metrics() -> None:
    _ = client.post("/predict", json=sample_payload())
    alerts = client.get("/alerts")
    assert alerts.status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    m = metrics.json()
    assert "total" in m
    assert "fraud_rate" in m


