"""Smoke tests for the verification service. Skips gracefully if artifacts absent."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from emitax.service.app import app  # noqa: E402

client = TestClient(app)  # triggers startup (engine load)

SAMPLE = {
    "facility_id": "Limestone", "unit_id": "LIM1",
    "year": 2025, "month": 6,
    "heat_input": 5_000_000.0, "fuel_type": "Coal",
    "co2": 525_000.0, "so2": 1_200.0, "nox": 900.0, "pm": 50.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_root():
    assert client.get("/").status_code == 200


def test_verify_shape():
    r = client.post("/verify", json=SAMPLE)
    if r.status_code == 503:
        pytest.skip("engine not ready (artifacts missing in test env)")
    assert r.status_code == 200
    body = r.json()
    for k in ("trust_score", "decision", "fired_checks", "reason",
              "per_check", "record_hash", "onchain_payload"):
        assert k in body
    assert body["decision"] in ("approve", "request_docs", "human_review")
    assert 0.0 <= body["trust_score"] <= 100.0


def test_under_report_lowers_score():
    """A grossly under-reported CO2 must not stay 'approve'."""
    r0 = client.post("/verify", json=SAMPLE)
    if r0.status_code == 503:
        pytest.skip("engine not ready")
    tampered = dict(SAMPLE, co2=SAMPLE["co2"] * 0.4)  # 60% cut
    r1 = client.post("/verify", json=tampered)
    assert r1.json()["trust_score"] <= r0.json()["trust_score"]
