import pytest
from fastapi.testclient import TestClient
from app import app

client: TestClient = TestClient(app)

# Reusable test payload (simulates a valid API request)
VALID_PAYLOAD: dict = {
    "age_conducteur1": 24,
    "anciennete_permis1": 2,
    "sex_conducteur1": "M",
    "din_vehicule": 130,
    "poids_vehicule": 1100,
    "utilisation": "WorkPrivate",
    "marque_vehicule": "Renault",
    "prix_vehicule": 25000.0,
    "type_vehicule": "Tourism",
    "freq_paiement": "Monthly"
}


def test_health() -> None:
    """Verifies that /health returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data: dict = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_frequency() -> None:
    """Verifies that /predict_frequency returns a valid probability."""
    response = client.post("/predict_frequency", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data: dict = response.json()
    prob: float = data["Predicted_Claim_Probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_amount() -> None:
    """Verifies that /predict_amount returns a positive amount."""
    response = client.post("/predict_amount", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data: dict = response.json()
    amount: float = data["Estimated_Claim_Cost_EUR"]
    assert amount > 0


def test_predict_full() -> None:
    """Verifies that /predict returns all 4 expected fields."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data: dict = response.json()
    assert "predicted_claim_frequency" in data
    assert "estimated_severity_eur" in data
    assert "technical_pure_premium_eur" in data
    assert "final_total_premium_ttc_eur" in data
    assert data["predicted_claim_frequency"] >= 0
    assert data["estimated_severity_eur"] > 0
    assert data["final_total_premium_ttc_eur"] >= 0


def test_predict_invalid_input() -> None:
    """Verifies that the API rejects incomplete JSON with a 422 error."""
    bad_payload: dict = {"age_conducteur1": 24}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_age() -> None:
    """Verifies that the API rejects an age below 18."""
    payload: dict = VALID_PAYLOAD.copy()
    payload["age_conducteur1"] = 10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422