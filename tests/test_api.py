import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Données de test réutilisables
VALID_PAYLOAD = {
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


def test_health():
    """Vérifie que /health retourne 200 et le bon statut."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_frequency():
    """Vérifie que /predict_frequency retourne une probabilité valide."""
    response = client.post("/predict_frequency", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    prob = data["Predicted_Claim_Probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_amount():
    """Vérifie que /predict_amount retourne un montant positif."""
    response = client.post("/predict_amount", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    amount = data["Estimated_Claim_Cost_EUR"]
    assert amount > 0


def test_predict_full():
    """Vérifie que /predict retourne les 4 champs attendus."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_claim_frequency" in data
    assert "estimated_severity_eur" in data
    assert "technical_pure_premium_eur" in data
    assert "final_total_premium_ttc_eur" in data
    assert data["predicted_claim_frequency"] >= 0
    assert data["estimated_severity_eur"] > 0
    assert data["final_total_premium_ttc_eur"] >= 0


def test_predict_invalid_input():
    """Vérifie que l'API rejette un JSON incomplet avec une erreur 422."""
    bad_payload = {"age_conducteur1": 24}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_invalid_age():
    """Vérifie que l'API rejette un âge inférieur à 18."""
    payload = VALID_PAYLOAD.copy()
    payload["age_conducteur1"] = 10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422