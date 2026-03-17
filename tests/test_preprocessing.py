import pytest
import numpy as np
import pandas as pd
from src.preprocessing import preprocess_single_event

def test_preprocess_single_event_logic():
    """
    Test de robustesse : vérifie que le traitement unitaire 
    calcule correctement les ratios et aligne les colonnes.
    """
    # 1. Entrée JSON simulée
    sample_data = {
        "age_conducteur1": 20,
        "anciennete_permis1": 2,
        "din_vehicule": 100,
        "poids_vehicule": 1000,
        "marque_vehicule": "Renault",
        "sex_conducteur1": "M"
    }
    
    # 2. Colonnes attendues (simulées depuis feature_names.pkl)
    expected_features = [
        "age_conducteur1", 
        "ratio_puissance_poids", 
        "age_obtention_permis",
        "marque_vehicule_Renault",
        "marque_vehicule_Peugeot"
    ]
    
    # 3. Exécution
    output = preprocess_single_event(sample_data, expected_features)
    
    # 4. Vérifications (Assertions)
    assert output.shape == (1, 5), "La dimension de sortie est incorrecte"
    
    # Vérification du calcul du ratio (100 / (1000 + 1))
    # Le ratio est à l'index 1 dans expected_features
    assert round(output[0, 1], 4) == round(100/1001, 4)
    
    # Vérification de l'encodage One-Hot
    assert output[0, 3] == 1, "Renault devrait être à 1"
    assert output[0, 4] == 0, "Peugeot devrait être à 0 (colonne créée par alignement)"

def test_preprocess_missing_values():
    """Vérifie que le code ne plante pas si des champs sont absents[cite: 36]."""
    incomplete_data = {"age_conducteur1": 30}
    features = ["age_conducteur1", "ratio_puissance_poids"]
    
    try:
        output = preprocess_single_event(incomplete_data, features)
        assert output[0, 0] == 30
        assert output[0, 1] == 0 # Ratio absent donc mis à 0 par reindex
    except Exception as e:
        pytest.fail(f"Le preprocessing a planté sur données incomplètes : {e}")