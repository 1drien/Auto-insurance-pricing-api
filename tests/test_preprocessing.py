import pytest
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from src.preprocessing import preprocess_single_event


def test_preprocess_single_event_logic() -> None:
    """
    Verifies that the single-event preprocessing correctly
    computes feature engineering ratios and aligns columns.
    """
    # 1. Simulated JSON input (what the API would receive)
    sample_data: dict = {
        "age_conducteur1": 20,
        "anciennete_permis1": 2,
        "din_vehicule": 100,
        "poids_vehicule": 1000,
        "marque_vehicule": "Renault",
        "sex_conducteur1": "M"
    }

    # 2. Expected model columns (simulates feature_names.pkl)
    expected_features: list[str] = [
        "age_conducteur1",
        "ratio_puissance_poids",
        "age_obtention_permis",
        "marque_vehicule_Renault",
        "marque_vehicule_Peugeot"
    ]

    # 3. Run preprocessing
    output: NDArray = preprocess_single_event(sample_data, expected_features)

    # 4. Check output shape: 1 row, 5 columns
    assert output.shape == (1, 5), "Output shape should be (1, 5)"

    # 5. Check power-to-weight ratio: 100 / (1000 + 1)
    assert round(output[0, 1], 4) == round(100 / 1001, 4)

    # 6. Check one-hot encoding: Renault = 1, Peugeot = 0
    assert output[0, 3] == 1, "Renault column should be 1"
    assert output[0, 4] == 0, "Peugeot column should be 0 (filled by reindex)"


def test_preprocess_missing_values() -> None:
    """Verifies that preprocessing handles incomplete input without crashing."""
    incomplete_data: dict = {"age_conducteur1": 30}
    features: list[str] = ["age_conducteur1", "ratio_puissance_poids"]

    try:
        output: NDArray = preprocess_single_event(incomplete_data, features)
        assert output[0, 0] == 30, "Age should be preserved"
        assert output[0, 1] == 0, "Missing ratio should default to 0"
    except Exception as e:
        pytest.fail(f"Preprocessing crashed on incomplete input: {e}")