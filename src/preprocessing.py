import pandas as pd
import numpy as np
from numpy.typing import NDArray


def _apply_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logique interne de transformation (Feature Engineering + Nettoyage).
    Appliquée de manière identique pour le Batch et l'Unitaire.
    """
    if 'din_vehicule' in df.columns and 'poids_vehicule' in df.columns:
        df['ratio_puissance_poids'] = df['din_vehicule'] / (df['poids_vehicule'] + 1)

    if 'vitesse_vehicule' in df.columns and 'poids_vehicule' in df.columns:
        df['ratio_vitesse_poids'] = df['vitesse_vehicule'] / (df['poids_vehicule'] + 1)

    if 'age_conducteur1' in df.columns and 'anciennete_permis1' in df.columns:
        df['age_obtention_permis'] = df['age_conducteur1'] - df['anciennete_permis1']

    if 'age_conducteur1' in df.columns and 'din_vehicule' in df.columns:
        df['risque_jeune_sportif'] = (1 / (df['age_conducteur1'] + 1)) * df['din_vehicule']

    if 'prix_vehicule' in df.columns:
        df['log_prix_vehicule'] = np.log1p(df['prix_vehicule'])

    if 'sex_conducteur2' in df.columns:
        df['sex_conducteur2'] = df['sex_conducteur2'].fillna('NoDriver')

    mappings: dict[str, tuple[list, list]] = {
        'sex_conducteur1': (['M', 'F'], [0, 1]),
        'type_vehicule': (['Tourism', 'Commercial'], [0, 1]),
        'utilisation': (['Retired', 'WorkPrivate', 'Professional', 'AllTrips'], [0, 1, 2, 4]),
        'freq_paiement': (['Monthly', 'Quarterly', 'Biannual', 'Yearly'], [0, 1, 2, 3])
    }

    for col, (vals, codes) in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(vals, codes).infer_objects(copy=False)

    cols_to_encode: list[str] = [
        'marque_vehicule', 'modele_vehicule', 'sex_conducteur2',
        'type_contrat', 'paiement', 'conducteur2', 'essence_vehicule'
    ]
    exist_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=exist_cols, dtype=int)

    return df


def load_and_clean_common_data(filepath: str) -> pd.DataFrame:
    """Charge un CSV et applique la logique de nettoyage."""
    print(f"Chargement des données : {filepath}")
    df = pd.read_csv(filepath)
    return _apply_logic(df)


def preprocess_single_event(data_dict: dict, model_features: list[str]) -> NDArray:
    """
    Transforme un dictionnaire JSON en vecteur NumPy pour l'API.

    Args:
        data_dict: Dictionnaire contenant les données d'une observation.
        model_features: Liste des noms de colonnes attendus par le modèle.

    Returns:
        Matrice NumPy de shape (1, n_features) alignée sur le modèle.
    """
    df = pd.DataFrame([data_dict])
    df_processed = _apply_logic(df)
    df_aligned = df_processed.reindex(columns=model_features, fill_value=0)
    return df_aligned.values


def prepare_for_severity(df: pd.DataFrame) -> tuple[NDArray, NDArray, pd.Index]:
    """Prépare la matrice pour le modèle Sévérité."""
    df_sev = df[df['nombre_sinistres'] > 0].copy()
    y = np.log1p(df_sev['montant_sinistre']).values

    cols_to_drop: list[str] = [
        'montant_sinistre', 'nombre_sinistres',
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    X_df = df_sev.drop(columns=cols_to_drop, errors='ignore')
    return X_df.values, y, X_df.columns


def prepare_for_frequency(df: pd.DataFrame) -> tuple[NDArray, NDArray, pd.Index]:
    """Prépare la matrice pour le modèle Fréquence."""
    y = (df['nombre_sinistres'] > 0).astype(int).values

    cols_to_drop: list[str] = [
        'montant_sinistre', 'nombre_sinistres',
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    return X_df.values, y, X_df.columns


def prepare_for_inference(df: pd.DataFrame, model_features: list[str]) -> tuple[NDArray, NDArray]:
    """Alignement pour l'inférence batch classique."""
    if 'id_contrat' in df.columns:
        ids = df['id_contrat'].values
    elif 'index' in df.columns:
        ids = df['index'].values
    else:
        ids = df.iloc[:, 0].values

    df_aligned = df.reindex(columns=model_features, fill_value=0)
    return df_aligned.values, ids