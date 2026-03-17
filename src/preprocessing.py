import pandas as pd
import numpy as np

# Configuration globale pour éviter les avertissements de types
pd.set_option('future.no_silent_downcasting', True)

def _apply_logic(df):
    """
    Logique interne de transformation (Feature Engineering + Nettoyage).
    Appliquée de manière identique pour le Batch et l'Unitaire[cite: 31, 33].
    """
    # 1. INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING) [cite: 18]
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

    # 2. NETTOYAGE ET ENCODAGE 
    if 'sex_conducteur2' in df.columns:
        df['sex_conducteur2'] = df['sex_conducteur2'].fillna('NoDriver')

    # Encodage ordinal manuel
    mappings = {
        'sex_conducteur1': (['M', 'F'], [0, 1]),
        'type_vehicule': (['Tourism', 'Commercial'], [0, 1]),
        'utilisation': (['Retired', 'WorkPrivate', 'Professional', 'AllTrips'], [0, 1, 2, 4]),
        'freq_paiement': (['Monthly', 'Quarterly', 'Biannual', 'Yearly'], [0, 1, 2, 3])
    }
    
    for col, (vals, codes) in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(vals, codes).infer_objects(copy=False)

    # 3. ENCODAGE DISJONCTIF (ONE-HOT) 
    cols_to_encode = [
        'marque_vehicule', 'modele_vehicule', 'sex_conducteur2', 
        'type_contrat', 'paiement', 'conducteur2', 'essence_vehicule'
    ]
    exist_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=exist_cols, dtype=int)
    
    return df

def load_and_clean_common_data(filepath):
    """Charge un CSV et applique la logique de nettoyage[cite: 18]."""
    print(f"Chargement des données : {filepath}")
    df = pd.read_csv(filepath)
    return _apply_logic(df)

def preprocess_single_event(data_dict, model_features):
    """
    Transforme un dictionnaire JSON en vecteur pour l'API (Étape 2.2 du TD)[cite: 32, 39].
    """
    # Conversion du dictionnaire en DataFrame d'une ligne
    df = pd.DataFrame([data_dict])
    
    # Application de la logique commune
    df_processed = _apply_logic(df)
    
    # Alignement strict sur les colonnes d'entraînement (remplit de 0 si absent)
    df_aligned = df_processed.reindex(columns=model_features, fill_value=0)
    
    return df_aligned.values

def prepare_for_severity(df):
    """Prépare la matrice pour le modèle Sévérité[cite: 19]."""
    df_sev = df[df['nombre_sinistres'] > 0].copy()
    y = np.log1p(df_sev['montant_sinistre']).values
    
    cols_to_drop = [
        'montant_sinistre', 'nombre_sinistres', 
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    X_df = df_sev.drop(columns=cols_to_drop, errors='ignore')
    return X_df.values, y, X_df.columns

def prepare_for_frequency(df):
    """Prépare la matrice pour le modèle Fréquence[cite: 19]."""
    y = (df['nombre_sinistres'] > 0).astype(int).values
    
    cols_to_drop = [
        'montant_sinistre', 'nombre_sinistres', 
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    return X_df.values, y, X_df.columns

def prepare_for_inference(df, model_features):
    """Alignement pour l'inférence batch classique[cite: 16]."""
    if 'id_contrat' in df.columns: ids = df['id_contrat'].values
    elif 'index' in df.columns: ids = df['index'].values
    else: ids = df.iloc[:, 0].values

    df_aligned = df.reindex(columns=model_features, fill_value=0)
    return df_aligned.values, ids