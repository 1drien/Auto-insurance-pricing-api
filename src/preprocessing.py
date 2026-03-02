import pandas as pd
import numpy as np

# Configuration globale pour anticiper les évolutions futures de la bibliothèque Pandas
# et éviter les avertissements liés aux conversions de types implicites.
pd.set_option('future.no_silent_downcasting', True)

def load_and_clean_common_data(filepath):
    """
    Charge le jeu de données brut, effectue la création de nouvelles variables 
    (Feature Engineering) et applique les procédures de nettoyage standards.
    
    Paramètres :
    - filepath (str) : Chemin d'accès au fichier CSV.
    
    Retourne :
    - df (DataFrame) : Le jeu de données enrichi et nettoyé.
    """
    print(f"Chargement et traitement initial des données depuis : {filepath}")
    df = pd.read_csv(filepath)

    # =========================================================
    # PARTIE A : INGÉNIERIE DES CARACTÉRISTIQUES (FEATURE ENGINEERING)
    # =========================================================
    # Ces transformations sont appliquées avant le nettoyage pour exploiter
    # les distributions brutes des variables physiques et comportementales.
    
    # 1. Ratio Puissance/Poids : Indique la nervosité du véhicule
    if 'din_vehicule' in df.columns and 'poids_vehicule' in df.columns:
        df['ratio_puissance_poids'] = df['din_vehicule'] / (df['poids_vehicule'] + 1)
        
    # 2. Intensité d'usage potentielle : Rapport entre la vitesse maximale et le poids
    if 'vitesse_vehicule' in df.columns and 'poids_vehicule' in df.columns:
        df['ratio_vitesse_poids'] = df['vitesse_vehicule'] / (df['poids_vehicule'] + 1)
        
    # 3. Expérience de conduite relative : Âge du conducteur pondéré par l'ancienneté du permis
    if 'age_conducteur1' in df.columns and 'anciennete_permis1' in df.columns:
        df['age_obtention_permis'] = df['age_conducteur1'] - df['anciennete_permis1']
        
    # 4. Indicateur de risque "Jeune Conducteur / Véhicule Puissant"
    if 'age_conducteur1' in df.columns and 'din_vehicule' in df.columns:
        df['risque_jeune_sportif'] = (1 / (df['age_conducteur1'] + 1)) * df['din_vehicule']

    # 5. Transformation logarithmique de la valeur du véhicule pour lisser la distribution
    if 'prix_vehicule' in df.columns:
        df['log_prix_vehicule'] = np.log1p(df['prix_vehicule'])


    # =========================================================
    # PARTIE B : NETTOYAGE ET ENCODAGE DES VARIABLES CATÉGORIELLES
    # =========================================================

    # Traitement des valeurs manquantes pour le second conducteur
    if 'sex_conducteur2' in df.columns:
        df['sex_conducteur2'] = df['sex_conducteur2'].fillna('NoDriver')

    # Encodage ordinal manuel pour les variables à hiérarchie intrinsèque ou binaire
    if 'sex_conducteur1' in df.columns:
        df['sex_conducteur1'] = df['sex_conducteur1'].replace(['M', 'F'], [0, 1]).infer_objects(copy=False)
    if 'type_vehicule' in df.columns:
        df['type_vehicule'] = df['type_vehicule'].replace(['Tourism', 'Commercial'], [0, 1]).infer_objects(copy=False)
    if 'utilisation' in df.columns:
        df['utilisation'] = df['utilisation'].replace(['Retired', 'WorkPrivate', 'Professional', 'AllTrips'], [0, 1, 2, 4]).infer_objects(copy=False)
    if 'freq_paiement' in df.columns:
        df['freq_paiement'] = df['freq_paiement'].replace(['Monthly', 'Quarterly', 'Biannual', 'Yearly'], [0, 1, 2, 3]).infer_objects(copy=False)

    # Encodage disjonctif complet (One-Hot Encoding) pour les variables nominales
    cols_to_encode = [
        'marque_vehicule', 'modele_vehicule', 'sex_conducteur2', 
        'type_contrat', 'paiement', 'conducteur2', 'essence_vehicule'
    ]
    exist_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=exist_cols, dtype=int)
    
    print(f"Pipeline de prétraitement terminé. Dimensions finales : {df.shape}")
    return df


def prepare_for_severity(df):
    """
    Prépare le sous-ensemble de données spécifiquement pour la modélisation 
    de la composante Sévérité (montant des sinistres).
    
    Filtre les observations pour ne conserver que les individus ayant déclaré
    au moins un sinistre et applique une transformation logarithmique sur la cible.
    """
    print("Préparation de la matrice d'apprentissage - Modèle Sévérité")
    
    # Isolation de la population sinistrée
    df_sev = df[df['nombre_sinistres'] > 0].copy()
    
    # Transformation de la variable cible (log1p)
    y = np.log1p(df_sev['montant_sinistre']).values
    
    # Suppression des variables cibles et des identifiants non prédictifs
    cols_to_drop = [
        'montant_sinistre', 'nombre_sinistres', 
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    X_df = df_sev.drop(columns=cols_to_drop, errors='ignore')
    
    return X_df.values, y, X_df.columns


def prepare_for_frequency(df):
    """
    Prépare le jeu de données intégral pour la modélisation de la composante
    Fréquence (probabilité d'occurrence d'un sinistre).
    
    Transforme la variable de comptage des sinistres en une variable cible binaire.
    """
    print("Préparation de la matrice d'apprentissage - Modèle Fréquence")
    
    # Création de la cible binaire : 1 si sinistre(s), 0 sinon
    y = (df['nombre_sinistres'] > 0).astype(int).values
    
    # Exclusion des informations relatives au post-sinistre et des identifiants
    cols_to_drop = [
        'montant_sinistre', 'nombre_sinistres', 
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    
    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    feature_names = X_df.columns
    
    return X_df.values, y, feature_names

def prepare_for_inference(df, model_features):
    """
    Prépare les données de test (ou de production) pour la phase d'inférence.
    
    S'assure que la matrice de test possède exactement la même dimensionnalité 
    et le même ordre de colonnes que la matrice d'entraînement. Gère également
    l'extraction de l'identifiant unique.
    """
    print("Alignement de la matrice de données pour la phase d'inférence")
    
    # Extraction dynamique de la clé primaire (Identifiant)
    if 'id_contrat' in df.columns:
        ids = df['id_contrat'].values
    elif 'id_client' in df.columns:
        ids = df['id_client'].values
    elif 'index' in df.columns:
        ids = df['index'].values
    elif 'id' in df.columns:
        ids = df['id'].values
    else:
        # Solution de repli : sélection de la première colonne par défaut
        ids = df.iloc[:, 0].values

    # Alignement strict des colonnes sur le format attendu par les modèles.
    # Les variables absentes du jeu de test seront créées et initialisées à 0.
    df_aligned = df.reindex(columns=model_features, fill_value=0)
    
    return df_aligned.values, ids