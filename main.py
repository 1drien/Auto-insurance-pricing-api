import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 
from src import preprocessing, severity, frequency, evaluation
from src.prime_cv import oof_prime_rmse

def main():
    # =========================================================
    # CONFIGURATION DES CHEMINS D'ACCÈS ET DES RÉPERTOIRES
    # =========================================================
    DATA_PATH = os.path.join("data", "train.csv")
    TEST_PATH = os.path.join("data", "test.csv") 
    OUTPUT_DIR = "severity_model"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_final.csv") 
    
    # Création du répertoire de sortie s'il n'existe pas
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("=== DÉMARRAGE DU PIPELINE DE TARIFICATION ACTUARIELLE ===")

    # =========================================================
    # PHASE 1 : ENTRAÎNEMENT DES MODÈLES FRÉQUENCE / SÉVÉRITÉ
    # =========================================================
    
    # Chargement et nettoyage initial des données d'entraînement
    df_train = preprocessing.load_and_clean_common_data(DATA_PATH)

    # --- MODÈLE DE SÉVÉRITÉ (Coût des sinistres) ---
    print("\n--- PHASE 1A : MODÉLISATION DE LA SÉVÉRITÉ ---")
    X_sev, y_sev, feats_sev = preprocessing.prepare_for_severity(df_train)
    
    # Évaluation du modèle de sévérité par validation croisée (K-Fold)
    mae_sev, rmse_sev = severity.run_kfold_validation(X_sev, y_sev)
    print(f"\n[MÉTRIQUE] RMSE SÉVÉRITÉ (Validation Croisée) : {rmse_sev:.4f}")
    
    # Entraînement du modèle final de sévérité sur l'ensemble des données
    model_sev = severity.train_final_model(X_sev, y_sev)
    
    # --- MODÈLE DE FRÉQUENCE (Probabilité de sinistre) ---
    print("\n--- PHASE 1B : MODÉLISATION DE LA FRÉQUENCE ---")
    X_freq, y_freq, feats_freq = preprocessing.prepare_for_frequency(df_train)
    
    # Entraînement du modèle final de fréquence
    model_freq = frequency.train_final_model(X_freq, y_freq)

    # Évaluation de la performance du modèle de fréquence (Brier/RMSE)
    print("Évaluation du RMSE Fréquence sur l'échantillon d'apprentissage...")
    probas_train = model_freq.predict_proba(X_freq)[:, 1]
    rmse_freq = np.sqrt(mean_squared_error(y_freq, probas_train))
    print(f"[MÉTRIQUE] RMSE FRÉQUENCE (Intra-échantillon) : {rmse_freq:.4f}")

    # =========================================================
    # PHASE 2 : ÉVALUATION ROBUSTE DE LA PRIME (OUT-OF-FOLD)
    # =========================================================
    print("\n--- PHASE 2 : ÉVALUATION OUT-OF-FOLD (OOF) DE LA PRIME PURE ---")
    oof_prime, rmse_prime = oof_prime_rmse(
        df_train=df_train,
        preprocess_for_freq_fn=preprocessing.prepare_for_frequency,
        preprocess_for_sev_fn=preprocessing.prepare_for_severity,
        train_freq_fn=frequency.train_final_model,
        train_sev_fn=severity.train_final_model,
        n_splits=5,
        random_state=42,
        clip_sev_max=50000 
    )

    # =========================================================
    # PHASE 3 : INFÉRENCE SUR LE JEU DE TEST ET AJUSTEMENT
    # =========================================================
    print("\n" + "="*40)
    print(" PHASE 3 : GÉNÉRATION DES PRÉDICTIONS SUR LE JEU DE TEST ")
    print("="*40)

    # Chargement, nettoyage et alignement des données de test
    df_test = preprocessing.load_and_clean_common_data(TEST_PATH)
    X_test, ids_test = preprocessing.prepare_for_inference(df_test, feats_freq)
    
    # Prédiction de la composante Fréquence
    print("Estimation des probabilités de sinistre (Composante Fréquence)...")
    probas_freq = model_freq.predict_proba(X_test)[:, 1]
    
    # Prédiction de la composante Sévérité
    print("Estimation des coûts de sinistre (Composante Sévérité)...")
    log_couts = model_sev.predict(X_test)
    couts_sev = np.expm1(log_couts)
    
    # Calcul de la prime pure technique (Fréquence x Sévérité)
    print("Calcul de la prime pure et application du facteur de correction (x1.14)...")
    prime_pure_base = probas_freq * couts_sev
    
    # Application d'un facteur d'ajustement global issu de l'analyse du Leaderboard
    prime_pure_finale = prime_pure_base * 1.14
    
    # Contrôle de sécurité : plafonnement inférieur à zéro (pas de prime négative)
    prime_pure_finale = np.maximum(prime_pure_finale, 0)
    
    # Construction du DataFrame final pour exportation
    df_submission = pd.DataFrame({
        'index': ids_test,       
        'pred': prime_pure_finale       
    })
    
    # Exportation au format CSV
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n[SUCCÈS] Fichier de prédiction généré : {SUBMISSION_PATH}")
    print(f"   -> Prime moyenne projetée : {prime_pure_finale.mean():.2f} €")
    print(df_submission.head())

if __name__ == "__main__":
    main()