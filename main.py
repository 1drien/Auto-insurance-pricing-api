import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 
from src import preprocessing, severity, frequency, evaluation, visualization
from src.prime_cv import oof_prime_rmse

def main():
    # =========================================================
    # 1. CONFIGURATION DES CHEMINS ET RÉPERTOIRES
    # =========================================================
    DATA_PATH = os.path.join("data", "train.csv")
    TEST_PATH = os.path.join("data", "test.csv") 
    
    # Dossier pour votre soutenance/présentation
    PRESENTATION_DIR = "plots-presentation"
    os.makedirs(PRESENTATION_DIR, exist_ok=True)
    
    # Dossier technique de sortie
    OUTPUT_DIR = "severity_model"
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_final.csv") 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=== DÉMARRAGE DU PIPELINE DE TARIFICATION ACTUARIELLE ===")

    # =========================================================
    # 2. CHARGEMENT ET ANALYSE DESCRIPTIVE
    # =========================================================
    print(f"\nChargement des données depuis : {DATA_PATH}")
    df_train = preprocessing.load_and_clean_common_data(DATA_PATH)

    print(f"Génération des graphiques descriptifs dans '{PRESENTATION_DIR}'...")
    visualization.plot_correlation_matrix(df_train, PRESENTATION_DIR)
    visualization.plot_actuarial_analysis(df_train, PRESENTATION_DIR)

    # =========================================================
    # 3. PHASE 1 : MODÉLISATION INDIVIDUELLE
    # =========================================================
    
    # --- MODÈLE DE SÉVÉRITÉ (Coût moyen d'un sinistre) ---
    print("\n--- PHASE 1A : MODÉLISATION DE LA SÉVÉRITÉ ---")
    X_sev, y_sev, feats_sev = preprocessing.prepare_for_severity(df_train)
    
    # Évaluation par validation croisée
    mae_sev, rmse_sev = severity.run_kfold_validation(X_sev, y_sev)
    print(f"[MÉTRIQUE] RMSE SÉVÉRITÉ (K-Fold) : {rmse_sev:.2f} €")
    
    # Entraînement final et importance des variables
    model_sev = severity.train_final_model(X_sev, y_sev)
    visualization.plot_feature_importance(model_sev, feats_sev, "Sévérité (Coût)", PRESENTATION_DIR, "importance_severite")

    # --- MODÈLE DE FRÉQUENCE (Probabilité d'accident) ---
    print("\n--- PHASE 1B : MODÉLISATION DE LA FRÉQUENCE ---")
    X_freq, y_freq, feats_freq = preprocessing.prepare_for_frequency(df_train)
    
    # Entraînement final
    model_freq = frequency.train_final_model(X_freq, y_freq)
    
    # Évaluation rapide
    probas_train = model_freq.predict_proba(X_freq)[:, 1]
    rmse_freq = np.sqrt(mean_squared_error(y_freq, probas_train))
    print(f"[MÉTRIQUE] RMSE FRÉQUENCE (Intra) : {rmse_freq:.4f}")
    
    # Importance des variables fréquence
    visualization.plot_feature_importance(model_freq, feats_freq, "Fréquence (Probabilité)", PRESENTATION_DIR, "importance_frequence")

    # =========================================================
    # 4. PHASE 2 : ÉVALUATION CROISÉE DE LA PRIME (OOF)
    # =========================================================
    print("\n--- PHASE 2 : ÉVALUATION GLOBALE (PRIME PURE) ---")
    # Cette étape est cruciale pour le rapport de PFE (mesure l'erreur réelle du système complet)
    oof_predictions, rmse_prime_global = oof_prime_rmse(
        df_train=df_train,
        preprocess_for_freq_fn=preprocessing.prepare_for_frequency,
        preprocess_for_sev_fn=preprocessing.prepare_for_severity,
        train_freq_fn=frequency.train_final_model,
        train_sev_fn=severity.train_final_model,
        n_splits=5,
        random_state=42
    )
    
    # Visualisation de la calibration (Réel vs Prédit) sur les données d'entraînement (OOF)
    y_reel_total = df_train['montant_sinistre'].fillna(0)
    visualization.plot_model_performance(y_reel_total, oof_predictions, "Modèle Global (Fréq x Sév)", PRESENTATION_DIR, "calibration_finale")

    # =========================================================
    # 5. PHASE 3 : GÉNÉRATION DE LA SOUMISSION (TEST)
    # =========================================================
    print("\n--- PHASE 3 : INFÉRENCE SUR LE JEU DE TEST ---")
    df_test = preprocessing.load_and_clean_common_data(TEST_PATH)
    
    # On utilise ici le preprocessing d'inférence qui aligne les colonnes
    X_test, ids_test = preprocessing.prepare_for_inference(df_test, feats_freq)
    
    # Calcul des composantes
    p_freq = model_freq.predict_proba(X_test)[:, 1]
    # Rappel : le modèle de sévérité a été entraîné sur le log, on applique expm1
    p_sev = np.expm1(model_sev.predict(X_test))
    
    # Calcul de la Prime Pure avec multiplicateur d'ajustement (ex: 1.18 ou 1.24)
    factor = 1.18 
    prime_finale = (p_freq * p_sev) * factor
    prime_finale = np.maximum(prime_finale, 0) # Sécurité technique

    # Exportation
    df_submission = pd.DataFrame({'index': ids_test, 'pred': prime_finale})
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\n[TERMINÉ]")
    print(f"-> Fichier généré : {SUBMISSION_PATH}")
    print(f"-> Graphiques disponibles dans : {PRESENTATION_DIR}")
    print(f"-> Prime moyenne test : {prime_finale.mean():.2f} €")

if __name__ == "__main__":
    main()