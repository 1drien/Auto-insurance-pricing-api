import os
import pandas as pd
import numpy as np
import pickle  # Importation pour la sérialisation demandée 
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from src import preprocessing, severity, frequency, evaluation, visualization
from src.prime_cv import oof_prime_rmse

def main():
    # 1. CONFIGURATION
    # Les chemins restent relatifs à l'emplacement du fichier main.py
    DATA_PATH = os.path.join("data", "train.csv")
    TEST_PATH = os.path.join("data", "test.csv") 
    PRESENTATION_DIR = "plots-presentation"
    os.makedirs(PRESENTATION_DIR, exist_ok=True)
    
    OUTPUT_DIR = "pricing_model"
    # Le dossier 'models' sera créé au même niveau que main.py
    MODELS_DIR = "models" 
    
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_final.csv") 
    FREQ_PATH = os.path.join(OUTPUT_DIR, "prediction_frequence.csv")
    SEV_PATH = os.path.join(OUTPUT_DIR, "prediction_severite.csv")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True) # Création du dossier au même niveau que main.py
    
    print("=== DÉMARRAGE DU PIPELINE DE TARIFICATION ACTUARIELLE ===")

    # 2. CHARGEMENT
    df_train_full = preprocessing.load_and_clean_common_data(DATA_PATH)
    visualization.plot_correlation_matrix(df_train_full, PRESENTATION_DIR)

    # 3. PHASE 1A : SÉVÉRITÉ
    print("\n--- PHASE 1A : MODÉLISATION DE LA SÉVÉRITÉ ---")
    X_sev, y_sev, feats_sev = preprocessing.prepare_for_severity(df_train_full)
    mae_sev, rmse_sev = severity.run_kfold_validation(X_sev, y_sev)
    print(f"[MÉTRIQUE] RMSE SÉVÉRITÉ (K-Fold) : {rmse_sev:.2f} €")
    model_sev = severity.train_final_model(X_sev, y_sev)

    # 4. PHASE 1B : FRÉQUENCE (AUC TRAIN vs TEST)
    print("\n--- PHASE 1B : MODÉLISATION DE LA FRÉQUENCE ---")
    X_f, y_f, feats_f = preprocessing.prepare_for_frequency(df_train_full)
    
    # Séparation 80% Entraînement / 20% Test interne
    X_train, X_test, y_train, y_test = train_test_split(
        X_f, y_f, test_size=0.20, random_state=42, stratify=y_f
    )
    
    print(f"Évaluation sur split interne : Train={len(X_train)} / Test={len(X_test)}")
    
    # Entraînement
    model_freq = frequency.train_final_model(X_train, y_train)
    
    # Calcul des probabilités
    probs_train = model_freq.predict_proba(X_train)[:, 1]
    probs_test = model_freq.predict_proba(X_test)[:, 1]
    
    # Calcul des AUC
    auc_train = roc_auc_score(y_train, probs_train)
    auc_test = roc_auc_score(y_test, probs_test)
    
    print(f"\n[SYNTHÈSE FRÉQUENCE]")
    print(f"-> AUC TRAIN : {auc_train:.4f}")
    print(f"-> AUC TEST  : {auc_test:.4f}")

    # Visualisations
    print(f"Génération de la courbe ROC (Données Test) dans '{PRESENTATION_DIR}'...")
    visualization.plot_roc_curve(y_test, probs_test, PRESENTATION_DIR)
    visualization.plot_feature_importance(model_freq, feats_f, "Fréquence", PRESENTATION_DIR, "importance_frequence")

    # 5. PHASE 2 : ÉVALUATION GLOBALE (OOF)
    print("\n--- PHASE 2 : ÉVALUATION GLOBALE (PRIME PURE OOF) ---")
    oof_predictions, rmse_prime_global = oof_prime_rmse(
        df_train=df_train_full,
        preprocess_for_freq_fn=preprocessing.prepare_for_frequency,
        preprocess_for_sev_fn=preprocessing.prepare_for_severity,
        train_freq_fn=frequency.train_final_model,
        train_sev_fn=severity.train_final_model,
        n_splits=5,
        random_state=42
    )
    print(f"=== ÉVALUATION GLOBALE (RMSE OOF) : {rmse_prime_global:.2f} ===")

    # 6. PHASE 3 : INFÉRENCE (SOUMISSION)
    print("\n--- PHASE 3 : INFÉRENCE SUR LE JEU DE TEST RÉEL ---")
    df_submission_data = preprocessing.load_and_clean_common_data(TEST_PATH)
    X_sub, ids_sub = preprocessing.prepare_for_inference(df_submission_data, feats_f)
    
    # Prédictions
    p_freq = model_freq.predict_proba(X_sub)[:, 1]
    p_sev = np.expm1(model_sev.predict(X_sub))

    # Sauvegarde des composantes individuelles (fichiers debug demandés)
    pd.DataFrame({'index': ids_sub, 'probabilite_sinistre': p_freq}).to_csv(FREQ_PATH, index=False)
    pd.DataFrame({'index': ids_sub, 'cout_moyen_estime': p_sev}).to_csv(SEV_PATH, index=False)
    
    # Prime finale
    prime_finale = (p_freq * p_sev) * 1.18
    df_submission = pd.DataFrame({'index': ids_sub, 'pred': np.maximum(prime_finale, 0)})
    df_submission.to_csv(SUBMISSION_PATH, index=False)

    
    # ÉTAPE 1 INDUSTRIALISATION : SÉRIALISATION (PICKLE) 
    
    print("\n--- ÉTAPE 1 INDUSTRIALISATION : EXPORT DES MODÈLES ---")
    
    # Sauvegarde du modèle de sévérité
    with open(os.path.join(MODELS_DIR, "model_severity.pkl"), "wb") as f:
        pickle.dump(model_sev, f)
        
    # Sauvegarde du modèle de fréquence
    with open(os.path.join(MODELS_DIR, "model_frequency.pkl"), "wb") as f:
        pickle.dump(model_freq, f)
        
    # Sauvegarde des noms de variables pour l'API future
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(list(feats_f), f)

    print(f"[INFO] Les modèles et features ont été sérialisés dans : {os.path.abspath(MODELS_DIR)}")
    
    print(f"\n[TERMINÉ] -> Fichier final : {SUBMISSION_PATH}")
    print(f"-> Détails exportés : {FREQ_PATH} et {SEV_PATH}")
    print(f"-> Prime moyenne : {prime_finale.mean():.2f} €")

if __name__ == "__main__":
    main()