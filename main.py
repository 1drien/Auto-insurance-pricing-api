import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from src import preprocessing, severity, frequency, evaluation, visualization
from src.prime_cv import oof_prime_rmse


def main() -> None:
    # 1. CONFIGURATION
    DATA_PATH: str = os.path.join("data", "train.csv")
    TEST_PATH: str = os.path.join("data", "test.csv")
    PRESENTATION_DIR: str = "plots-presentation"
    os.makedirs(PRESENTATION_DIR, exist_ok=True)

    OUTPUT_DIR: str = "pricing_model"
    MODELS_DIR: str = "models"

    SUBMISSION_PATH: str = os.path.join(OUTPUT_DIR, "submission_final.csv")
    FREQ_PATH: str = os.path.join(OUTPUT_DIR, "prediction_frequence.csv")
    SEV_PATH: str = os.path.join(OUTPUT_DIR, "prediction_severite.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== STARTING ACTUARIAL PRICING PIPELINE ===")

    # 2. DATA LOADING
    df_train_full: pd.DataFrame = preprocessing.load_and_clean_common_data(DATA_PATH)
    visualization.plot_correlation_matrix(df_train_full, PRESENTATION_DIR)

    # 3. PHASE 1A: SEVERITY MODEL
    print("\n--- PHASE 1A: SEVERITY MODELING ---")
    X_sev, y_sev, feats_sev = preprocessing.prepare_for_severity(df_train_full)
    mae_sev, rmse_sev = severity.run_kfold_validation(X_sev, y_sev)
    print(f"[METRIC] SEVERITY RMSE (K-Fold): {rmse_sev:.2f} EUR")
    model_sev = severity.train_final_model(X_sev, y_sev)

    # 4. PHASE 1B: FREQUENCY MODEL (AUC TRAIN vs TEST)
    print("\n--- PHASE 1B: FREQUENCY MODELING ---")
    X_f, y_f, feats_f = preprocessing.prepare_for_frequency(df_train_full)

    # 80% Train / 20% Internal Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_f, y_f, test_size=0.20, random_state=42, stratify=y_f
    )

    print(f"Internal split evaluation: Train={len(X_train)} / Test={len(X_test)}")

    # Training
    model_freq = frequency.train_final_model(X_train, y_train)

    # Probability predictions
    probs_train: np.ndarray = model_freq.predict_proba(X_train)[:, 1]
    probs_test: np.ndarray = model_freq.predict_proba(X_test)[:, 1]

    # AUC scores
    auc_train: float = roc_auc_score(y_train, probs_train)
    auc_test: float = roc_auc_score(y_test, probs_test)

    print(f"\n[FREQUENCY SUMMARY]")
    print(f"-> AUC TRAIN: {auc_train:.4f}")
    print(f"-> AUC TEST:  {auc_test:.4f}")

    # Visualizations
    print(f"Generating ROC curve (Test Data) in '{PRESENTATION_DIR}'...")
    visualization.plot_roc_curve(y_test, probs_test, PRESENTATION_DIR)
    visualization.plot_feature_importance(model_freq, feats_f, "Frequency", PRESENTATION_DIR, "importance_frequency")

    # 5. PHASE 2: GLOBAL EVALUATION (OOF)
    print("\n--- PHASE 2: GLOBAL EVALUATION (PURE PREMIUM OOF) ---")
    oof_predictions, rmse_prime_global = oof_prime_rmse(
        df_train=df_train_full,
        preprocess_for_freq_fn=preprocessing.prepare_for_frequency,
        preprocess_for_sev_fn=preprocessing.prepare_for_severity,
        train_freq_fn=frequency.train_final_model,
        train_sev_fn=severity.train_final_model,
        n_splits=5,
        random_state=42
    )
    print(f"=== GLOBAL EVALUATION (OOF RMSE): {rmse_prime_global:.2f} ===")

    # 6. PHASE 3: INFERENCE (SUBMISSION)
    print("\n--- PHASE 3: INFERENCE ON REAL TEST SET ---")
    df_submission_data: pd.DataFrame = preprocessing.load_and_clean_common_data(TEST_PATH)
    X_sub, ids_sub = preprocessing.prepare_for_inference(df_submission_data, feats_f)

    # Predictions
    p_freq: np.ndarray = model_freq.predict_proba(X_sub)[:, 1]
    p_sev: np.ndarray = np.expm1(model_sev.predict(X_sub))

    # Save individual components (debug files)
    pd.DataFrame({'index': ids_sub, 'claim_probability': p_freq}).to_csv(FREQ_PATH, index=False)
    pd.DataFrame({'index': ids_sub, 'estimated_avg_cost': p_sev}).to_csv(SEV_PATH, index=False)

    # Final premium
    final_premium: np.ndarray = (p_freq * p_sev) * 1.18
    df_submission: pd.DataFrame = pd.DataFrame({'index': ids_sub, 'pred': np.maximum(final_premium, 0)})
    df_submission.to_csv(SUBMISSION_PATH, index=False)

    # INDUSTRIALIZATION STEP 1: SERIALIZATION (PICKLE)

    print("\n--- INDUSTRIALIZATION STEP 1: MODEL EXPORT ---")

    # Save severity model
    with open(os.path.join(MODELS_DIR, "model_severity.pkl"), "wb") as f:
        pickle.dump(model_sev, f)

    # Save frequency model
    with open(os.path.join(MODELS_DIR, "model_frequency.pkl"), "wb") as f:
        pickle.dump(model_freq, f)

    # Save feature names for API alignment
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(list(feats_f), f)

    print(f"[INFO] Models and features serialized in: {os.path.abspath(MODELS_DIR)}")

    print(f"\n[DONE] -> Final file: {SUBMISSION_PATH}")
    print(f"-> Details exported: {FREQ_PATH} and {SEV_PATH}")
    print(f"-> Average premium: {final_premium.mean():.2f} EUR")


if __name__ == "__main__":
    main()