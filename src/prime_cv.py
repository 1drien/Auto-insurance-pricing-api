# src/prime_cv.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def oof_prime_rmse(
    df_train,
    preprocess_for_freq_fn,
    preprocess_for_sev_fn,
    train_freq_fn,
    train_sev_fn,
    n_splits=5,
    random_state=42,
    clip_sev_max=None
):
    """
    Calcule des prédictions OOF pour:
      - proba fréquence
      - coût gravité
      - prime = proba * coût
    et renvoie la RMSE OOF sur le coût réel (0 si pas de sinistre, montant sinon).
    """

    # Cible stratification: sinistre oui/non
    y_freq_full = (df_train["nombre_sinistres"] > 0).astype(int).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_prime = np.zeros(len(df_train), dtype=float)

    # coût réel utilisé par Kaggle: 0 si pas sinistre sinon montant
    y_cost_full = np.where(
        df_train["nombre_sinistres"].values > 0,
        df_train["montant_sinistre"].values,
        0.0
    ).astype(float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_train, y_freq_full), start=1):
        df_tr = df_train.iloc[tr_idx].copy()
        df_va = df_train.iloc[va_idx].copy()

        # --- fréquence ---
        X_tr_f, y_tr_f, feats_f = preprocess_for_freq_fn(df_tr)
        X_va_f = df_va.drop(columns=[
            'montant_sinistre', 'nombre_sinistres',
            'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
        ], errors='ignore')
        # align colonnes
        X_va_f = X_va_f.reindex(columns=feats_f, fill_value=0).values

        model_f = train_freq_fn(X_tr_f, y_tr_f)
        p_va = model_f.predict_proba(X_va_f)[:, 1]

        # --- gravité ---
        # entraînement uniquement sur sinistrés du train fold
        X_tr_s, y_tr_s, feats_s = preprocess_for_sev_fn(df_tr)

        # Si dans un fold il y a très peu de sinistrés, on sécurise
        if len(y_tr_s) < 20:
            # fallback: coût moyen des sinistrés du train fold (en euros)
            mean_cost = float(df_tr.loc[df_tr["nombre_sinistres"] > 0, "montant_sinistre"].mean())
            cost_va = np.full(len(df_va), mean_cost, dtype=float)
        else:
            model_s = train_sev_fn(X_tr_s, y_tr_s)

            # features val alignées sur feats_s
            X_va_s = df_va.drop(columns=[
                'montant_sinistre', 'nombre_sinistres',
                'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
            ], errors='ignore')
            X_va_s = X_va_s.reindex(columns=feats_s, fill_value=0).values

            log_cost_va = model_s.predict(X_va_s)
            cost_va = np.expm1(log_cost_va)

        # option anti-RMSE-explosion
        if clip_sev_max is not None:
            cost_va = np.clip(cost_va, 0, clip_sev_max)

        prime_va = p_va * cost_va
        oof_prime[va_idx] = prime_va

        fold_rmse = rmse(y_cost_full[va_idx], oof_prime[va_idx])
        print(f"[Fold {fold}/{n_splits}] RMSE prime: {fold_rmse:.4f}")

    total_rmse = rmse(y_cost_full, oof_prime)
    print(f"\n=== OOF PRIME RMSE (global) : {total_rmse:.4f} ===")
    return oof_prime, total_rmse