# MODULE DE VALIDATION CROISÉE POUR L'ESTIMATION DE LA PRIME PURE

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    """
    Calcule la Racine de l'Erreur Quadratique Moyenne (RMSE).
    
    Paramètres :
    - y_true : Vecteur des valeurs réelles observées.
    - y_pred : Vecteur des valeurs prédites par le modèle.
    
    Retourne :
    - L'erreur RMSE sous forme de flottant.
    """
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
    Implémente une procédure de validation croisée Out-Of-Fold (OOF) pour 
    évaluer l'architecture de tarification (Modèle Fréquence-Sévérité).
    
    Cette fonction estime la prime pure (Probabilité x Coût) et calcule 
    la performance prédictive globale du modèle selon la métrique RMSE.
    """

    # Définition de la variable de stratification (présence ou absence de sinistre)
    # Cela garantit une répartition équitable des sinistres dans chaque fold.
    y_freq_full = (df_train["nombre_sinistres"] > 0).astype(int).values

    # Initialisation de la validation croisée stratifiée
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialisation du vecteur qui stockera les prédictions Out-Of-Fold
    oof_prime = np.zeros(len(df_train), dtype=float)

    # Construction du vecteur des coûts réels observés (cible globale)
    # Règle : 0 si aucun sinistre, sinon le montant exact du sinistre.
    y_cost_full = np.where(
        df_train["nombre_sinistres"].values > 0,
        df_train["montant_sinistre"].values,
        0.0
    ).astype(float)

    # Itération sur les plis d'apprentissage et de validation
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df_train, y_freq_full), start=1):
        df_tr = df_train.iloc[tr_idx].copy()
        df_va = df_train.iloc[va_idx].copy()

        # COMPOSANTE 1 : MODÉLISATION DE LA FRÉQUENCE
        
        # Préparation des données d'apprentissage pour la fréquence
        X_tr_f, y_tr_f, feats_f = preprocess_for_freq_fn(df_tr)
        
        # Préparation des données de validation (exclusion des variables cibles)
        X_va_f = df_va.drop(columns=[
            'montant_sinistre', 'nombre_sinistres',
            'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
        ], errors='ignore')
        
        # Alignement strict des colonnes de validation sur celles d'apprentissage
        X_va_f = X_va_f.reindex(columns=feats_f, fill_value=0).values

        # Entraînement et prédiction des probabilités de sinistre
        model_f = train_freq_fn(X_tr_f, y_tr_f)
        p_va = model_f.predict_proba(X_va_f)[:, 1]

        # COMPOSANTE 2 : MODÉLISATION DE LA SÉVÉRITÉ
        
        # L'entraînement s'effectue exclusivement sur la population sinistrée
        X_tr_s, y_tr_s, feats_s = preprocess_for_sev_fn(df_tr)

        # Mécanisme de sécurité : gestion de la faible volumétrie
        # Si le pli contient moins de 20 sinistres, l'entraînement d'un modèle 
        # complexe est instable. On a recours à un estimateur naïf (moyenne empirique).
        if len(y_tr_s) < 20:
            mean_cost = float(df_tr.loc[df_tr["nombre_sinistres"] > 0, "montant_sinistre"].mean())
            cost_va = np.full(len(df_va), mean_cost, dtype=float)
        else:
            # Entraînement du modèle de régression
            model_s = train_sev_fn(X_tr_s, y_tr_s)

            # Alignement des variables de validation pour la sévérité
            X_va_s = df_va.drop(columns=[
                'montant_sinistre', 'nombre_sinistres',
                'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
            ], errors='ignore')
            X_va_s = X_va_s.reindex(columns=feats_s, fill_value=0).values

            # Inférence et transformation inverse (logarithme vers espace monétaire)
            log_cost_va = model_s.predict(X_va_s)
            cost_va = np.expm1(log_cost_va)

        # CALCUL DE LA PRIME PURE ET ÉVALUATION      
        # Plafonnement optionnel pour limiter l'impact des valeurs extrêmes (outliers)
        if clip_sev_max is not None:
            cost_va = np.clip(cost_va, 0, clip_sev_max)

        # Formulation de la prime pure technique : Espérance de fréquence x Espérance de sévérité
        prime_va = p_va * cost_va
        oof_prime[va_idx] = prime_va

        # Évaluation locale du pli courant
        fold_rmse = rmse(y_cost_full[va_idx], oof_prime[va_idx])
        print(f"[Pli {fold}/{n_splits}] RMSE Prime Pure : {fold_rmse:.4f}")

    # Évaluation globale sur l'intégralité de l'échantillon Out-Of-Fold
    total_rmse = rmse(y_cost_full, oof_prime)
    print(f"\n=== ÉVALUATION GLOBALE (RMSE OOF Prime) : {total_rmse:.4f} ===")
    
    return oof_prime, total_rmse