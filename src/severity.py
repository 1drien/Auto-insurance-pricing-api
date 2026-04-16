import numpy as np
from numpy.typing import NDArray
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

PARAMS: dict = {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'verbosity': 0,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'max_depth': 3,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}


def run_kfold_validation(X: NDArray, y: NDArray, n_splits: int = 5) -> tuple[float, float]:
    """
    Évalue les performances du modèle de sévérité via validation croisée K-Fold.

    Args:
        X: Matrice des variables explicatives.
        y: Vecteur cible (montants en logarithme).
        n_splits: Nombre de plis pour la validation croisée.

    Returns:
        Tuple (MAE moyenne, RMSE moyenne) en euros.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)
    mae_scores: list[float] = []
    rmse_scores: list[float] = []

    print(f"Début de la validation croisée à {n_splits} plis (Composante Sévérité)...")

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X_train, y_train)

        pred_log = model.predict(X_val)
        pred_euros = np.expm1(pred_log)
        y_true_euros = np.expm1(y_val)

        mae_scores.append(mean_absolute_error(y_true_euros, pred_euros))
        rmse_scores.append(np.sqrt(mean_squared_error(y_true_euros, pred_euros)))

    return float(np.mean(mae_scores)), float(np.mean(rmse_scores))


def train_final_model(X: NDArray, y: NDArray) -> xgb.XGBRegressor:
    """
    Entraîne le modèle de sévérité définitif sur l'intégralité du jeu de données.

    Args:
        X: Matrice complète des variables explicatives.
        y: Vecteur complet de la variable cible.

    Returns:
        Estimateur XGBoost entraîné.
    """
    print("Entraînement du modèle final de sévérité (Mise en production)...")
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y)
    return model