import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


# CONFIGURATION DES HYPERPARAMÈTRES

# Paramètres du modèle XGBoost optimisés pour la régression.
# La configuration inclut des mécanismes de régularisation 
# afin de limiter le surapprentissage (overfitting).
PARAMS = {
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

def run_kfold_validation(X, y, n_splits=5):
    """
    Évalue les performances du modèle de sévérité via une validation croisée (K-Fold).
    
    Paramètres :
    - X : Matrice des variables explicatives (features).
    - y : Vecteur de la variable cible (montants des sinistres en logarithme).
    - n_splits : Nombre de plis pour la validation croisée.
    
    Retourne :
    - La moyenne de l'Erreur Absolue Moyenne (MAE) et de la Racine de l'Erreur 
      Quadratique Moyenne (RMSE) évaluées sur l'échelle réelle (en euros).
    """
    # Initialisation de la validation croisée aléatoire
    kf = KFold(n_splits=5, shuffle=True, random_state=None)    
    mae_scores = []
    rmse_scores = []

    print(f"Début de la validation croisée à {n_splits} plis (Composante Sévérité)...")

    for train_index, val_index in kf.split(X):
        # 1. Séparation des ensembles d'apprentissage et de validation
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 2. Instanciation et entraînement du modèle sur le pli courant
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X_train, y_train)
        
        # 3. Prédiction sur l'ensemble de validation (les prédictions sont en log)
        pred_log = model.predict(X_val)

        # 4. Transformation inverse (Passage de l'échelle logarithmique à l'échelle monétaire)
        # L'utilisation de np.expm1 garantit l'inversion exacte de np.log1p
        pred_euros = np.expm1(pred_log)
        y_true_euros = np.expm1(y_val)

        # 5. Calcul et stockage des métriques d'évaluation
        mae_scores.append(mean_absolute_error(y_true_euros, pred_euros))
        rmse_scores.append(np.sqrt(mean_squared_error(y_true_euros, pred_euros)))

    return np.mean(mae_scores), np.mean(rmse_scores)


def train_final_model(X, y):
    """
    Entraîne le modèle de sévérité définitif sur l'intégralité du jeu de données.
    
    Paramètres :
    - X : Matrice complète des variables explicatives.
    - y : Vecteur complet de la variable cible.
    
    Retourne :
    - model : L'estimateur XGBoost entraîné, prêt pour l'inférence.
    """
    print("Entraînement du modèle final de sévérité (Mise en production)...")
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y)
    
    return model