import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

# =========================================================
# CLASSE D'ENSEMBLE POUR LA MODÉLISATION DE LA FRÉQUENCE
# =========================================================

class FrequencyEnsemble(BaseEstimator, ClassifierMixin):
    """
    Classe personnalisée (Wrapper) implémentant une approche d'apprentissage 
    ensembliste (Ensemble Learning) pour la prédiction de la fréquence des sinistres.
    
    Hérite de BaseEstimator et ClassifierMixin pour garantir la compatibilité 
    avec l'écosystème Scikit-Learn. Le modèle combine les prédictions d'un algorithme 
    XGBoost et d'un HistGradientBoosting via une moyenne des probabilités (Soft Voting).
    """
    def __init__(self, ratio):
        self.ratio = ratio
        
        # 1. Estimateur de base 1 : XGBoost
        # Configuration optimisée pour traiter le fort déséquilibre de classes 
        # (scale_pos_weight) inhérent aux données d'assurance.
        xgb_base = XGBClassifier(
            n_estimators=354, 
            learning_rate=0.01, 
            max_depth=4, 
            scale_pos_weight=self.ratio, 
            random_state=42, 
            n_jobs=-1,
            verbosity=0
        )
        # Application d'une calibration isotonique en validation croisée (5 plis)
        # Objectif : Obtenir des probabilités réelles de sinistre et non de simples scores.
        self.xgb_calibrated = CalibratedClassifierCV(xgb_base, method='isotonic', cv=5)
        
        # 2. Estimateur de base 2 : HistGradientBoosting
        # Modèle robuste et rapide, utilisant une pondération interne équilibrée.
        hgb_base = HistGradientBoostingClassifier(
            learning_rate=0.05, 
            max_iter=200, 
            class_weight='balanced', 
            random_state=42
        )
        self.hgb_calibrated = CalibratedClassifierCV(hgb_base, method='isotonic', cv=5)

    def fit(self, X, y):
        """
        Ajuste les modèles de base calibrés sur les données d'apprentissage.
        """
        print("      -> Ajustement du modèle XGBoost (Calibration Isotonique, CV=5)...")
        self.xgb_calibrated.fit(X, y)
        
        print("      -> Ajustement du modèle HistGradientBoosting (Calibration Isotonique, CV=5)...")
        self.hgb_calibrated.fit(X, y)
        
        return self

    def predict_proba(self, X):
        """
        Estime la probabilité d'occurrence d'un sinistre pour chaque observation.
        """
        # Récupération des probabilités individuelles de chaque estimateur
        p_xgb = self.xgb_calibrated.predict_proba(X)
        p_hgb = self.hgb_calibrated.predict_proba(X)
        
        # Agrégation des prédictions (Moyenne arithmétique stricte)
        # Cette approche réduit la variance et stabilise les prédictions globales.
        return (p_xgb + p_hgb) / 2
        
    def predict(self, X):
        """
        Génère la classe prédite (0 ou 1) basée sur un seuil de probabilité standard.
        Méthode requise par l'interface Scikit-Learn.
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= 0.5).astype(int)


# =========================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT
# =========================================================

def train_final_model(X, y):
    """
    Fonction d'interface appelée par les modules main.py et prime_cv.py.
    Instancie et entraîne le modèle composite (Ensemble) pour la fréquence.
    
    Paramètres :
    - X : Matrice des variables explicatives.
    - y : Vecteur cible binaire (Présence/Absence de sinistre).
    
    Retourne :
    - model : L'estimateur composite entraîné.
    """
    print("Entraînement du modèle composite Fréquence (XGBoost + HistGradientBoosting)...")
    
    # Détermination du ratio de déséquilibre des classes (Non-sinistré / Sinistré)
    # Nécessaire pour pénaliser correctement l'erreur sur la classe minoritaire.
    ratio = (y == 0).sum() / (y == 1).sum()
    
    # Instanciation de la classe sur-mesure avec le ratio calculé
    model = FrequencyEnsemble(ratio=ratio)
    
    # Entraînement global de la structure d'ensemble
    model.fit(X, y)
    
    return model