import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier


class FrequencyEnsemble(BaseEstimator, ClassifierMixin):
    """
    Modèle d'ensemble combinant XGBoost et HistGradientBoosting
    avec calibration isotonique.
    """
    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

        xgb_base = XGBClassifier(
            n_estimators=354,
            learning_rate=0.01,
            max_depth=4,
            scale_pos_weight=self.ratio,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        self.xgb_calibrated = CalibratedClassifierCV(xgb_base, method='isotonic', cv=5)

        hgb_base = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            class_weight='balanced',
            random_state=42
        )
        self.hgb_calibrated = CalibratedClassifierCV(hgb_base, method='isotonic', cv=5)

    def fit(self, X: NDArray, y: NDArray) -> "FrequencyEnsemble":
        self.classes_ = np.unique(y)

        print("      -> Ajustement du modèle XGBoost (Calibration Isotonique)...")
        self.xgb_calibrated.fit(X, y)

        print("      -> Ajustement du modèle HistGradientBoosting (Calibration Isotonique)...")
        self.hgb_calibrated.fit(X, y)

        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        p_xgb = self.xgb_calibrated.predict_proba(X)
        p_hgb = self.hgb_calibrated.predict_proba(X)
        return (p_xgb + p_hgb) / 2

    def predict(self, X: NDArray) -> NDArray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= 0.5).astype(int)


def train_final_model(X: NDArray, y: NDArray) -> FrequencyEnsemble:
    """Fonction d'interface pour l'entraînement."""
    ratio = (y == 0).sum() / (y == 1).sum()
    model = FrequencyEnsemble(ratio=ratio)
    model.fit(X, y)
    return model