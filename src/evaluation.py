# =========================================================
# MODULE D'ÉVALUATION ET DE DIAGNOSTIC DES MODÈLES
# =========================================================

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

# Configuration du backend matplotlib pour un environnement sans interface graphique
# Permet de générer et sauvegarder les graphiques sans bloquer l'exécution du script.
import matplotlib
matplotlib.use('Agg')

def plot_feature_importance(model, feature_names, save_path):
    """
    Génère et exporte le graphique d'importance des variables explicatives.
    
    Cette fonction permet d'interpréter le modèle XGBoost en identifiant 
    les caractéristiques ayant le plus grand pouvoir discriminant.
    
    Paramètres :
    - model : L'estimateur XGBoost entraîné.
    - feature_names : Liste des noms des variables explicatives.
    - save_path : Chemin de sauvegarde de la figure générée.
    """
    # Réinjection explicite des noms de variables dans l'objet Booster
    model.get_booster().feature_names = list(feature_names)
    
    plt.figure(figsize=(10, 8))
    
    # importance_type='weight' : Mesure l'importance par la fréquence d'utilisation 
    # de la variable comme critère de scission (split) dans l'ensemble des arbres.
    xgb.plot_importance(
        model, 
        max_num_features=15, 
        height=0.5, 
        importance_type='weight', 
        title="Importance des variables explicatives (XGBoost)"
    )
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graphique d'importance des variables exporté sous : {save_path}")


def check_overfitting(model, X, y, test_rmse_avg):
    """
    Analyse l'écart de performance entre l'échantillon d'apprentissage et 
    l'échantillon de validation croisée pour diagnostiquer le surapprentissage.
    
    Paramètres :
    - model : L'estimateur entraîné.
    - X, y : Données d'apprentissage.
    - test_rmse_avg : Erreur RMSE moyenne estimée via validation croisée.
    """
    print("\n--- ANALYSE DE LA CAPACITÉ DE GÉNÉRALISATION (OVERFITTING) ---")
    
    # 1. Inférence sur les données d'apprentissage (Performance intra-échantillon)
    pred_log = model.predict(X)
    
    # Transformation inverse (Logarithme vers valeur monétaire réelle)
    pred_euros = np.expm1(pred_log)
    y_true_euros = np.expm1(y)
    
    # Calcul de la métrique d'erreur sur l'ensemble d'apprentissage
    rmse_train = np.sqrt(mean_squared_error(y_true_euros, pred_euros))
    
    # 2. Restitution des métriques comparatives
    print(f"Erreur d'apprentissage (Train RMSE)       : {rmse_train:.2f} EUR")
    print(f"Erreur de validation (Test RMSE estimé)   : {test_rmse_avg:.2f} EUR")
    
    ecart = test_rmse_avg - rmse_train
    print(f"Écart de généralisation                   : {ecart:.2f} EUR")
    
    # 3. Diagnostic automatisé
    if ecart > 300:
        print("DIAGNOSTIC : Risque significatif de surapprentissage (Overfitting) détecté.")
    elif ecart < 0:
        print("DIAGNOSTIC : Écart négatif atypique (Sous-apprentissage ou variance d'échantillonnage).")
    else:
        print("DIAGNOSTIC : Capacité de généralisation satisfaisante (Architecture robuste).")


def plot_frequency_metrics(y_true, y_prob, save_path):
    """
    Génère les graphiques d'évaluation des performances pour le modèle de fréquence.
    Inclut l'analyse de calibration (fiabilité des probabilités) et l'analyse ROC 
    (pouvoir de discrimination).
    
    Paramètres :
    - y_true : Vecteur des classes réelles (0 ou 1).
    - y_prob : Probabilités prédites par le modèle.
    - save_path : Chemin de sauvegarde de la figure générée.
    """
    plt.figure(figsize=(12, 5))
    
    # 1. Courbe de Calibration (Fiabilité probabiliste)
    # Évalue si les probabilités prédites correspondent aux fréquences empiriques observées.
    plt.subplot(1, 2, 1)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Modèle calibré', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibration idéale')
    plt.xlabel("Probabilité Prédite")
    plt.ylabel("Fréquence Empirique Réelle")
    plt.title("Analyse de la Calibration")
    plt.legend()
    plt.grid(True)
    
    # 2. Courbe ROC (Receiver Operating Characteristic)
    # Évalue la capacité du modèle à discriminer les classes (Sinistré vs Non-sinistré).
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.plot(fpr, tpr, color='orange', lw=2, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graphiques d'évaluation de la fréquence exportés sous : {save_path}")