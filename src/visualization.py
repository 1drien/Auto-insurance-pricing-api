import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def save_plot(name, folder):
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{name}.png"), dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, title, folder, filename):
    """Affiche les variables les plus contributrices, gère les modèles calibrés."""
    
    # --- LOGIQUE D'EXTRACTION DE L'IMPORTANCE ---
    try:
        if hasattr(model, 'feature_importances_'):
            # Modèle standard (XGBoost, RandomForest)
            importance = model.feature_importances_
        elif hasattr(model, 'calibrated_classifiers_'):
            # Cas de votre modèle Fréquence (CalibratedClassifierCV)
            # On prend l'importance moyenne de tous les modèles de la calibration
            all_importances = [
                clf.estimator.feature_importances_ 
                for clf in model.calibrated_classifiers_
            ]
            importance = np.mean(all_importances, axis=0)
        else:
            print(f"Impossible d'extraire l'importance pour {title} : Type de modèle non supporté.")
            return
    except Exception as e:
        print(f"Erreur lors de l'extraction de l'importance ({title}) : {e}")
        return

    # --- LE RESTE DU CODE RESTE LE MÊME ---
    indices = np.argsort(importance)[-15:]  # Top 15 variables
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top 15 - Importance des Variables ({title})")
    plt.barh(range(len(indices)), importance[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Relative')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{filename}.png"), dpi=300)
    plt.close()

def plot_correlation_matrix(df, folder):
    """Matrice de corrélation pour identifier les redondances."""
    plt.figure(figsize=(12, 10))
    # On ne prend que les variables numériques principales pour la lisibilité
    cols = df.select_dtypes(include=[np.number]).columns[:20] 
    sns.heatmap(df[cols].corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Matrice de Corrélation (Variables Numériques)")
    save_plot("matrice_correlation", folder)

def plot_actuarial_analysis(df, folder):
    """Analyses métiers : Sinistres par Âge, Genre, et Puissance."""
    # ON TRAVAILLE SUR UNE COPIE POUR NE PAS POLLUER LE DF D'ENTRAINEMENT
    temp_df = df.copy() 
    
    # 1. Sinistres par tranche d'âge
    # On utilise temp_df au lieu de df
    temp_df['tranche_age'] = pd.cut(temp_df['age_conducteur1'], bins=[18, 25, 35, 50, 65, 90])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tranche_age', y='montant_sinistre', data=temp_df, estimator=np.mean)
    plt.title("Coût Moyen des Sinistres par Tranche d'Âge")
    save_plot("cout_moyen_par_age", folder)

    # 2. Sinistres par Genre
    if 'sex_conducteur1' in temp_df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='sex_conducteur1', y='montant_sinistre', data=temp_df[temp_df['montant_sinistre'] > 0])
        plt.yscale('log')
        plt.title("Distribution des Coûts par Genre (Échelle Log)")
        save_plot("distribution_genre", folder)

    # 3. Puissance véhicule vs Fréquence
    plt.figure(figsize=(10, 6))
    sns.regplot(x='din_vehicule', y='nombre_sinistres', data=temp_df, scatter_kws={'alpha':0.1}, lowess=False)
    plt.title("Relation : Puissance Véhicule vs Nombre de Sinistres")
    save_plot("puissance_vs_frequence", folder)

def plot_model_performance(y_true, y_pred, title, folder, filename):
    """Graphique de calibration : Valeurs Réelles vs Prédictions."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, color='orange')
    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'k--', lw=2)
    plt.xlabel('Valeurs Réelles (€)')
    plt.ylabel('Valeurs Prédites (€)')
    plt.title(f"Calibration : {title}")
    save_plot(filename, folder)