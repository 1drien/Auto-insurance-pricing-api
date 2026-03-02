# =========================================================
# SCRIPT UTILITAIRE : VÉRIFICATION DE LA STRUCTURE DES DONNÉES
# =========================================================

import pandas as pd

# Chargement d'un échantillon réduit (5 premières lignes) du fichier de test.
# Cette approche permet d'inspecter rapidement la structure (schéma) des données 
# sans nécessiter le chargement de l'intégralité du fichier en mémoire.
df_test = pd.read_csv("data/test.csv", nrows=5)

print("--- STRUCTURE DU FICHIER : NOMS DES VARIABLES ---")
# Extraction et affichage sous forme de liste des noms de colonnes.
# Cette vérification est cruciale pour s'assurer que le fichier d'inférence 
# correspond rigoureusement au format attendu par le pipeline de prétraitement.
print(df_test.columns.tolist())

print("\n--- APERÇU DES IDENTIFIANTS UNIQUES (CLÉS PRIMAIRES) ---")
# Sélection de la première colonne via une indexation positionnelle (iloc).
# En contexte actuariel ou d'évaluation (Kaggle), l'identifiant unique 
# (ID ou Index) est généralement situé en première position. Cette étape 
# valide son format avant la génération du fichier de soumission.
print(df_test.iloc[:, 0].head())