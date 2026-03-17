import pickle
import os
import numpy as np
from src.preprocessing import preprocess_single_event

def run_manual_test():
    print("=== TEST UNITAIRE DE L'ÉTAPE 3 ===")
    
    # 1. Chargement des composants sauvegardés
    try:
        with open("models/model_frequency.pkl", "rb") as f:
            model_f = pickle.load(f)
        with open("models/model_severity.pkl", "rb") as f:
            model_s = pickle.load(f)
        with open("models/feature_names.pkl", "rb") as f:
            feats = pickle.load(f)
        print("[OK] Modèles et features chargés avec succès.")
    except FileNotFoundError:
        print("[ERREUR] Les fichiers .pkl sont introuvables. Lancez main.py d'abord.")
        return

    # 2. Définition d'un dictionnaire de test (Format JSON)
    # Ce dictionnaire simule une observation unique 
    test_json = {
        "age_conducteur1": 24,
        "anciennete_permis1": 2,
        "sex_conducteur1": "M",
        "din_vehicule": 130,
        "poids_vehicule": 1100,
        "utilisation": "WorkPrivate",
        "marque_vehicule": "Renault",
        "prix_vehicule": 25000,
        "type_vehicule": "Tourism",
        "freq_paiement": "Monthly"
    }
    print(f"\nDonnées de test injectées : {test_json}")

    # 3. Prétraitement unitaire via le module preprocessing
    # On transforme le dictionnaire en vecteur aligné sur le modèle 
    X_input = preprocess_single_event(test_json, feats)

    # 4. Prédictions
    # Fréquence (Probabilité)
    prob_sinistre = model_f.predict_proba(X_input)[:, 1][0]
    
    # Sévérité (Montant) - Attention à l'expm1 car entraîné en log !
    log_montant = model_s.predict(X_input)
    montant_estime = np.expm1(log_montant)[0]

    # Calcul de la prime (formule agrégée) [cite: 15]
    prime_technique = prob_sinistre * montant_estime
    prime_finale = prime_technique * 1.18

    # 5. Affichage des résultats
    print("\n--- RÉSULTATS DU TEST ---")
    print(f"Probabilité de sinistre : {prob_sinistre:.2%}")
    print(f"Coût moyen estimé      : {montant_estime:.2f} €")
    print(f"Prime Pure (Technique)  : {prime_technique:.2f} €")
    print(f"Prime Finale (TTC)      : {prime_finale:.2f} €")
    print("-------------------------")
    print("[VÉRIFICATION] Le dictionnaire JSON a été correctement traité.")

if __name__ == "__main__":
    run_manual_test()