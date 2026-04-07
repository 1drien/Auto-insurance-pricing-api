# 🛡️ Modèle de Tarification Assurance Auto (Severity & Frequency)

![Status](https://img.shields.io/badge/Status-Termin%C3%A9-green)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Valid%C3%A9-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Pass-success)

Projet de Data Science et ML Engineering visant à prédire la **Prime Pure** d'assurance automobile. 
Le projet intègre désormais une **API FastAPI** industrialisée via **Docker** et une **CI/CD GitHub Actions**.

👤 **Auteurs :** [@1drien](https://github.com/1drien)
* [@elkiliayma-sys](https://github.com/elkiliayma-sys)
---

## 📂 Architecture du projet

Le code est structuré de manière modulaire et professionnelle :

```text
.
├── .github/workflows/    # Pipeline CI/CD (Tests, Lint, Docker Build)
├── data/                 # Données d'entraînement et test
├── models/               # Modèles entraînés (XGBoost & HistGradient)
├── src/                  # Logique métier
│   ├── preprocessing.py  # Nettoyage et préparation des données
│   ├── severity.py       # Modèle de coût moyen
│   └── frequency.py      # Modèle de probabilité de sinistre
├── tests/                # Tests unitaires (Pytest)
├── app.py                # API FastAPI (Point d'entrée de production)
├── Dockerfile            # Recette de conteneurisation
└── requirements.txt      # Dépendances du projet