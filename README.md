# Actuarial Pricing API — Assurance Auto

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)
![UV](https://img.shields.io/badge/UV-Package%20Manager-blueviolet)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success)

API de tarification assurance automobile basée sur deux modèles de Machine Learning (fréquence + sévérité). Ce projet industrialise des modèles développés en notebook pour les exposer via une API REST, avec conteneurisation Docker et intégration continue.

**Auteurs :** [@1drien](https://github.com/1drien) · [@elkiliayma-sys](https://github.com/elkiliayma-sys) · [@Kiane06](https://github.com/Kiane06)

---

## Principe de tarification

La prime d'assurance est calculée selon la formule actuarielle :

```
Prime TTC = P(sinistre) × Coût moyen du sinistre × 1.18
```

Deux modèles ML interviennent :
- **Modèle de fréquence** — Ensemble XGBoost + HistGradientBoosting avec calibration isotonique. Prédit la probabilité qu'un assuré ait au moins un sinistre.
- **Modèle de sévérité** — XGBoost régresseur entraîné en échelle logarithmique. Prédit le coût moyen si un sinistre survient.

Les modèles sont entraînés via `main.py`, sérialisés avec pickle dans `models/`, puis chargés par l'API au démarrage.

---

## Architecture du projet

```
.
├── .github/workflows/ci.yml   # Pipeline CI/CD GitHub Actions
├── app.py                     # API FastAPI (4 routes)
├── interface.py               # Interface Streamlit (client web)
├── main.py                    # Pipeline d'entraînement des modèles
├── conftest.py                # Configuration pytest
├── Dockerfile                 # Image Docker de production
├── pyproject.toml             # Dépendances et métadonnées (UV)
├── uv.lock                    # Versions figées des dépendances
├── models/
│   ├── model_frequency.pkl    # Modèle de fréquence sérialisé
│   ├── model_severity.pkl     # Modèle de sévérité sérialisé
│   └── feature_names.pkl      # Noms des features pour l'alignement
├── src/
│   ├── preprocessing.py       # Feature engineering + traitement unitaire
│   ├── frequency.py           # Définition du modèle de fréquence
│   ├── severity.py            # Définition du modèle de sévérité
│   ├── prime_cv.py            # Validation croisée Out-Of-Fold
│   ├── evaluation.py          # Métriques et diagnostics
│   └── visualization.py       # Graphiques d'analyse
├── tests/
│   ├── test_api.py            # Tests des 4 routes API + validation
│   └── test_preprocessing.py  # Tests du preprocessing unitaire
├── data/
│   ├── train.csv              # Données d'entraînement
│   └── test.csv               # Données de test
└── notebooks/
    └── eda_preprocessing.ipynb # Exploration et développement
```

---

## Installation et lancement

### Prérequis

- Python ≥ 3.11
- [UV](https://docs.astral.sh/uv/) (gestionnaire de dépendances)

### Installation

```bash
# Installer UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Cloner le projet
git clone https://github.com/1drien/Prediction-du-Montant-des-Sinistres.git
cd Prediction-du-Montant-des-Sinistres

# Installer les dépendances
uv sync
```

### Lancer l'API

```bash
uv run uvicorn app:app --reload
```

L'API est accessible sur `http://127.0.0.1:8000`. La documentation Swagger est sur `http://127.0.0.1:8000/docs`.

### Lancer l'interface Streamlit

```bash
uv run streamlit run interface.py
```

### Lancer les tests

```bash
uv run pytest tests/ -v
```

---

## Routes de l'API

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Vérifie l'état de santé de l'API |
| POST | `/predict_frequency` | Prédit la probabilité de sinistre |
| POST | `/predict_amount` | Prédit le coût moyen d'un sinistre |
| POST | `/predict` | Calcule la prime complète (fréquence × sévérité × 1.18) |

### Exemple de requête

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Exemple de réponse

```json
{
  "predicted_claim_frequency": 0.0417,
  "estimated_severity_eur": 1860.79,
  "technical_pure_premium_eur": 77.62,
  "final_total_premium_ttc_eur": 91.59
}
```

---

## Docker

```bash
# Construire l'image
docker build -t actuarial-pricing-api .

# Lancer le conteneur
docker run -p 8000:8000 actuarial-pricing-api
```

---

## CI/CD

Le pipeline GitHub Actions (`.github/workflows/ci.yml`) s'exécute automatiquement à chaque push sur `main` ou `dev` :

1. **Installation** — UV + dépendances
2. **Lint** — Flake8 (qualité du code)
3. **Tests** — Pytest (8 tests : preprocessing + routes API)
4. **Docker** — Validation du build de l'image

---

## Gestion des dépendances

Le projet utilise **UV** au lieu de pip/requirements.txt :

- `pyproject.toml` — décrit le projet et sépare dépendances de production et de développement
- `uv.lock` — fige les versions exactes pour garantir la reproductibilité
- `uv sync` — installe l'environnement, `uv run` — exécute dans l'environnement virtuel

---

## Stack technique

- **API** — FastAPI + Uvicorn
- **ML** — scikit-learn 1.8, XGBoost 2.1
- **Validation** — Pydantic (schémas input/output + documentation Swagger auto-générée)
- **Interface** — Streamlit
- **Dépendances** — UV (pyproject.toml + uv.lock)
- **Conteneurisation** — Docker
- **CI/CD** — GitHub Actions
- **Tests** — Pytest (8 tests)
- **Qualité** — Flake8
