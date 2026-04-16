# Actuarial Pricing API — Auto Insurance

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)
![UV](https://img.shields.io/badge/UV-Package%20Manager-blueviolet)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success)

REST API for auto insurance pricing based on two Machine Learning models (frequency + severity). This project industrializes models developed in notebooks and exposes them via a REST API, with Docker containerization and continuous integration.

**Authors:** [@1drien](https://github.com/1drien) · [@elkiliayma-sys](https://github.com/elkiliayma-sys) · [@Kiane06](https://github.com/Kiane06)

---

## Pricing Principle

The insurance premium is calculated using the actuarial formula:

```
Final Premium (incl. tax) = P(claim) × Average claim cost × 1.18
```

Two ML models are involved:
- **Frequency model** — XGBoost + HistGradientBoosting ensemble with isotonic calibration. Predicts the probability that a policyholder will have at least one claim.
- **Severity model** — XGBoost regressor trained on a log scale. Predicts the average cost if a claim occurs.

Models are trained via `main.py`, serialized with pickle into `models/`, then loaded by the API at startup.

---

## Project Architecture

```
.
├── .github/workflows/ci.yml   # CI/CD GitHub Actions pipeline
├── app.py                     # FastAPI application (4 routes)
├── interface.py               # Streamlit interface (web client)
├── main.py                    # Model training pipeline
├── conftest.py                # Pytest configuration
├── Dockerfile                 # Production Docker image
├── pyproject.toml             # Dependencies and metadata (UV)
├── uv.lock                    # Pinned dependency versions
├── models/
│   ├── model_frequency.pkl    # Serialized frequency model
│   ├── model_severity.pkl     # Serialized severity model
│   └── feature_names.pkl      # Feature names for alignment
├── src/
│   ├── preprocessing.py       # Feature engineering + unit processing
│   ├── frequency.py           # Frequency model definition
│   ├── severity.py            # Severity model definition
│   ├── prime_cv.py            # Out-Of-Fold cross-validation
│   ├── evaluation.py          # Metrics and diagnostics
│   └── visualization.py       # Analysis charts
├── tests/
│   ├── test_api.py            # Tests for all 4 API routes + validation
│   └── test_preprocessing.py  # Unit tests for preprocessing
├── data/
│   ├── train.csv              # Training data
│   └── test.csv               # Test data
└── notebooks/
    └── eda_preprocessing.ipynb # Exploration and development
```

---

## Installation & Setup

### Prerequisites

- Python ≥ 3.11
- [UV](https://docs.astral.sh/uv/) (dependency manager)

### Installation

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the repository
git clone https://github.com/1drien/Prediction-du-Montant-des-Sinistres.git
cd Prediction-du-Montant-des-Sinistres

# Install dependencies
uv sync
```

### Run the API

```bash
uv run uvicorn app:app --reload
```

The API is available at `http://127.0.0.1:8000`. Swagger documentation is at `http://127.0.0.1:8000/docs`.

### Run the Streamlit Interface

```bash
uv run streamlit run interface.py
```

### Run Tests

```bash
uv run pytest tests/ -v
```

---

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Check API health status |
| POST | `/predict_frequency` | Predict claim probability |
| POST | `/predict_amount` | Predict average claim cost |
| POST | `/predict` | Calculate full premium (frequency × severity × 1.18) |

### Example Request

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

### Example Response

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
# Build the image
docker build -t actuarial-pricing-api .

# Run the container
docker run -p 8000:8000 actuarial-pricing-api
```

---

## CI/CD

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs automatically on every push to `main` or `dev`:

1. **Install** — UV + dependencies
2. **Lint** — Flake8 (code quality)
3. **Tests** — Pytest (8 tests: preprocessing + API routes)
4. **Docker** — Image build validation

---

## Dependency Management

This project uses **UV** instead of pip/requirements.txt:

- `pyproject.toml` — describes the project and separates production from development dependencies
- `uv.lock` — pins exact versions for reproducibility
- `uv sync` — installs the environment; `uv run` — executes within the virtual environment

---

## Tech Stack

- **API** — FastAPI + Uvicorn
- **ML** — scikit-learn 1.8, XGBoost 2.1
- **Validation** — Pydantic (input/output schemas + auto-generated Swagger docs)
- **Interface** — Streamlit
- **Dependencies** — UV (pyproject.toml + uv.lock)
- **Containerization** — Docker
- **CI/CD** — GitHub Actions
- **Tests** — Pytest (8 tests)
- **Code quality** — Flake8
