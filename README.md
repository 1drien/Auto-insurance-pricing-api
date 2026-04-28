<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:0d0221,30:0a1628,60:1a0533,100:2d1b69&height=280&section=header&text=Actuarial%20Pricing%20API&fontSize=52&fontColor=00d4ff&animation=fadeIn&fontAlignY=40&desc=⚡%20ML-Powered%20Auto%20Insurance%20Premium%20Engine%20⚡&descAlignY=62&descColor=bd93f9&stroke=00d4ff&strokeWidth=2" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Frequency+%C3%97+Severity+%C3%97+1.18+%3D+Final+Premium;XGBoost+%2B+HistGradientBoosting+Ensemble;FastAPI+%7C+Docker+%7C+GitHub+Actions+CI%2FCD;Built+by+%401drien+%C2%B7+%40elkiliayma-sys+%C2%B7+%40Kiane06" alt="Typing SVG" />

<br/><br/>
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189AB4?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![UV](https://img.shields.io/badge/UV-Package%20Manager-7C3AED?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)

<br/>

> **REST API for auto insurance pricing** based on two Machine Learning models (frequency + severity).  
> Industrializes notebook models into a production-ready REST API with Docker & CI/CD.

<br/>

**Authors**

[![1drien](https://img.shields.io/badge/@1drien-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/1drien)
[![elkiliayma-sys](https://img.shields.io/badge/@elkiliayma--sys-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/elkiliayma-sys)
[![Kiane06](https://img.shields.io/badge/@Kiane06-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Kiane06)

</div>

---

##  Pricing Principle

<div align="center">

```
Final Premium (incl. tax) = P(claim) × Average claim cost × 1.18
```

</div>

| Model | Algorithm | Role |
|-------|-----------|------|
|  **Frequency** | XGBoost + HistGradientBoosting + Isotonic Calibration | Predicts P(at least one claim) |
|  **Severity** | XGBoost Regressor (log scale) | Predicts average cost if claim occurs |

Models are trained via `main.py`, serialized with pickle into `models/`, then loaded by the API at startup.

---

##  Project Architecture

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

##  Installation & Setup

### Prerequisites

![Python](https://img.shields.io/badge/Python-≥3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![UV](https://img.shields.io/badge/UV-required-7C3AED?style=flat-square)

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

> API available at `http://127.0.0.1:8000` · Swagger docs at `http://127.0.0.1:8000/docs`

### Run the Streamlit Interface

```bash
uv run streamlit run interface.py
```

### Run Tests

```bash
uv run pytest tests/ -v
```

---

##  API Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Check API health status |
| `POST` | `/predict_frequency` | Predict claim probability |
| `POST` | `/predict_amount` | Predict average claim cost |
| `POST` | `/predict` | Calculate full premium (frequency × severity × 1.18) |

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

##  Docker

```bash
# Build the image
docker build -t actuarial-pricing-api .

# Run the container
docker run -p 8000:8000 actuarial-pricing-api
```

---

##  CI/CD

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs automatically on every push to `main` or `dev`:

```
Push to main/dev
      │
      ▼
 1. Install ──► UV + dependencies
      │
      ▼
 2. Lint ────► Flake8 (code quality)
      │
      ▼
 3. Tests ───► Pytest (8 tests: preprocessing + API routes)
      │
      ▼
 4. Docker ──► Image build validation
```

---

##  Dependency Management

This project uses **UV** instead of pip/requirements.txt:

| File | Role |
|------|------|
| `pyproject.toml` | Project description, separates prod vs dev dependencies |
| `uv.lock` | Pins exact versions for full reproducibility |
| `uv sync` | Installs the environment |
| `uv run` | Executes within the virtual environment |

---

##  Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI + Uvicorn |
| **ML** | scikit-learn 1.8 · XGBoost 2.1 |
| **Validation** | Pydantic (schemas + auto Swagger) |
| **Interface** | Streamlit |
| **Dependencies** | UV (pyproject.toml + uv.lock) |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Tests** | Pytest (8 tests) |
| **Code quality** | Flake8 |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:2d1b69,50:1a0533,100:0d0221&height=140&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20%26%20XGBoost&fontSize=24&fontColor=00d4ff&animation=twinkling&desc=@1drien%20·%20@elkiliayma-sys%20·%20@Kiane06&descColor=bd93f9&descAlignY=72" width="100%"/>

</div>
