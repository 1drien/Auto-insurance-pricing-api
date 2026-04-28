<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:0d0221,30:0a1628,60:1a0533,100:2d1b69&height=280&section=header&text=Actuarial%20Pricing%20API&fontSize=52&fontColor=00d4ff&animation=fadeIn&fontAlignY=40&desc=⚡%20ML-Powered%20Auto%20Insurance%20Premium%20Engine%20⚡&descAlignY=62&descColor=bd93f9&stroke=00d4ff&strokeWidth=2" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Frequency+%C3%97+Severity+%C3%97+%3D+Final+Premium;XGBoost+%2B+HistGradientBoosting+Ensemble;FastAPI+%7C+Docker+%7C+GitHub+Actions+CI%2FCD;Built+by+%401drien+%C2%B7+%40elkiliayma-sys+%C2%B7+%40Kiane06" alt="Typing SVG" />

<br/><br/>

[![Python](https://img.shields.io/badge/Python_3.11-FFD43B?style=for-the-badge&logo=python&logoColor=306998)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/sklearn_1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-0db7ed?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![UV](https://img.shields.io/badge/UV-7C3AED?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)

<br/>

[![1drien](https://img.shields.io/badge/─%20@1drien%20─-0d1117?style=flat-square&logo=github&logoColor=00d4ff)](https://github.com/1drien)
[![elkiliayma](https://img.shields.io/badge/─%20@elkiliayma--sys%20─-0d1117?style=flat-square&logo=github&logoColor=bd93f9)](https://github.com/elkiliayma-sys)
[![Kiane06](https://img.shields.io/badge/─%20@Kiane06%20─-0d1117?style=flat-square&logo=github&logoColor=ff79c6)](https://github.com/Kiane06)

</div>

<br/>

---

## ◈ The Formula

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    Premium (TTC)  =  P(sinistre)  ×  Coût moyen  ×  1.18   ║
║                           ↑               ↑                  ║
║                     Frequency         Severity               ║
║                       Model            Model                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

<table align="center">
<tr>
<td align="center" width="50%">

###  Frequency Model
`XGBoost` + `HistGradientBoosting`
Ensemble + Isotonic Calibration
**→ P(au moins un sinistre)**

</td>
<td align="center" width="50%">

###  Severity Model
`XGBoost Regressor`
Entraîné sur échelle logarithmique
**→ Coût moyen si sinistre**

</td>
</tr>
</table>

---

## ◈ Architecture

```
 Prediction-du-Montant-des-Sinistres
│
├──  .github/workflows/ci.yml     ← CI/CD pipeline
├──  app.py                        ← FastAPI (4 routes)
├──  interface.py                 ← Streamlit web client
├──  main.py                       ← Model training pipeline
├──  Dockerfile                    ← Production image
├──  pyproject.toml                ← UV dependencies
│
├──  models/
│   ├── model_frequency.pkl          ← Frequency model
│   ├── model_severity.pkl           ← Severity model
│   └── feature_names.pkl            ← Feature alignment
│
├──  src/
│   ├── preprocessing.py             ← Feature engineering
│   ├── frequency.py                 ← Frequency model def
│   ├── severity.py                  ← Severity model def
│   ├── prime_cv.py                  ← OOF cross-validation
│   ├── evaluation.py                ← Metrics & diagnostics
│   └── visualization.py            ← Analysis charts
│
├──   tests/
│   ├── test_api.py                  ← 4 routes + validation
│   └── test_preprocessing.py       ← Unit tests
│
└──   data/
    ├── train.csv
    └── test.csv
```

---

## ◈ Getting Started

### Prerequisites

<div align="center">

[![Python](https://skillicons.dev/icons?i=python)](https://python.org)
[![Docker](https://skillicons.dev/icons?i=docker)](https://docker.com)
[![GitHub Actions](https://skillicons.dev/icons?i=githubactions)](https://github.com/features/actions)

</div>

###  Quick Install

```bash
# 1 — Install UV (blazing fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2 — Clone & install
git clone https://github.com/1drien/Prediction-du-Montant-des-Sinistres.git
cd Prediction-du-Montant-des-Sinistres
uv sync

# 3 — Launch API
uv run uvicorn app:app --reload
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs  (Swagger)

# 4 — Launch Streamlit UI
uv run streamlit run interface.py

# 5 — Run tests
uv run pytest tests/ -v
```

---

## ◈ API Routes

<div align="center">

| Method | Route | Description |
|:------:|-------|-------------|
| ![GET](https://img.shields.io/badge/GET-00C853?style=flat-square) | `/health` | Health check |
| ![POST](https://img.shields.io/badge/POST-2196F3?style=flat-square) | `/predict_frequency` | P(sinistre) |
| ![POST](https://img.shields.io/badge/POST-2196F3?style=flat-square) | `/predict_amount` | Coût moyen estimé |
| ![POST](https://img.shields.io/badge/POST-9C27B0?style=flat-square) | `/predict` | **Prime finale TTC** |

</div>

### Request

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

### Response

```json
{
  "predicted_claim_frequency":   0.0417,
  "estimated_severity_eur":   1860.79,
  "technical_pure_premium_eur":  77.62,
  "final_total_premium_ttc_eur": 91.59
}
```

---

## ◈ Docker

```bash
docker build -t actuarial-pricing-api .
docker run -p 8000:8000 actuarial-pricing-api
```

---

## ◈ CI/CD Pipeline

<div align="center">

```
  push to main/dev
        │
        ▼
  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  📦 Install  │────▶│  🔍  Lint    │────▶│  🧪  Tests   │────▶│  🐳  Docker  │
  │  UV + deps  │     │   Flake8     │     │  Pytest ×8   │     │ Image build  │
  └─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

</div>

---

## ◈ Tech Stack

<div align="center">

[![Python](https://skillicons.dev/icons?i=python)](https://python.org)
[![FastAPI](https://skillicons.dev/icons?i=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://skillicons.dev/icons?i=docker)](https://docker.com)
[![GitHub Actions](https://skillicons.dev/icons?i=githubactions)](https://github.com/features/actions)
[![Linux](https://skillicons.dev/icons?i=linux)](https://linux.org)

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI + Uvicorn |
| **ML** | scikit-learn 1.8 · XGBoost 2.1 |
| **Validation** | Pydantic + auto Swagger |
| **UI** | Streamlit |
| **Packages** | UV — `pyproject.toml` + `uv.lock` |
| **Container** | Docker |
| **CI/CD** | GitHub Actions |
| **Tests** | Pytest · 8 tests |
| **Quality** | Flake8 |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:2d1b69,50:1a0533,100:0d0221&height=140&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20%26%20XGBoost&fontSize=24&fontColor=00d4ff&animation=twinkling&desc=@1drien%20·%20@elkiliayma-sys%20·%20@Kiane06&descColor=bd93f9&descAlignY=72" width="100%"/>

</div>
