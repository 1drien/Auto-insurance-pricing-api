import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.preprocessing import preprocess_single_event

# 1. API INITIALIZATION
app = FastAPI(
    title="Actuarial Pricing API - M2 FINTECH",
    description=(
        "### Industrialized Insurance Pricing Solution\n"
        "This API provides real-time insurance premium calculations based on two Machine Learning models.\n\n"
        "**Standard Workflow:**\n"
        "1. **Input:** Descriptive driver and vehicle profile (JSON).\n"
        "2. **Processing:** Automatic feature engineering (ratios, encoding).\n"
        "3. **Output:** Predicted claim frequency, estimated cost (severity), and total premium (TTC)."
    ),
    version="1.0.0"
)

# 2. MODEL LOADING
MODELS_DIR = "models"
try:
    with open(os.path.join(MODELS_DIR, "model_frequency.pkl"), "rb") as f:
        model_freq = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "model_severity.pkl"), "rb") as f:
        model_sev = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
        feats_name = pickle.load(f)
    print("[INFO] Models and features successfully loaded.")
except Exception as e:
    print(f"[ERROR] Loading failed: {e}")

# 3. DATA SCHEMA WITH FULL EXPLANATIONS (For the Schemas section)
class PricingResponse(BaseModel):
    frequency: float = Field(
        ..., 
        alias="predicted_claim_frequency", 
        description="Probability of at least one claim occurring (0.0 to 1.0)",
        example=0.0368
    )
    severity: float = Field(
        ..., 
        alias="estimated_severity_eur", 
        description="Average estimated cost if a claim occurs",
        example=1979.50
    )
    pure_premium: float = Field(
        ..., 
        alias="technical_pure_premium_eur", 
        description="Expected loss cost (Frequency * Severity)",
        example=72.81
    )
    total_premium: float = Field(
        ..., 
        alias="final_total_premium_ttc_eur", 
        description="Final price charged to the client including",
        example=85.92
    )

    class Config:
        populate_by_name = True
        title = "Insurance Pricing Results"

class InsuranceObservation(BaseModel):
    age_conducteur1: int = Field(
        ..., 
        alias="Primary_Driver_Age", 
        example=30,
        description="**Age of the main policyholder.** Must be an integer; the Age"
    )
    anciennete_permis1: int = Field(
        ..., 
        alias="Years_of_License", 
        example=10,
        description="**Driving experience.** Number of years since the driving license was issued"
    )
    sex_conducteur1: str = Field(
        ..., 
        alias="Driver_Gender", 
        example="M",
        description="**Gender of the driver.** 'M' for Male or 'F' for Female"
    )
    din_vehicule: int = Field(
        ..., 
        alias="Engine_Horsepower", 
        example=110,
        description="**Vehicle engine power (DIN hp).** Must be an integer representing the engine power of the car"
    )
    poids_vehicule: int = Field(
        ..., 
        alias="Vehicle_Weight_KG", 
        example=1300,
        description="**Vehicle curb weight in kilograms.** The weight of the Car"
    )
    utilisation: str = Field(
        ..., 
        alias="Usage_Type", 
        example="WorkPrivate",
        description="**Main use of the vehicle.** 4 Options: 'Retired', 'WorkPrivate', 'Professional', or 'AllTrips'"
    )
    marque_vehicule: str = Field(
        ..., 
        alias="Car_Brand", 
        example="Renault",
        description="**Manufacturer brand.** The name of the Car Brand"
    )
    prix_vehicule: float = Field(
        ..., 
        alias="Purchase_Price_EUR", 
        example=25000.0,
        description="**Market value of the car in Euros.** The price of the car in Euros"
    )
    type_vehicule: str = Field(
        ..., 
        alias="Vehicle_Category", 
        example="Tourism",
        description="**Vehicle type.** 'Tourism' for Standard car or 'Commercial' for Van/Utility"
    )
    freq_paiement: str = Field(
        ..., 
        alias="Payment_Frequency", 
        example="Monthly",
        description="**Premium billing cycle.** 4 Options : 'Monthly', 'Quarterly', 'Biannual' or 'Yearly'"
    )

    class Config:
        populate_by_name = True
        # This title will appear in the Schemas section
        title = "Individual Insurance Policy Input"

# 4. THE 4 MANDATORY ROUTES (Pillars)

@app.get("/health", tags=["System Monitoring"], summary="Check API Status")
def health():
    """Returns the health status of the API and models."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict_frequency", tags=["Pricing Models"], summary="Predict Claim Frequency")
def predict_frequency(observation: InsuranceObservation):
    """Predicts the probability of at least one claim (Frequency)."""
    data = observation.model_dump(by_alias=False)
    X = preprocess_single_event(data, feats_name)
    prob = float(model_freq.predict_proba(X)[:, 1][0])
    return {"Predicted_Claim_Probability": round(prob, 4)}

@app.post("/predict_amount", tags=["Pricing Models"], summary="Predict Claim Severity")
def predict_amount(observation: InsuranceObservation):
    """Predicts the estimated cost per claim (Severity)."""
    data = observation.model_dump(by_alias=False)
    X = preprocess_single_event(data, feats_name)
    amount = float(np.expm1(model_sev.predict(X))[0])
    return {"Estimated_Claim_Cost_EUR": round(amount, 2)}

@app.post("/predict", tags=["Pricing Models"], summary="Calculate Full Final Premium", response_model=PricingResponse)
def predict_all(observation: InsuranceObservation):
    """Calculates all components: Frequency, Severity, and Final Premium."""
    data = observation.model_dump(by_alias=False)
    X = preprocess_single_event(data, feats_name)
    
    prob = float(model_freq.predict_proba(X)[:, 1][0])
    amount = float(np.expm1(model_sev.predict(X))[0])
    
    pure_p = prob * amount
    total_p = pure_p * 1.18
    
    # On retourne un dictionnaire avec les clés internes de PricingResponse
    # FastAPI s'occupera d'afficher les ALIAS dans Swagger automatiquement
    return {
        "frequency": round(prob, 4),
        "severity": round(amount, 2),
        "pure_premium": round(pure_p, 2),
        "total_premium": round(max(0, total_p), 2)
    }