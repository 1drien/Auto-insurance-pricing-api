import streamlit as st
import requests

st.set_page_config(page_title="Actuarial Pricing Simulator", page_icon="📊", layout="wide")

st.title("🚗 Auto Insurance Pricing Simulator")
st.markdown("---")

# API URL in the sidebar
url_api = st.sidebar.text_input("FastAPI Endpoint URL", "http://127.0.0.1:8000/predict")

# FULL FORM
with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Driver Profile")
        age = st.number_input("Driver Age", 18, 100, 24)
        permis = st.number_input("Years of License", 0, 80, 2)
        sexe = st.selectbox("Gender", ["M", "F"])
        utilisation = st.selectbox("Vehicle Usage", ["WorkPrivate", "Retired", "Professional", "AllTrips"])
        paiement = st.selectbox("Payment Frequency", ["Monthly", "Yearly", "Quarterly", "Biannual"])

    with col2:
        st.subheader("🚙 Vehicle Details")
        marque = st.selectbox("Car Brand", 
                              ["Renault", "Peugeot", "Citroen", "Volkswagen", "BMW", "Audi", "Fiat", "Ford", "Mercedes", "Opel", "Toyota", "Other"])
        
        prix = st.number_input("Vehicle Purchase Price (€)", value=25000)
        poids = st.number_input("Vehicle Weight (kg)", value=1100)
        puissance = st.number_input("Engine Power (DIN hp)", value=130)
        categorie = st.selectbox("Vehicle Category", ["Tourism", "Commercial"])

    st.markdown("---")
    submit = st.form_submit_button("🚀 CALCULATE PREMIUM")

# SENDING LOGIC
if submit:
    # Payload construction exactly like your successful unit test
    payload = {
        "age_conducteur1": int(age),
        "anciennete_permis1": int(permis),
        "sex_conducteur1": sexe,
        "din_vehicule": int(puissance),
        "poids_vehicule": int(poids),
        "utilisation": utilisation,
        "marque_vehicule": marque,
        "prix_vehicule": float(prix),
        "type_vehicule": categorie,
        "freq_paiement": paiement
    }

    try:
        r = requests.post(url_api, json=payload, timeout=10)
        
        if r.status_code == 200:
            res = r.json()
            st.success("✅ Calculation successful!")
            
            # CLEAN AND ROBUST DISPLAY
            st.subheader("📈 Prediction Results")
            
            # Using Alias names from your app.py
            prob = res.get("predicted_claim_frequency", 0)
            sev = res.get("estimated_severity_eur", 0)
            pure = res.get("technical_pure_premium_eur", 0)
            total = res.get("final_total_premium_ttc_eur", 0)

            c1, c2, c3 = st.columns(3)
            c1.metric("Claim Probability", f"{prob*100:.2f} %")
            c2.metric("Estimated Severity", f"{sev:.2f} €")
            c3.metric("FINAL PREMIUM (TTC)", f"{total:.2f} €")
            
            # Detailed JSON data hidden in expander
            with st.expander("Show raw API response (JSON)"):
                st.json(res)
        else:
            st.error(f"Error {r.status_code}")
            st.write(r.text)
            
    except Exception as e:
        st.error(f"Connection Error: {e}")