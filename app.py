import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")

# 2. Load the Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('best_concrete_model.pkl')
    scaler = joblib.load('concrete_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. App Header
st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("""
This application uses a trained **XGBoost Machine Learning Model** to predict the compressive strength of high-performance concrete. 
Enter your mix design parameters below.
---
""")

# 4. User Input Fields
st.subheader("Mix Proportions & Curing Age")

col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cement (kg/m³)", min_value=100.0, max_value=600.0, value=281.0, step=5.0)
    slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=400.0, value=74.0, step=5.0)
    fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=300.0, value=54.0, step=5.0)
    water = st.number_input("Water (kg/m³)", min_value=100.0, max_value=250.0, value=181.0, step=2.0)

with col2:
    superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=50.0, value=6.0, step=1.0)
    coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=800.0, max_value=1200.0, value=973.0, step=10.0)
    fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=500.0, max_value=1000.0, value=774.0, step=10.0)
    age = st.number_input("Curing Age (days)", min_value=1, max_value=365, value=28, step=1)

# 5. Prediction Logic
st.markdown("---")
if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
    
    # Bundle the user inputs into a 2D array
    input_values = [[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]]
    
    # THE FIX: Dynamically apply the exact column names the scaler memorized!
    input_data = pd.DataFrame(input_values, columns=scaler.feature_names_in_)
    
    # Apply the exact same scaling used during training
    scaled_input = scaler.transform(input_data)
    
    # Generate the prediction
    prediction = model.predict(scaled_input)[0]
    
   # Display the result
    st.success("Prediction Complete!")
    st.markdown(f"### 🏗️ Predicted Strength: **{prediction:.2f} MPa**")
