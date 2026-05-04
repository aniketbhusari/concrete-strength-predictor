import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. Page Configuration (Wide layout looks more professional)
st.set_page_config(page_title="Concrete AI", page_icon="🏗️", layout="wide")

# Custom CSS for cleaner spacing
st.markdown("""
    <style>
    .main {padding-top: 2rem;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white;}
    </style>
    """, unsafe_allow_html=True)

# 2. Load the Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('best_concrete_model.pkl')
    scaler = joblib.load('concrete_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. App Header
st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("Predict the compressive strength of high-performance concrete using our trained **XGBoost Machine Learning Model**.")
st.divider()

# Layout: Split the screen into two main columns (Inputs on left, Output on right)
col_inputs, col_output = st.columns([1.5, 1], gap="large")

with col_inputs:
    st.subheader("⚙️ Mix Design Parameters")
    
    # Use tabs to organize inputs cleanly
    tab1, tab2 = st.tabs(["Binders & Water", "Aggregates & Admixtures"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 281.0, help="Main binding material")
            slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 74.0, help="Supplementary cementitious material")
        with col2:
            fly_ash = st.number_input("Fly Ash (kg/m³)", 0.0, 300.0, 54.0, help="Improves workability and late strength")
            water = st.number_input("Water (kg/m³)", 100.0, 250.0, 181.0, help="Crucial for hydration; affects water-cement ratio")
            
    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 50.0, 6.0, help="Water reducer for high-performance concrete")
            age = st.slider("Curing Age (days)", 1, 365, 28, help="Standard testing is usually done at 28 days")
        with col4:
            coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 800.0, 1200.0, 973.0)
            fine_agg = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 774.0)

    st.write("") # spacing
    predict_btn = st.button("Calculate Compressive Strength", type="primary")

with col_output:
    st.subheader("📊 Prediction Results")
    
    if predict_btn:
        # Bundle inputs
        input_values = [[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]]
        input_data = pd.DataFrame(input_values, columns=scaler.feature_names_in_)
        
        # Scale and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        # Draw a beautiful Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Strength (MPa)", 'font': {'size': 24}},
            number = {'suffix': " MPa", 'font': {'size': 40, 'color': "darkgreen"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkgreen"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': "#ffcccb"},   # Low strength (Reddish)
                    {'range': [20, 40], 'color': "#ffffe0"},  # Normal strength (Yellowish)
                    {'range': [40, 100], 'color': "#e0ffe0"}  # High strength (Greenish)
                ]
            }
        ))
        
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Note:** This prediction is based on the XGBoost model which achieved an R² score of 0.90 on the training dataset.")
    else:
        st.write("👈 Enter your mix design parameters on the left and click **Calculate** to see the interactive results.")
