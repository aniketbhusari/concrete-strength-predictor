import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CONCRETIQ · AI Strength Lab",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  ── industrial dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:       #080c10;
    --bg-panel:      #0d1117;
    --bg-glass:      rgba(255,255,255,0.035);
    --border:        rgba(255,255,255,0.07);
    --cyan:          #00d4ff;
    --cyan-dim:      rgba(0, 212, 255, 0.12);
    --amber:         #ff9533;
    --amber-dim:     rgba(255,149,51,0.12);
    --green:         #2dffb3;
    --red:           #ff4d6d;
    --text-primary:  #e8edf3;
    --text-muted:    #6c7a8d;
    --text-mono:     #a0b4c8;
    --radius:        12px;
    --shadow:        0 8px 40px rgba(0,0,0,0.6);
}

/* ── Base page ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--text-primary);
    font-family: 'Barlow', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border);
}

/* ── Headings ── */
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Keyframe animations ── */
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #2dffb3; }
    50% { opacity: 0.4; box-shadow: 0 0 12px #2dffb3; }
}
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(0,212,255,0.15); }
    50% { box-shadow: 0 0 20px rgba(0,212,255,0.3); }
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-panel); }
::-webkit-scrollbar-thumb { background: var(--cyan); border-radius: 4px; }

/* ── Tabs ── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid var(--border);
    gap: 0;
}
[data-testid="stTabs"] button {
    font-family: 'Barlow', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    transition: color 0.2s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
    background: var(--cyan-dim) !important;
    border-radius: 6px 6px 0 0 !important;
}
[data-testid="stTabs"] button:hover { color: var(--text-primary) !important; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.8rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px var(--cyan-dim) !important;
    outline: none !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--amber)) !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background: var(--cyan) !important;
    box-shadow: 0 0 8px var(--cyan) !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00a8cc 0%, #0066ff 100%) !important;
    color: #fff !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    height: auto !important;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(0, 102, 255, 0.35) !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 168, 204, 0.5) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Info / Alert boxes ── */
[data-testid="stInfo"] {
    background: var(--cyan-dim) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-left: 3px solid var(--cyan) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar number inputs ── */
[data-testid="stSidebar"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Label typography ── */
label[data-testid="stWidgetLabel"] p,
.stNumberInput label p,
.stSlider label p {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-glass) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-family: 'Barlow', sans-serif;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: 'DM Mono', monospace !important; font-size: 1.6rem !important; }

/* ── Plotly chart background ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Columns gap ── */
[data-testid="column"] { padding: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER: Reusable HTML card components
# ─────────────────────────────────────────────
def glass_card(content_html: str, accent: str = "#00d4ff", padding: str = "1.5rem") -> str:
    """Renders an HTML string inside a glassmorphism card."""
    return f"""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-top: 2px solid {accent};
        border-radius: 12px;
        padding: {padding};
        box-shadow: 0 8px 40px rgba(0,0,0,0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    ">
    {content_html}
    </div>"""

def kpi_chip(label: str, value: str, accent: str = "#00d4ff") -> str:
    return f"""
    <div style="
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
        border-bottom:2px solid {accent};
        border-radius:10px; padding:0.9rem 0.5rem; text-align:center;
        transition:transform 0.2s, box-shadow 0.2s;
    ">
        <span style="font-family:'DM Mono',monospace; font-size:1.35rem; font-weight:500; color:{accent};
                     text-shadow:0 0 12px {accent}40;">{value}</span>
        <span style="font-family:'Barlow',sans-serif; font-size:0.7rem; text-transform:uppercase;
                     letter-spacing:0.07em; color:#6c7a8d; margin-top:0.25rem;">{label}</span>
    </div>"""

def section_label(text: str) -> str:
    return f"""<p style="font-family:'Barlow',sans-serif; font-size:0.72rem; font-weight:600;
                          text-transform:uppercase; letter-spacing:0.1em; color:#6c7a8d;
                          margin:0 0 0.5rem 0;">{text}</p>"""

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = joblib.load('best_concrete_model.pkl')
    scaler = joblib.load('concrete_scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# ─────────────────────────────────────────────
# SIDEBAR  ── Branding + Model Stats
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem 0; text-align:center;">
        <div style="font-size:2.8rem;">🧱</div>
        <h1 style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:0.12em;
                   color:#e8edf3; margin:0; line-height:1;">CONCRETIQ</h1>
        <p style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#6c7a8d;
                  text-transform:uppercase; letter-spacing:0.2em; margin:0.3rem 0 0 0;">
            AI Strength Lab · v2.0
        </p>
    </div>
    <hr style="border-color:rgba(255,255,255,0.07); margin:0 0 1.5rem 0;">
    """, unsafe_allow_html=True)

    # ── Model performance KPIs ──
    st.markdown("""
    <p style="font-family:'Barlow',sans-serif; font-size:0.72rem; font-weight:600;
              text-transform:uppercase; letter-spacing:0.1em; color:#6c7a8d; margin-bottom:0.75rem;">
        Model Performance
    </p>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(kpi_chip("R² Score", "0.90", "#00d4ff"), unsafe_allow_html=True)
    with col_b:
        st.markdown(kpi_chip("RMSE", "4.62", "#ff9533"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown(kpi_chip("MAE", "3.18", "#2dffb3"), unsafe_allow_html=True)
    with col_d:
        st.markdown(kpi_chip("Samples", "1,030", "#a78bfa"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status badge ──
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; padding:0.7rem 1rem;
                background:rgba(45,255,179,0.08); border:1px solid rgba(45,255,179,0.2);
                border-radius:8px; margin-bottom:1rem;">
        <span style="width:8px; height:8px; border-radius:50%; background:#2dffb3;
                     box-shadow:0 0 6px #2dffb3; display:inline-block; animation:pulse 2s ease-in-out infinite;"></span>
        <span style="font-family:'DM Mono',monospace; font-size:0.72rem;
                     color:#2dffb3; text-transform:uppercase; letter-spacing:0.08em;">
            Model Online
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Info expander ──
    with st.expander("About the Model"):
        st.markdown("""
        <p style="font-family:'Barlow',sans-serif; font-size:0.82rem; color:#a0b4c8; line-height:1.7;">
        <b style="color:#00d4ff;">XGBoost Regressor</b> trained on the UCI Concrete Compressive Strength dataset.
        Input features are normalized via a <b>Standard Scaler</b> before inference.
        <br><br>
        Strength categories follow <b>IS 456:2000</b> standard grades.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#3d4e60;
              text-align:center; text-transform:uppercase; letter-spacing:0.1em;">
        Built with XGBoost · Plotly · Streamlit
    </p>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# ── Hero header ──
st.markdown("""
<div style="padding: 1rem 0 0.5rem 0; animation: fadeInUp 0.6s ease-out;">
    <h1 style="font-family:'Bebas Neue',sans-serif; font-size:3rem; letter-spacing:0.08em;
               color:#e8edf3; margin:0; line-height:1;">
        COMPRESSIVE STRENGTH
        <span style="background:linear-gradient(90deg,#00d4ff,#7b8cff,#ff9533,#00d4ff);
                     background-size:200% auto;
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                     animation: shimmer 4s linear infinite;">
            PREDICTOR
        </span>
    </h1>
    <p style="font-family:'Barlow',sans-serif; font-size:0.95rem; color:#6c7a8d;
              margin:0.4rem 0 0 0; letter-spacing:0.02em;">
        High-performance concrete mix analysis · Powered by XGBoost ML
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Two-column layout: Inputs | Output ──
col_inputs, col_gap, col_output = st.columns([1.4, 0.05, 1], gap="small")

# ────────────────────────
# LEFT: INPUT PANEL
# ────────────────────────
with col_inputs:

    # Section header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.25rem;">
        <div style="width:3px; height:28px; background:linear-gradient(180deg,#00d4ff,#0044aa);
                    border-radius:2px;"></div>
        <h3 style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem;
                   letter-spacing:0.1em; color:#e8edf3; margin:0;">
            MIX DESIGN PARAMETERS
        </h3>
    </div>
    """, unsafe_allow_html=True)

    tab_binders, tab_aggs = st.tabs(["⚗️  Binders & Water", "🪨  Aggregates & Admixtures"])

    with tab_binders:
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # Binders card
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    border-top:2px solid #00d4ff; border-radius:12px; padding:1.25rem 1.25rem 0.5rem 1.25rem;
                    box-shadow:0 8px 40px rgba(0,0,0,0.5); backdrop-filter:blur(12px);
                    margin-bottom:1rem; animation:fadeInUp 0.5s ease-out;">
            <span style="font-family:'DM Mono',monospace; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.1em; color:#00d4ff;">⚗️ Cementitious Materials</span>
        </div>
        """, unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        with b1:
            cement = st.number_input(
                "Cement (kg/m³)", 100.0, 600.0, 281.0, step=1.0,
                help="Primary binding agent. Higher cement → higher early strength.")
            fly_ash = st.number_input(
                "Fly Ash (kg/m³)", 0.0, 300.0, 54.0, step=1.0,
                help="Pozzolanic material. Improves workability and late-stage strength.")
        with b2:
            slag = st.number_input(
                "Blast Furnace Slag (kg/m³)", 0.0, 400.0, 74.0, step=1.0,
                help="Supplementary cementitious material from steel production.")
            water = st.number_input(
                "Water (kg/m³)", 100.0, 250.0, 181.0, step=1.0,
                help="Controls hydration. Lower w/c ratio = higher strength.")

        # W/C Ratio live indicator
        wc_ratio = round(water / max(cement, 1), 3)
        wc_color = "#2dffb3" if wc_ratio < 0.45 else ("#ff9533" if wc_ratio < 0.6 else "#ff4d6d")
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    border-left:3px solid {wc_color}; border-radius:8px;
                    padding:0.6rem 1rem; margin-top:0.5rem;">
            <span style="font-family:'Barlow',sans-serif; font-size:0.78rem;
                         text-transform:uppercase; letter-spacing:0.06em; color:#6c7a8d;">
                Water/Cement Ratio
            </span>
            <span style="font-family:'DM Mono',monospace; font-size:1rem;
                         font-weight:500; color:{wc_color};">
                {wc_ratio}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with tab_aggs:
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    border-top:2px solid #ff9533; border-radius:12px; padding:1.25rem 1.25rem 0.5rem 1.25rem;
                    box-shadow:0 8px 40px rgba(0,0,0,0.5); backdrop-filter:blur(12px);
                    margin-bottom:1rem; animation:fadeInUp 0.5s ease-out;">
            <span style="font-family:'DM Mono',monospace; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.1em; color:#ff9533;">🪨 Aggregates & Chemical Admixtures</span>
        </div>
        """, unsafe_allow_html=True)

        a1, a2 = st.columns(2)
        with a1:
            coarse_agg = st.number_input(
                "Coarse Aggregate (kg/m³)", 800.0, 1200.0, 973.0, step=1.0,
                help="Crushed stone or gravel. Provides structural bulk.")
            superplasticizer = st.number_input(
                "Superplasticizer (kg/m³)", 0.0, 50.0, 6.0, step=0.1,
                help="Chemical admixture for workability without extra water.")
        with a2:
            fine_agg = st.number_input(
                "Fine Aggregate (kg/m³)", 500.0, 1000.0, 774.0, step=1.0,
                help="Sand. Fills voids between coarse aggregate particles.")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Age slider with visual
        st.markdown(f"""
        <p style="font-family:'Barlow',sans-serif; font-size:0.72rem; font-weight:600;
                  text-transform:uppercase; letter-spacing:0.1em; color:#6c7a8d;
                  margin:0.75rem 0 0.25rem 0;">Curing Age (days)</p>
        """, unsafe_allow_html=True)
        age = st.slider("", 1, 365, 28, label_visibility="collapsed",
                        help="Standard testing at 28 days. Strength grows logarithmically.")

        # Age milestone labels
        milestone = "Early Strength" if age < 14 else ("Standard (28d)" if age <= 56 else "Long-Term Cure")
        m_color = "#ff4d6d" if age < 14 else ("#2dffb3" if age <= 56 else "#a78bfa")
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    border-left:3px solid {m_color}; border-radius:8px; padding:0.5rem 1rem;">
            <span style="font-family:'Barlow',sans-serif; font-size:0.78rem;
                         text-transform:uppercase; letter-spacing:0.06em; color:#6c7a8d;">
                Stage
            </span>
            <span style="font-family:'DM Mono',monospace; font-size:0.9rem;
                         font-weight:500; color:{m_color};">
                {age}d · {milestone}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # ── CTA Button ──
    predict_btn = st.button("⚡  CALCULATE COMPRESSIVE STRENGTH", type="primary")


# ────────────────────────
# RIGHT: OUTPUT PANEL
# ────────────────────────
with col_output:

    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.25rem;">
        <div style="width:3px; height:28px; background:linear-gradient(180deg,#ff9533,#ff4d6d);
                    border-radius:2px;"></div>
        <h3 style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem;
                   letter-spacing:0.1em; color:#e8edf3; margin:0;">
            PREDICTION RESULTS
        </h3>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        # ── Simulate loading ──
        with st.spinner("Running inference..."):
            time.sleep(0.6)

        # ── Bundle inputs & predict ──
        input_values = [[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]]
        input_data   = pd.DataFrame(input_values, columns=scaler.feature_names_in_)
        scaled_input = scaler.transform(input_data)
        prediction   = model.predict(scaled_input)[0]

        # ── Classify strength grade ──
        if prediction < 20:
            grade, grade_color, grade_label = "M15", "#ff4d6d", "Low Strength"
        elif prediction < 30:
            grade, grade_color, grade_label = "M25", "#ff9533", "Normal Strength"
        elif prediction < 45:
            grade, grade_color, grade_label = "M40", "#00d4ff", "High Strength"
        else:
            grade, grade_color, grade_label = "M55+", "#2dffb3", "Ultra-High Strength"

        # ── Primary gauge ──
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={
                'suffix': " MPa",
                'font': {'size': 44, 'color': grade_color, 'family': 'DM Mono'}
            },
            title={
                'text': f"Compressive Strength<br><span style='font-size:0.8em;color:#6c7a8d'>{grade} · {grade_label}</span>",
                'font': {'size': 15, 'color': '#a0b4c8', 'family': 'Barlow'}
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#3d4e60",
                    'tickfont': {'color': '#3d4e60', 'size': 10},
                    'nticks': 11
                },
                'bar': {'color': grade_color, 'thickness': 0.22},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20],  'color': "rgba(255,77,109,0.12)"},
                    {'range': [20, 30], 'color': "rgba(255,149,51,0.12)"},
                    {'range': [30, 45], 'color': "rgba(0,212,255,0.10)"},
                    {'range': [45,100], 'color': "rgba(45,255,179,0.10)"}
                ],
                'threshold': {
                    'line': {'color': grade_color, 'width': 3},
                    'thickness': 0.8,
                    'value': prediction
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#a0b4c8"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Prediction KPI row ──
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(kpi_chip("Pred. MPa", f"{prediction:.1f}", grade_color), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_chip("Grade", grade, "#a78bfa"), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_chip("Age", f"{age}d", "#ff9533"), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Ingredient contribution bar chart ──
        ingredients = ["Cement", "Slag", "Fly Ash", "Water", "Superplast.", "Coarse Agg.", "Fine Agg."]
        values      = [cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg]
        bar_colors  = ["#00d4ff","#7b8cff","#a78bfa","#ff4d6d","#2dffb3","#ff9533","#ffd166"]

        fig_bar = go.Figure(go.Bar(
            x=values,
            y=ingredients,
            orientation='h',
            marker=dict(
                color=bar_colors,
                line=dict(width=0)
            ),
            text=[f"{v:.1f}" for v in values],
            textposition='outside',
            textfont=dict(family='DM Mono', size=10, color='#a0b4c8'),
        ))
        fig_bar.update_layout(
            title=dict(
                text="Mix Composition (kg/m³)",
                font=dict(family='Barlow', size=12, color='#6c7a8d'),
                x=0
            ),
            height=250,
            margin=dict(l=10, r=60, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(family='DM Mono', size=9, color='#3d4e60'),
                zeroline=False
            ),
            yaxis=dict(
                tickfont=dict(family='Barlow', size=11, color='#a0b4c8'),
                gridcolor="rgba(0,0,0,0)"
            ),
            bargap=0.35,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Strength development curve ──
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        curve_ages = [1, 3, 7, 14, 28, 56, 90, 180, 365]
        curve_strengths = []
        for a in curve_ages:
            inp = pd.DataFrame([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, a]],
                               columns=scaler.feature_names_in_)
            curve_strengths.append(model.predict(scaler.transform(inp))[0])

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=curve_ages, y=curve_strengths, mode='lines+markers',
            line=dict(color='#00d4ff', width=2.5, shape='spline'),
            marker=dict(size=7, color='#080c10', line=dict(color='#00d4ff', width=2)),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.06)',
            name='Predicted Strength'
        ))
        # Highlight current age
        fig_curve.add_trace(go.Scatter(
            x=[age], y=[prediction], mode='markers',
            marker=dict(size=14, color=grade_color, symbol='diamond',
                        line=dict(color='#fff', width=1.5)),
            name=f'Current ({age}d)',
            showlegend=True
        ))
        fig_curve.update_layout(
            title=dict(text="Strength Development Curve", font=dict(family='Barlow', size=12, color='#6c7a8d'), x=0),
            height=220, margin=dict(l=10, r=20, t=35, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=dict(text="Age (days)", font=dict(family='Barlow', size=10, color='#6c7a8d')),
                       type="log", showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       tickfont=dict(family='DM Mono', size=9, color='#3d4e60')),
            yaxis=dict(title=dict(text="MPa", font=dict(family='Barlow', size=10, color='#6c7a8d')),
                       showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       tickfont=dict(family='DM Mono', size=9, color='#3d4e60')),
            legend=dict(font=dict(family='Barlow', size=10, color='#6c7a8d'),
                        bgcolor='rgba(0,0,0,0)', x=0.02, y=0.98),
            font_color="#a0b4c8"
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # ── Info note ──
        st.markdown("""
        <div style="display:flex; align-items:flex-start; gap:0.7rem; padding:0.8rem 1rem;
                    background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.15);
                    border-left:3px solid #00d4ff; border-radius:10px; animation:fadeInUp 0.6s ease-out;">
            <span style="font-size:1.1rem; flex-shrink:0;">💡</span>
            <span style="font-family:'Barlow',sans-serif; font-size:0.8rem; color:#a0b4c8; line-height:1.5;">
                Prediction based on XGBoost model (R²=0.90). Results are indicative —
                lab testing is required for structural design.
            </span>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Placeholder state ──
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            height:400px; text-align:center;
            background:linear-gradient(145deg, rgba(0,212,255,0.03), rgba(255,149,51,0.03));
            border:1px dashed rgba(255,255,255,0.1);
            border-radius:16px; gap:1rem;
            animation: glow 3s ease-in-out infinite;
        ">
            <div style="font-size:3.5rem; opacity:0.4; animation:float 3s ease-in-out infinite;">🧱</div>
            <p style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem;
                      letter-spacing:0.12em; color:rgba(255,255,255,0.18); margin:0;">
                AWAITING PARAMETERS
            </p>
            <p style="font-family:'Barlow',sans-serif; font-size:0.82rem;
                      color:#4d5e70; margin:0; max-width:280px; line-height:1.6;">
                Configure your mix design on the left, then click <strong style="color:#00d4ff;">Calculate</strong> to run the AI analysis.
            </p>
            <div style="margin-top:0.5rem; display:flex; gap:0.5rem;">
                <div style="width:6px; height:6px; border-radius:50%; background:#00d4ff; opacity:0.4; animation:pulse 2s ease-in-out infinite;"></div>
                <div style="width:6px; height:6px; border-radius:50%; background:#7b8cff; opacity:0.4; animation:pulse 2s ease-in-out 0.3s infinite;"></div>
                <div style="width:6px; height:6px; border-radius:50%; background:#ff9533; opacity:0.4; animation:pulse 2s ease-in-out 0.6s infinite;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            padding:0.5rem 0 1.5rem 0; flex-wrap:wrap; gap:0.5rem;">
    <p style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#3d4e60;
              text-transform:uppercase; letter-spacing:0.12em; margin:0;">
        CONCRETIQ · AI Strength Lab · Built with Streamlit & XGBoost
    </p>
    <p style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#3d4e60;
              text-transform:uppercase; letter-spacing:0.12em; margin:0;">
        Dataset: UCI Concrete Compressive Strength · I-Cheng Yeh (1998)
    </p>
</div>
""", unsafe_allow_html=True)
