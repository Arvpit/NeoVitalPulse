import streamlit as st
import sys
import os
import io
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =====================================================
# PROJECT PATH SETUP
# =====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.predict import predict_heart_disease

st.set_page_config(page_title="NeoVitalPulse", layout="centered")

# =====================================================
# MODERN CSS DESIGN
# =====================================================
st.markdown("""
<style>

/* ================================
   BACKGROUND
================================ */
.stApp {
    background: linear-gradient(to right, #e3f2fd, #bbdefb);
}

/* ================================
   MAIN & SECTION TITLES
================================ */
/* Main Title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #0d47a1;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #1565c0;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #0d47a1;
    margin-top: 25px;
    margin-bottom: 15px;
}

.sub-header {
    font-size: 20px;
    font-weight: 700;
    color: #0d47a1;
    margin-top: 20px;
    margin-bottom: 10px;
}

/* ================================
   INPUT LABELS (BLUE)
================================ */
label {
    color: #0d47a1 !important;
    font-weight: 600 !important;
}

/* ================================
   INPUT VALUE TEXT (WHITE)
================================ */
div[data-baseweb="input"] input {
    color: white !important;
    font-weight: 500;
}

/* ================================
   INPUT BOX BACKGROUND (DARK BLUE)
================================ */
div[data-baseweb="input"] {
    background-color: #1565c0 !important;
    border-radius: 8px;
}

/* ================================
   SELECTBOX TEXT (WHITE)
================================ */
div[data-baseweb="select"] span {
    color: white !important;
    font-weight: 500;
}

/* SELECTBOX BACKGROUND */
div[data-baseweb="select"] > div {
    background-color: #1565c0 !important;
    border-radius: 8px;
}

/* ================================
   SLIDER VALUE TEXT (WHITE)
================================ */
.stSlider span {
    color: white !important;
}

/* ================================
   FACTOR LIST TEXT (WHITE)
================================ */
.factor-item {
    color: white;
    font-size: 16px;
    margin-left: 10px;
    margin-bottom: 4px;
}

/* ================================
   BUTTON STYLE
================================ */
div.stButton > button {
    background: linear-gradient(90deg, #1565c0, #0d47a1);
    color: white !important;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    border: none;
    font-weight: 600;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0d47a1, #1565c0);
}

/* ================================
   PROGRESS BAR COLOR
================================ */
div[data-testid="stProgress"] > div > div > div > div {
    background-color: #0d47a1;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER SECTION
# =====================================================
st.markdown('<div class="main-title">🫀 NeoVitalPulse</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Heart Disease Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# =====================================================
# PATIENT INPUT SECTION
# =====================================================
st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol Level", 100, 600, 200)
    max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

with col2:
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type",
        ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

fasting_bs = st.selectbox("Fasting Blood Sugar",
    ["Lower than 120 mg/ml", "Greater than 120 mg/ml"])

rest_ecg = st.selectbox("Rest ECG",
    ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])

slope = st.selectbox("Slope of ST Segment",
    ["Upsloping", "Flat", "Downsloping"])

vessels = st.selectbox("Number of Major Vessels Colored",
    ["Zero", "One", "Two", "Three", "Four"])

thal = st.selectbox("Thalassemia",
    ["Normal", "Fixed Defect", "Reversable Defect", "No"])

st.markdown("---")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def clean_feature(name):
    return name.replace("_", " ").title()

def generate_pdf(input_data, result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>NeoVitalPulse Clinical Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Patient Information:</b>", styles['Heading2']))
    for key, value in input_data.items():
        elements.append(Paragraph(f"{key}: {value}", styles['Normal']))

    elements.append(Spacer(1, 0.3 * inch))

    probability = result["probability"] * 100
    risk = result["risk_level"]

    elements.append(Paragraph("<b>Prediction Result:</b>", styles['Heading2']))
    elements.append(Paragraph(f"Risk Probability: {probability:.2f}%", styles['Normal']))
    elements.append(Paragraph(f"Risk Category: {risk}", styles['Normal']))

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Top Risk Factors:</b>", styles['Heading2']))
    for item in result["top_risk_factors"]:
        elements.append(Paragraph(
            f"- {item['feature']} (Impact: {item['impact']:.3f})",
            styles['Normal']
        ))

    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("<b>Protective Factors:</b>", styles['Heading2']))
    for item in result["top_protective_factors"]:
        elements.append(Paragraph(
            f"- {item['feature']} (Impact: {item['impact']:.3f})",
            styles['Normal']
        ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# PREDICTION BUTTON
# =====================================================
predict_button = st.button("🚀 Predict Heart Disease Risk", use_container_width=True)

# =====================================================
# PREDICTION LOGIC
# =====================================================
if predict_button:

    input_data = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain,
        "resting_blood_pressure": resting_bp,
        "cholestoral": cholesterol,
        "fasting_blood_sugar": fasting_bs,
        "rest_ecg": rest_ecg,
        "Max_heart_rate": max_hr,
        "exercise_induced_angina": exercise_angina,
        "oldpeak": oldpeak,
        "slope": slope,
        "vessels_colored_by_flourosopy": vessels,
        "thalassemia": thal
    }

    result = predict_heart_disease(input_data)
    probability = result["probability"] * 100
    risk = result["risk_level"]

    # ------------------ GAUGE ------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        title={'text': "Heart Disease Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig)

    # ------------------ RISK CARD ------------------
    if risk == "Low Risk":
        color = "#2e7d32"
    elif risk == "Moderate Risk":
        color = "#f57c00"
    else:
        color = "#c62828"

    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:white;
            border-left:6px solid {color};
            font-weight:600;
            font-size:18px;
            color:#0d47a1;">
            Risk Probability: {probability:.2f}% <br>
            Risk Category: <span style="color:{color}">{risk}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(probability / 100)

    st.markdown("---")

    # ------------------ SHAP BAR ------------------
    st.markdown('<div class="section-title">Feature Contribution Analysis</div>', unsafe_allow_html=True)

    contrib_data = result["all_contributions"][:8]
    features = [clean_feature(item["feature"]) for item in contrib_data]
    values = [item["impact"] for item in contrib_data]
    colors = ["red" if v > 0 else "green" for v in values]

    bar_fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors
    ))

    bar_fig.update_layout(height=400)
    st.plotly_chart(bar_fig)

    # ------------------ TOP FACTORS ------------------
    st.markdown("<div style='color:blue'>🔺 Factors Increasing Risk</div>", unsafe_allow_html=True)
    for item in result["top_risk_factors"]:
        st.write(f" - {clean_feature(item['feature'])} (Impact: {item['impact']:.3f})")

    st.markdown("<div style='color:blue'>🔻 Protective Factors</div>", unsafe_allow_html=True)
    for item in result["top_protective_factors"]:
        st.write(f" - {clean_feature(item['feature'])} (Impact: {item['impact']:.3f})")

    # ------------------ PDF DOWNLOAD ------------------
    pdf_file = generate_pdf(input_data, result)

    st.download_button(
        label="📄 Download Clinical Report (PDF)",
        data=pdf_file,
        file_name="NeoVitalPulse_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )