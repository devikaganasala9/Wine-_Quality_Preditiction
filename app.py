import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# -----------------------------
# Background Image Function
# -----------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Make main container transparent */
        section[data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.6);
        }}

        div[data-testid="stAppViewContainer"] > .main {{
            background-color: rgba(255, 255, 255, 0.75);
            padding: 20px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Add Background Image
# -----------------------------
# Make sure "wine5.jpg" is in the SAME folder as this .py file
add_bg_from_local("wine5.jpg")

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("finalized_RFmodel.sav", "rb"))
    scaler = pickle.load(open("scaler_model.sav", "rb"))
    return model, scaler

RF_model, scaler = load_model()

# -----------------------------
# App Title
# -----------------------------
st.title("🍷 Wine Quality Prediction App")
st.markdown(
    """
    This application predicts the **quality of red wine** using a  
    **Random Forest Machine Learning model**.
    """
)

# -----------------------------
# Sidebar – User Input
# -----------------------------
st.sidebar.header("🔧 Input Wine Features")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.70)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 1.0, 0.00)
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 2.0)
chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.08)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
density = st.sidebar.number_input("Density", 0.9900, 1.0100, 0.9968, format="%.4f")
pH = st.sidebar.number_input("pH", 2.0, 4.5, 3.31)
sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.66)
alcohol = st.sidebar.number_input("Alcohol", 5.0, 15.0, 10.5)

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# -----------------------------
# Log Transform (Same as Training)
# -----------------------------
input_data["residual sugar"] = np.log1p(input_data["residual sugar"])
input_data["chlorides"] = np.log1p(input_data["chlorides"])
input_data["free sulfur dioxide"] = np.log1p(input_data["free sulfur dioxide"])
input_data["total sulfur dioxide"] = np.log1p(input_data["total sulfur dioxide"])
input_data["sulphates"] = np.log1p(input_data["sulphates"])

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Wine Quality"):
    input_scaled = scaler.transform(input_data)
    prediction = RF_model.predict(input_scaled)

    st.success(f"🍷 **Predicted Wine Quality:** {int(prediction[0])}")

    if prediction[0] >= 7:
        st.balloons()
        st.info("✅ High quality wine!")
    elif prediction[0] >= 5:
        st.warning("⚠️ Average quality wine")
    else:
        st.error("❌ Low quality wine")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🍷 *Built using Machine Learning & Streamlit*")
