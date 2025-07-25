
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===== Title =====
st.title("⚽ Football Header Impact Classifier")

st.write("""
This tool predicts whether a football header impact is **High** or **Low**
based on 5 biomechanical features extracted from match footage.
""")

# === Target variable selection ===
target_choice = st.radio("Choose target label:", ["PLA", "PAA"], horizontal=True)

# === Model selection ===
model_choice = st.selectbox("Choose your model:", ["RandomForest", "SVM", "LogisticRegression", "XGBoost"])

# === Load selected model, scaler and threshold ===
model_filename = f"{model_choice.lower()}_{target_choice}.pkl"
scaler_filename = f"scaler_{target_choice}.pkl"
threshold_filename = f"threshold_{model_choice.lower()}_{target_choice}.pkl"

try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    threshold = joblib.load(threshold_filename)
except FileNotFoundError:
    st.error(f"Missing model or scaler file: {model_filename}, {scaler_filename}, or {threshold_filename}")
    st.stop()

# === Optional: Adjust threshold manually ===
threshold = st.slider("Prediction threshold", 0.1, 0.9, float(threshold), step=0.01)

st.markdown("---")

# === Input features ===
st.subheader("📥 Enter Impact Features")

dist_bef = st.slider("1️⃣ Distance before impact (meters)", 0.0, 10.0, 2.0, step=0.1)
flight_bef = st.slider("2️⃣ Ball flight time before header (seconds)", 0.0, 2.0, 0.5, step=0.05)
flight_aft = st.slider("3️⃣ Ball flight time after header (seconds)", 0.0, 2.0, 0.5, step=0.05)
dist_aft = st.slider("4️⃣ Distance after impact (meters)", 0.0, 10.0, 2.0, step=0.1)
head_type = st.selectbox("5️⃣ Header type", ['Frontal', 'Lateral', 'Other'])

# Encode header type
head_type_encoded = {'Frontal': 0, 'Lateral': 1, 'Other': 2}[head_type]

# === Build input DataFrame ===
input_data = pd.DataFrame([{
    "1_Dist_Bef_Head": dist_bef,
    "2_Fli_Bef_Head": flight_bef,
    "3_Fli_Aft_Head": flight_aft,
    "4_Dist_Aft_Head": dist_aft,
    "5_Head_Type": head_type_encoded
}])

# === Feature scaling ===
input_scaled = scaler.transform(input_data)

# === Predict ===
proba = model.predict_proba(input_scaled)[0][1]
pred = "High" if proba > threshold else "Low"

# === Display result ===
st.subheader("📊 Prediction Result")

if pred == "High":
    st.success(f"🟥 High impact predicted (probability: {proba:.2f})")
else:
    st.info(f"🟩 Low impact predicted (probability: {proba:.2f})")

# === Feature info ===
with st.expander("ℹ️ About the Features"):
    st.markdown("""
    - **1_Dist_Bef_Head**: Distance covered before the header
    - **2_Fli_Bef_Head**: Ball flight duration before the header
    - **3_Fli_Aft_Head**: Ball flight duration after the header
    - **4_Dist_Aft_Head**: Distance covered after the header
    - **5_Head_Type**: Type of header (frontal, lateral, other)
    """)
