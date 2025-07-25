
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Football Header Impact Classifier", layout="centered")

st.title("⚽ Football Header Impact Classifier")

st.write("""
This tool predicts whether a football header impact is **High** or **Low**  
based on 5 binary questions derived from video analysis.
""")

# === Target variable selection ===
target_choice = st.radio("Choose target label:", ["PLA", "PAA"], horizontal=True)

# === Model selection ===
model_choice = st.selectbox("Choose your model:", ["RandomForest", "SVM", "LogisticRegression", "XGBoost"])

# === Load model and threshold ===
model_filename = f"{model_choice.lower()}_{target_choice}.pkl"
threshold_filename = f"threshold_{model_choice.lower()}_{target_choice}.pkl"

try:
    model = joblib.load(model_filename)
    threshold = joblib.load(threshold_filename)
except FileNotFoundError:
    st.error("Missing model or threshold file. Please train and export them first.")
    st.stop()

# === Optional threshold adjustment ===
threshold = st.slider("Prediction threshold (probability > threshold → High)", 0.1, 0.9, float(threshold), step=0.01)

st.markdown("---")

st.subheader("📋 Impact Questions (Yes / No)")

def yes_no(question: str, key: str):
    return st.radio(question, ["No", "Yes"], key=key) == "Yes"

q1 = yes_no("1️⃣ Does the ball travel more than 35 meters BEFORE the header?", "q1")
q2 = yes_no("2️⃣ Is the ball velocity high BEFORE the header?", "q2")
q3 = yes_no("3️⃣ Is there a change in direction AFTER the header?", "q3")
q4 = yes_no("4️⃣ Does the ball travel more than 10 meters AFTER the header?", "q4")
q5 = yes_no("5️⃣ Is the header flicked or glanced?", "q5")

# === Prepare input ===
expected_columns = [
    "1_Dist_Bef_Head",
    "2_Fli_Bef_Head",
    "3_Fli_Aft_Head",
    "4_Dist_Aft_Head",
    "5_Head_Type"
]

input_data = pd.DataFrame([{
    "1_Dist_Bef_Head": int(q1),
    "2_Fli_Bef_Head": int(q2),
    "3_Fli_Aft_Head": int(q3),
    "4_Dist_Aft_Head": int(q4),
    "5_Head_Type": int(q5)
}])[expected_columns]  # reorder right here


# === Prediction ===
proba = model.predict_proba(input_data)[0][1]
prediction = "High" if proba > threshold else "Low"

# === Display result ===
st.subheader("📊 Prediction Result")

if prediction == "High":
    st.success(f"🟥 High impact predicted (probability: {proba:.2f})")
else:
    st.info(f"🟩 Low impact predicted (probability: {proba:.2f})")

# === Explanation toggle ===
with st.expander("ℹ️ Explanation of each question"):
    st.markdown("""
- **Q1**: Did the ball travel more than half the field (~35m) before being headed?
- **Q2**: Was the ball moving fast before the header? (e.g., driven vs floated)
- **Q3**: Did the direction of the ball clearly change after the header?
- **Q4**: Did the ball travel more than 10 meters after the header?
- **Q5**: Was it a flicked/glanced header (vs. a solid direction-changing one)?
    """)
