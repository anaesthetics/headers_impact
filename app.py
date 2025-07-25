
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
q5 = yes_no("5️⃣ Is the header a flick/glance? (Yes = flick/glance, No = solid header)", "q5")

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
with st.expander("ℹ️ Detailed Explanation of each question"):
    st.markdown("""
    ## High/Low Force Classification - Impact Characteristics
    
    ### BEFORE HEADER:
    
    **Q1 - Distance Ball Travels BEFORE Header**
    - **Question**: Does the ball travel MORE than 35 metres BEFORE it is headed?
    - **Guide**: Use half the pitch as your guide (does the ball travel more than half the length of the pitch?)
    - **Note**: The distance includes any bounces and/or deflections (e.g., if the ball travels ~20/25 metres in the air before bouncing a further ~10/15 metres without interference before being headed, it has travelled more than 35 metres)
    
    **Q2 - Flight Course of the Ball BEFORE Header**
    - **Question**: Is the velocity (directed speed and acceleration) of the ball high AFTER it has been kicked/thrown?
    - **Consider**: Factors that affect acceleration/speed after the ball has been kicked or headed before the header itself (e.g., does the ball bounce several times which may reduce speed/acceleration? Is there a deflection that changes the speed/acceleration?)
    - **Note**: Consider the type of ball delivery - speed/acceleration following a lofted/floated pass is likely different than following a driven pass/cross
    
    ### AFTER HEADER:
    
    **Q3 - Flight Course of the Ball AFTER Header**
    - **Question**: Is there a change in the motion/direction that the ball is travelling in AFTER it has been headed?
    - **Consider**: Does the direction or motion change after the header? (e.g., ball travels back towards goalkeeper after a header from a goal kick vs. little-to-no change in direction for a flick/glance)
    
    **Q4 - Distance the Ball Travels AFTER Header**
    - **Question**: Does the ball travel MORE than 10 metres AFTER it has been headed?
    - **Guide**: Use the distance between the goal and penalty spot (12 yards = ~10 metres) as your guide
    - **Note**: If the ball is blocked/hits an obstacle, imagine it continuing in the same trajectory to determine if it would have travelled more than 10 metres
    
    **Q5 - Type of Header**
    - **Question**: Is the header flicked on or glanced?
    - **Purpose**: Determine whether the motion/direction of the ball remains unaffected after the header
    - **Note**: This overlaps with Q3 - we want to know if the ball's direction changes, which can be determined by the type of header performed
    """)
