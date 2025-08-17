import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide", initial_sidebar_state="collapsed")

# --- Load Model ---
# Load the trained model and the expected feature list
try:
    model = joblib.load('models/final_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'final_model.pkl' is in the 'models/' directory.")
    st.stop()

# This is the exact list of columns the model was trained on.
# The order must be preserved.
EXPECTED_FEATURES = ['thalach', 'oldpeak', 'thal_7.0', 'cp_4.0', 'age', 'chol',
                     'trestbps', 'exang_1.0', 'slope_2.0', 'sex_1.0', 'ca_1.0', 'cp_3.0']

# --- Page Title and Description ---
st.title('🩺 Heart Disease Prediction App')
st.markdown("""
This app uses a machine learning model to predict the risk of heart disease. 
Enter the patient's details below to get a prediction.
""")
st.markdown("---")

# --- Input Fields ---
# Organize inputs into two columns for a cleaner layout
col1, col2 = st.columns(2)

# Column 1: Primary Numerical Inputs
with col1:
    st.header("Primary Vitals")
    age = st.slider("Age", 20, 80, 50)
    trestbps = st.slider("Resting Blood Pressure (trestbps in mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (chol in mg/dl)", 100, 600, 240)
    thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 220, 150)
    oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.2, 1.0)

# Column 2: Categorical and Other Inputs
with col2:
    st.header("Medical History & Demographics")
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: {0: "Female", 1: "Male"}[x])
    cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4], format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}[x])
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [1, 2, 3], format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])


# --- Data Processing and Prediction ---
# Function to process inputs and create the feature vector for the model
def process_inputs():
    # Initialize a dictionary for the encoded data with all expected features set to 0.
    encoded_data = {feat: 0 for feat in EXPECTED_FEATURES}
    
    # Populate the numerical features directly from user input.
    encoded_data['age'] = age
    encoded_data['trestbps'] = trestbps
    encoded_data['chol'] = chol
    encoded_data['thalach'] = thalach
    encoded_data['oldpeak'] = oldpeak

    # Handle one-hot encoded features based on the final selected list
    if sex == 1: encoded_data['sex_1.0'] = 1
    if cp == 3: encoded_data['cp_3.0'] = 1
    if cp == 4: encoded_data['cp_4.0'] = 1
    if exang == 1: encoded_data['exang_1.0'] = 1
    if slope == 2: encoded_data['slope_2.0'] = 1
    if ca == 1: encoded_data['ca_1.0'] = 1
    if thal == 7: encoded_data['thal_7.0'] = 1
    
    # Create a DataFrame in the correct order
    features = pd.DataFrame([encoded_data])
    return features[EXPECTED_FEATURES]

input_df = process_inputs()

st.markdown("---")

# --- Prediction Button and Output ---
if st.button("Predict Heart Disease Risk", type="primary"):
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display results in an expander for a clean look
    with st.expander("Prediction Result", expanded=True):
        if prediction[0] == 1:
            st.error('Prediction: High Risk of Heart Disease', icon="⚠️")
        else:
            st.success('Prediction: Low Risk of Heart Disease', icon="✅")

        st.subheader("Prediction Probability")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric(label="Probability of No Heart Disease", value=f"{prediction_proba[0][0]*100:.2f}%")
        with prob_col2:
            st.metric(label="Probability of Heart Disease", value=f"{prediction_proba[0][1]*100:.2f}%")

# --- Disclaimer ---
st.info("Disclaimer: This prediction is based on a machine learning model and is not a substitute for professional medical advice. Please consult a doctor for any health concerns.", icon="ℹ️")
