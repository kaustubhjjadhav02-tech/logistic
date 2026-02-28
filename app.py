import pandas as pd
import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("logistic_model.pkl")

# App title
st.title("Diabetes Prediction using Logistic Regression")
st.write("Enter patient details to predict diabetes outcome")

# User inputs (EXACTLY as per dataset)
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", format="%.2f")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", format="%.3f")
Age = st.number_input("Age", min_value=0)

# Prediction button
if st.button("Predict Diabetes"):
    
    # Combine inputs into array (ORDER IS CRITICAL)
    input_data = pd.DataFrame({
    "Pregnancies": [Pregnancies],
    "Glucose": [Glucose],
    "BloodPressure": [BloodPressure],
    "SkinThickness": [SkinThickness],
    "Insulin": [Insulin],
    "BMI": [BMI],
    "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
    "Age": [Age]
})
    
    # Predict
    prediction = model.predict(input_data.values)
    
    # Output
    if prediction[0] == 1:
        st.error("⚠️ Prediction: Diabetic")
    else:

        st.success("✅ Prediction: Not Diabetic")
