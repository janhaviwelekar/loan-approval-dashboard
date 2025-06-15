# loan_dashboard.py

import streamlit as st
import pandas as pd
import joblib

# Load model and expected input columns
model = joblib.load("loan_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏦 Loan Approval Predictor")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create input DataFrame
input_df = pd.DataFrame([{
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}])

# One-hot encode input and align with training columns
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0][prediction]
        label = 'Approved' if prediction == 1 else 'Not Approved'
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {proba:.2f}")
        st.markdown("[🔎 View SHAP Explainability Dashboard](https://your-explainer-url.onrender.com)", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
