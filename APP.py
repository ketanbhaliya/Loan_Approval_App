import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Credit Wise", page_icon="💰")

st.title("💰 Credit Wise: Loan Approval Predictor")
st.markdown("Enter the applicant details below to predict loan eligibility.")

# 1. Load Data
@st.cache_data
def load_and_prep():
    df = pd.read_csv("loan_approval_data.csv")
    
    # Feature selection based on your notebook
    features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    X = df[features].copy()
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Simple Imputing
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Train the NB model
    model = GaussianNB()
    model.fit(X_scaled, y)
    
    return model, scaler, imputer

try:
    model, scaler, imputer = load_and_prep()

    # 2. Sidebar Inputs
    st.sidebar.header("User Input Features")
    app_inc = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
    co_inc = st.sidebar.number_input("Co-applicant Income", min_value=0, value=0)
    loan_amt = st.sidebar.number_input("Loan Amount", min_value=0, value=150)
    term = st.sidebar.selectbox("Term (Days)", [360, 180, 120, 84, 60])
    hist = st.sidebar.radio("Credit History (1.0 = Good, 0.0 = Bad)", [1.0, 0.0])

    # 3. Prediction Logic
    if st.button("Predict Approval Status"):
        user_data = np.array([[app_inc, co_inc, loan_amt, term, hist]])
        user_scaled = scaler.transform(user_data)
        prediction = model.predict(user_scaled)
        
        if prediction[0] == 1:
            st.success("✅ Prediction: **Loan Approved**")
        else:
            st.error("❌ Prediction: **Loan Rejected**")

except FileNotFoundError:
    st.error("Error: 'loan_approval_data.csv' not found. Please ensure it is in your GitHub repository.")
