import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Credit Wise", page_icon="💰")
st.title("💰 Credit Wise: Loan Approval Predictor")

@st.cache_data
def load_and_train():
    # Load your specific CSV
    df = pd.read_csv("loan_approval_data.csv")
    
    # These are the exact columns found in your notebook
    features = ['Applicant_Income', 'Coapplicant_Income', 'Credit_Score', 'Loan_Amount']
    target = 'Loan_Approved'
    
    X = df[features]
    # Map your 'Yes'/'No' to 1/0
    y = df[target].map({'Yes': 1, 'No': 0})
    
    # Handle missing values as you did in your notebook
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Best model according to your project: Naive Bayes
    model = GaussianNB()
    model.fit(X_scaled, y)
    
    return model, scaler

try:
    model, scaler = load_and_train()

    st.sidebar.header("Applicant Details")
    app_inc = st.sidebar.number_input("Applicant Income", value=5000)
    co_inc = st.sidebar.number_input("Coapplicant Income", value=0)
    score = st.sidebar.slider("Credit Score", 300, 850, 700)
    loan_amt = st.sidebar.number_input("Loan Amount", value=20000)

    if st.button("Predict Approval"):
        user_input = np.array([[app_inc, co_inc, score, loan_amt]])
        user_scaled = scaler.transform(user_input)
        prediction = model.predict(user_scaled)
        
        if prediction[0] == 1:
            st.success("🎉 Loan Approved!")
        else:
            st.error("❌ Loan Denied.")

except Exception as e:
    st.error(f"Please ensure 'loan_approval_data.csv' is uploaded to GitHub. Error: {e}")
