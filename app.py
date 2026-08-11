import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_default_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Loan Default Prediction")
st.write("Enter the applicant details to predict loan default.")

st.divider()

# Input fields
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150000
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

employment_years = st.number_input(
    "Employment Years",
    min_value=0,
    max_value=50,
    value=5
)

existing_loans = st.number_input(
    "Existing Loans",
    min_value=0,
    max_value=20,
    value=1
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=6,
    max_value=120,
    value=36
)

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=10.0
)

debt_to_income = st.number_input(
    "Debt to Income Ratio (%)",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

st.divider()

# Prediction button
if st.button("🔍 Predict Loan Default", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "EmploymentYears": [employment_years],
        "ExistingLoans": [existing_loans],
        "LoanTerm": [loan_term],
        "InterestRate": [interest_rate],
        "DebtToIncome": [debt_to_income]
    })

    # Use the same features used during training
    features = model.feature_names_in_

    input_data = input_data[features]

    # Prediction
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Loan Default Risk: HIGH")
        st.write("The applicant is predicted to default on the loan.")
    else:
        st.success("✅ Loan Default Risk: LOW")
        st.write("The applicant is predicted not to default on the loan.")
