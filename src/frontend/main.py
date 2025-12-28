import requests
import streamlit as st
from datetime import date

# Request processing function
def predict():
    # Calculate monthly installment
    r = st.session_state.int_rate / 1200
    n = int(st.session_state.term.split()[0])
    st.session_state.installment = st.session_state.loan_amnt * (r) * ((1 + r) ** n) / ((1 + r) ** n - 1)
    
    # Update values
    issue_d = st.session_state.issue_d.strftime("%b-%Y")
    earliest_cr_line = st.session_state.earliest_cr_line.strftime("%b-%Y")
    application_type = "_".join(st.session_state.application_type.upper().split())
    purpose = "_".join(st.session_state.purpose.lower().split())
    home_ownership = st.session_state.home_ownership.upper()

    # Input format & order for the model
    model_input = {
        "loan_amnt": st.session_state.loan_amnt,
        "term": st.session_state.term,
        "int_rate": st.session_state.int_rate,
        "installment": st.session_state.installment,
        "grade": st.session_state.grade,
        "sub_grade": st.session_state.sub_grade,
        "emp_length": st.session_state.emp_length,
        "home_ownership": home_ownership,
        "annual_inc": st.session_state.annual_inc,
        "verification_status": st.session_state.verification_status,
        "issue_d": issue_d,
        "purpose": purpose,
        "dti": st.session_state.dti,
        "earliest_cr_line": earliest_cr_line,
        "open_acc": st.session_state.open_acc,
        "pub_rec": st.session_state.pub_rec,
        "revol_bal": st.session_state.revol_bal,
        "revol_util": st.session_state.revol_util,
        "total_acc": st.session_state.total_acc,
        "initial_list_status": st.session_state.initial_list_status,
        "application_type": application_type,
        "mort_acc": st.session_state.mort_acc,
        "pub_rec_bankruptcies": st.session_state.pub_rec_bankruptcies,
        "address": st.session_state.address,
    }

    server_url = "http://fastapi-app:8000/api/v1/predict"
    try:
        response = requests.post(
            url=server_url,
            json=model_input
        )
        if response.text.strip('"').startswith("Not"):
            st.markdown(f"<p style='text-align: center; color: green; font-size: 20px;'>Prediction: {response.text.strip('"')}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; color: red; font-size: 20px;'>Prediction: {response.text.strip('"')}</p>", unsafe_allow_html=True)
    except Exception as e:
        st.write(f"Error occurred: {e}")


# Application title
st.markdown("""
    <div style="text-align: center;">
        <div style="background-color: tomato; width: 100%; height: 70px; display: flex; justify-content: center; align-items: center; border-radius: 8px; margin: 20px auto;">
            <h1 style="color: white; margin: 0; padding: 0; font-size: 30px;">
                RiskAnalyzer
            </h1>
        </div>
    </div>
""", unsafe_allow_html=True)

form_container = st.form(key="details")
with form_container:
    st.number_input(label="Loan amount", min_value=1, value=10000, key="loan_amnt")
    st.selectbox(label="Loan Term (months)", options=["36 months", "60 months"], key="term")
    st.number_input(label="Interest rate (%)", min_value=1.0, value=10.0, step=0.01, key="int_rate")
    st.selectbox(label="Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"], key="grade")
    st.selectbox(label="Loan Sub-Grade", 
                 options=["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", \
                          "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", \
                          "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", \
                          "E5", "F1", "F2", "F3", "F4", "F5", "G1", "G2", "G3", "G4", "G5"], 
                 key="sub_grade")
    st.selectbox(label="Initial Loan status", options=["f", "w"], key="initial_list_status")
    st.date_input(label="Loan issue date", format="DD-MM-YYYY", min_value=date(2010, 1, 1), max_value="today", key="issue_d")
    st.selectbox(label="Loan application type", options=["Individual", "Non individual"], key="application_type")
    st.selectbox(label="Loan application purpose", 
                 options=["Debt consolidation", "Credit card", "Home improvement", "Small business", \
                          "Moving", "Vacation", "House", "Car", "Medical", "Major purchase", \
                          "Renewable energy", "Wedding", "Educational", "Other"], 
                 key="purpose")
    st.selectbox(label="Home ownership type", options=["Rent", "Mortgage", "Own", "Other"], key="home_ownership")
    st.number_input(label="Annual Income", min_value=1, value=10000, key="annual_inc")
    st.selectbox(label="Income source verification", options=["Verified", "Not Verified"], key="verification_status")
    st.selectbox(label="Total work experience (years)", 
                 options=["< 1 year", "1 year", "2 years", "3 years", "4 years", \
                          "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"], 
                 key="emp_length")
    st.number_input(label="Debt to income ratio (%)", min_value=0.0, value=10.0, step=0.01, key="dti")
    st.date_input(label="Earliest Credit-line date", format="DD-MM-YYYY", min_value=date(1944, 1, 1), max_value="today", key="earliest_cr_line")
    st.number_input(label="Revolving credit balance", min_value=0.0, value=1000.0, step=0.01, key="revol_bal")
    st.number_input(label="Revolving utilization (%)", min_value=0.0, value=10.0, step=0.01, key="revol_util")
    st.number_input(label="Number of Open Credit-lines", min_value=0, value=0, key="open_acc")
    st.number_input(label="Number of Total Credit-lines", min_value=0, value=0, key="total_acc")
    st.number_input(label="Number of Mortgaged Credit-lines", min_value=0, value=0, key="mort_acc")
    st.number_input(label="Number of Public derogatory records", min_value=0, value=0, key="pub_rec")
    st.number_input(label="Number of Public bankruptcy records", min_value=0, value=0, key="pub_rec_bankruptcies")
    st.text_input(label="Address", value="539 Martinez Landing, West Kayleeville, IN 22690", help="Enter address with 5-digit Pincode (at the end)", key="address")

    submit = st.form_submit_button(label="Predict", type="primary", use_container_width=True)

if submit:
    predict()