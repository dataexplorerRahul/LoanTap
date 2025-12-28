import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request, HTTPException


router = APIRouter()


# Data model for input
class InputRecord(BaseModel):
    loan_amnt: float = Field(gt=0, description="Principal amount of the loan")
    term: Literal["36 months", "60 months"] = Field(description="Duration of the loan")
    int_rate: float = Field(gt=0, lt=101, description="Interest rate on the principal amount")
    installment: float = Field(gt=0, description="Installment amount per month")
    grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(description="Grade assigned to the loan")
    sub_grade: Literal["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5", "G1", "G2", "G3", "G4", "G5"] \
        = Field(description="Grade assigned to the loan")
    emp_length: Literal["10+ years", "4 years", "8 years", "9 years", "1 year", "3 years", "< 1 year", "6 years", "7 years", "5 years", "2 years"] \
        = Field(description="Work experience of applicant in years")
    home_ownership: Literal["OWN", "RENT", "MORTGAGE", "OTHER"] = Field(description="Type of Ownership of home")
    annual_inc: float = Field(gt=0, description="Annual income of the applicant")
    verification_status: Literal["Verified", "Not Verified"] = Field(description="Verfication status of applicant's income source")
    issue_d: str = Field(description="Issue date of the loan by the lender")
    purpose: str = Field(description="Purpose of the loan")
    dti: float = Field(gt=-1, description="Monthly total-debt to income ratio")
    earliest_cr_line: str = Field(description="Date of the first credit line of the applicant")
    open_acc: float = Field(gt=-1, description="Number of open credit lines of the applicant")
    pub_rec: float = Field(gt=-1, description="Number of derogatory public records")
    revol_bal: float = Field(gt=-1, description="Total credit revolving balance")
    revol_util: float = Field(gt=-1, lt=101, description="Revolving line utilization rate")
    total_acc: float = Field(gt=-1, description="Total number of credit lines of the applicant")
    initial_list_status: Literal["w", "f"] = Field(description="Initial listing status of the loan")
    application_type: Literal["INDIVIDUAL", "NON_INDIVIDUAL"] = Field(description="Type of loan applicant i.e. Individual or not")
    mort_acc: float = Field(gt=-1, description="Number of mortgage accounts")
    pub_rec_bankruptcies: float = Field(gt=-1, description="Number of public record bankruptcies")
    address: str = Field(description="Address of the applicant")


@router.post("/predict")
def predict(record: InputRecord, request: Request):
    try:
        model = request.app.state.model

        if not model:
            raise HTTPException(status_code=500, detail="Model not loaded")

        data_dict = record.model_dump()
        record_df = pd.DataFrame([data_dict])
        y_probs = model.predict_proba(record_df)[:, 1] # probability for class=1

        # Return prediction
        threshold = 0.6088
        if y_probs[0] >= threshold:
            prediction = "Defaulter"
        else:
            prediction = "Not a defaulter"
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    