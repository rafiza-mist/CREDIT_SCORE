from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Load the trained model
model = joblib.load("credit_model.pkl")

# Set up the API
app = FastAPI()

# Define the request format
class CreditRequest(BaseModel):
    age: int
    occupation: str  # PSN, BUS, GOV, SEM
    annual_income: float
    avg_balance: int
    debt_inc_ratio: float

# Define a map for occupation
occupation_map = {'PSN': 0, 'BUS': 1, 'GOV': 2, 'SEM': 3}

@app.post("/predict")
def predict_credit_risk(data: CreditRequest):
    occ_code = occupation_map.get(data.occupation.upper(), -1)
    if occ_code == -1:
        return {"error": "Invalid occupation"}

    df = pd.DataFrame([{
        'AGE': data.age,
        'OCCUPATION': occ_code,
        'ANNUAL_INCOME': data.annual_income,
        'AVG_BALANCE': data.avg_balance,
        'DEBT_INC_RATIO': data.debt_inc_ratio
    }])

    prediction = model.predict(df)[0]
    return {"credit_risk": (prediction)}  # e.g., 0 = High, 1 = Low
