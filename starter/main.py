import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

from starter.ml.data import process_data
from starter.ml.model import inference

# Load the trained model, encoder, and label binarizer
with open('model/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)

# Create a FastAPI instance
app = FastAPI()

# Define the categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class EmployeeInfo(BaseModel):
    """
    Pydantic schema for the input data, including an example.
    Using underscore for variables due 'SyntaxError: illegal target for annotation'
    """
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }

class SalaryPrediction(BaseModel):
    salary_prediction: str

@app.get('/')
async def root():
    """
    Root endpoint with a welcome message.
    """
    return {"message": "Welcome to the salary prediction API!"}

@app.post('/predict/', response_model=SalaryPrediction)
async def predict_salary(employee_info: EmployeeInfo):
    """
    Run model inference on the input employee information and return the salary prediction.

    :param employee_info: Employee information as a Pydantic schema
    :return: Salary prediction as a JSON response
    """
    try:
        sample_dict = {key.replace('_', '-'): [value] for key, value in employee_info.__dict__.items()}
        data = pd.DataFrame.from_dict(sample_dict)

        X_input, _, _, _ = process_data(data, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb)
        preds = inference(rf_model, X_input)
        salary_prediction = lb.inverse_transform(preds)

        return {"salary_prediction": salary_prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred during processing: {e}")
