import json
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root():
    """
    Test the root endpoint to ensure it returns a 200 status code and a welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the salary prediction API!"}


def test_salary_prediction_over_50k():
    """
    Test the salary prediction endpoint with an input that should predict a salary over 50k.
    """
    data = {
        "age": 50,
        "workclass": "Self-emp-inc",
        "fnlwgt": 83311,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 99999,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States",
    }
    response = client.post(
        "/predict/",
        json.dumps(data),
        headers={
            "Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json() == {"salary_prediction": ">50K"}


def test_salary_prediction_under_50k():
    """
    Test the salary prediction endpoint with an input that should predict a salary under 50k.
    """
    data = {
        "age": 30,
        "workclass": "Private",
        "fnlwgt": 230801,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Never-married",
        "occupation": "Sales",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    response = client.post(
        "/predict/",
        json.dumps(data),
        headers={
            "Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json() == {"salary_prediction": "<=50K"}
