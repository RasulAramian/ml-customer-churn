from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_predict_valid_customer():

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 1920.0
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result

    assert result["prediction"] in ["Yes", "No"]

    assert 0 <= result["churn_probability"] <= 1


def test_predict_invalid_customer():

    customer = {
        "gender": "Unknown",
        "SeniorCitizen": 5,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Three year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": -50.0,
        "TotalCharges": 1000.0
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 422
    
    
    
