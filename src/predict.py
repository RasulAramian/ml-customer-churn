import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"


# Load model once
model = joblib.load(MODEL_PATH)


def predict_churn(customer):
    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0, 1]

    return prediction, probability


def predict_batch(customers):
    predictions = model.predict(customers)
    probabilities = model.predict_proba(customers)[:, 1]

    results = customers.copy()

    results["Prediction"] = predictions
    results["ChurnProbability"] = probabilities

    return results


if __name__ == "__main__":

    new_customer = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [5],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [85.0],
        "TotalCharges": [425.0]
    })

    prediction, probability = predict_churn(new_customer)

    print("Prediction:", prediction)
    print(f"Churn probability: {probability:.2%}")

    customers = pd.concat(
        [
            new_customer,
            new_customer.copy()
        ],
        ignore_index=True
    )

    results = predict_batch(customers)

    print("\nBatch Prediction")
    print("================")
    print(results[["Prediction", "ChurnProbability"]])
