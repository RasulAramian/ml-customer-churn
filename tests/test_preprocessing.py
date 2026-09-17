import pandas as pd

from src.preprocessing import create_preprocessor


def test_preprocessor():
    data = pd.DataFrame({
        "SeniorCitizen": [0, 1],
        "tenure": [5, 20],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": [250.0, 1600.0],

        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Bank transfer (automatic)"
        ]
    })

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(data)

    assert X_transformed.shape[0] == 2
    
    
def test_preprocessor_handles_unknown_category():
    train_data = pd.DataFrame({
        "SeniorCitizen": [0, 1],
        "tenure": [5, 20],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": [250.0, 1600.0],

        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Bank transfer (automatic)"
        ]
    })

    test_data = train_data.copy()

    test_data.loc[0, "Contract"] = "Three year"

    preprocessor = create_preprocessor()

    preprocessor.fit(train_data)

    transformed = preprocessor.transform(test_data)

    assert transformed.shape[0] == 2    
    
