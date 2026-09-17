# Customer Churn Prediction

## Project Overview

This project predicts whether a telecom customer is likely to churn using machine learning.

The project covers the complete machine learning workflow:

- Data loading
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature preprocessing
- Model training & saving
- Model evaluation
- Prediction pipeline
- REST API with FastAPI
- Automated testing with pytest

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains information about telecom customers, including demographic information, services, contract type, payment method, and charges.

The target variable is `Churn`.

Possible values:
- `Yes`
- `No`

## Machine Learning Model & Performance

- **Model:** Logistic Regression
- **Performance:** ROC-AUC ~ 0.836

The preprocessing pipeline includes:
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features

The preprocessing and model are combined into a single scikit-learn `Pipeline` and serialized using `joblib`.

## Project Structure

```text
ml-customer-churn/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   └── logistic_regression_pipeline.joblib
├── notebooks/
│   └── customer_churn.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── api.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── README.md
└── requirements.txt
```

