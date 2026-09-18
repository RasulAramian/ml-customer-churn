from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import logging

from prometheus_fastapi_instrumentator import Instrumentator

from src.model_metadata import MODEL_NAME, MODEL_VERSION
from src.predict import predict_churn


# =========================
# Logging
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# FastAPI application
# =========================

app = FastAPI()

Instrumentator().instrument(app).expose(app)

# =========================
# Health check
# =========================

@app.get("/health")
def health():
    return {"status": "healthy"}


# =========================
# Model information
# =========================

@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION
    }


# =========================
# Customer input schema
# =========================

class Customer(BaseModel):

    gender: Literal["Female", "Male"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0)

    TotalCharges: float = Field(ge=0)


# =========================
# Prediction endpoint
# =========================

@app.post("/predict")
def predict(customer: Customer):

    logger.info("Prediction request received")

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    prediction, probability = predict_churn(
        customer_df
    )

    logger.info(
        "Prediction completed | model=%s | version=%s | prediction=%s",
        MODEL_NAME,
        MODEL_VERSION,
        prediction
    )

    return {
        "prediction": prediction,
        "churn_probability": probability
    }
