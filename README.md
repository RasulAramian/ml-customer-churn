# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn in a telecommunications company.

This project demonstrates a production-oriented machine learning workflow, covering the complete lifecycle from data exploration and preprocessing to model training, evaluation, REST API development, containerization, monitoring, automated testing, CI/CD, and deployment.

---

## 🚀 Project Overview

Customer churn prediction is a common machine learning problem in which the goal is to identify customers who are likely to leave a service.

In this project, a supervised machine learning pipeline is developed using the IBM Telco Customer Churn dataset.

The project goes beyond model training and demonstrates how a machine learning model can be transformed into a deployable and monitorable service.

### End-to-End Workflow

```text
Raw Data
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Preprocessing Pipeline
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Model Persistence
   │
   ▼
FastAPI REST API
   │
   ▼
Docker Container
   │
   ├──────────────► Prometheus
   │                     │
   │                     ▼
   │                  Grafana
   │
   ▼
Automated Testing
   │
   ▼
GitHub Actions CI/CD
   │
   ▼
GitHub Container Registry
   │
   ▼
Automated Deployment
```

🎯 Project Objectives

The main objectives of this project are:

Analyze customer behavior and identify churn-related patterns.
Build a reproducible machine learning preprocessing pipeline.
Train and evaluate multiple classification models.
Select a production-ready model based on predictive performance and deployment requirements.
Save the complete preprocessing and modeling pipeline.
Expose the model through a REST API.
Containerize the application using Docker.
Add automated API and preprocessing tests.
Implement application monitoring using Prometheus.
Visualize service metrics using Grafana.
Automate testing, image publishing, and deployment with GitHub Actions.
Demonstrate a complete production-oriented ML workflow.
📊 Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information such as:

Demographics
Account information
Contract type
Internet service
Payment method
Monthly charges
Total charges
Tenure
Additional services
Churn status
Dataset Statistics
Property	Value
Number of customers	7,043
Number of features	20
Target variable	Churn
Missing TotalCharges values	11
Churn = No	5,174
Churn = Yes	1,869

The raw dataset is stored at:

data/Telco-Customer-Churn.csv
🔎 Exploratory Data Analysis

The notebook performs a structured exploratory analysis to understand the dataset before model development.

The analysis includes:

Dataset shape and schema
Data types
Missing values
Duplicate records
Target distribution
Contract type and churn
Tenure and churn
Internet service and churn
Payment method and churn
Monthly charges
Total charges
Customer segmentation

Examples of questions investigated during EDA include:

How does churn vary across contract types?
Are customers with shorter tenure more likely to churn?
How is churn distributed across internet service types?
Are payment methods associated with different churn rates?
How are monthly charges distributed across customers?

The complete analysis is available in:

notebooks/customer_churn.ipynb
🧹 Data Cleaning and Preprocessing

The preprocessing stage is implemented using a reproducible scikit-learn pipeline.

Data Cleaning

The dataset requires several preprocessing steps, including:

Converting TotalCharges to numeric values.
Handling missing values.
Removing non-predictive columns.
Separating features from the target variable.
Removing exploratory-only variables before model training.
Feature Types

Numerical features include:

SeniorCitizen
tenure
MonthlyCharges
TotalCharges

Categorical features include variables such as:

gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
Numerical Preprocessing

Numerical features are standardized using:

StandardScaler
Categorical Preprocessing

Categorical features are encoded using:

OneHotEncoder(handle_unknown="ignore")

This allows the production API to handle previously unseen categorical values without breaking the preprocessing pipeline.

🤖 Machine Learning Models

Several classification models are evaluated during the modeling stage.

The project investigates:

Logistic Regression
Decision Tree
Random Forest

The primary production model is:

Logistic Regression

The final production artifact contains both preprocessing and the trained model inside a single scikit-learn pipeline.

This prevents inconsistencies between training-time preprocessing and inference-time preprocessing.

📈 Model Evaluation

The models are evaluated using several classification metrics.

These include:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC Curve
ROC-AUC
Precision-Recall Curve
Classification threshold analysis

The main evaluation metric used for model assessment is:

ROC-AUC ≈ 0.836

The notebook also investigates how changing the classification threshold affects precision and recall.

This is important in churn prediction because the business impact of false positives and false negatives may not be identical.

💾 Model Persistence

The complete production pipeline is saved using joblib.

models/logistic_regression_pipeline.joblib

The saved artifact contains:

Preprocessing
     +
Logistic Regression

Therefore, the API can load one artifact and perform the same transformations used during training.

🔮 Prediction Pipeline

The production prediction flow is:

Customer Input
      │
      ▼
Input Validation
      │
      ▼
Preprocessing
      │
      ▼
Logistic Regression
      │
      ▼
Churn Probability
      │
      ▼
Churn Prediction

The API returns both:

Predicted churn class
Churn probability

Example response:

{
  "churn_prediction": 0,
  "churn_probability": 0.112
}
🌐 REST API

The trained model is exposed through a FastAPI application.

The API implementation is located at:

src/api.py
Start the API locally
uvicorn src.api:app --reload --port 8000

The API will be available at:

http://localhost:8000
Interactive API Documentation

FastAPI automatically provides interactive documentation at:

http://localhost:8000/docs
/predict

The main endpoint is:

POST /predict

It accepts customer information and returns a churn prediction and probability.

The API includes input validation using FastAPI/Pydantic.

Invalid requests are rejected with an appropriate HTTP status code.

🧪 Testing

Automated tests are implemented using pytest.

The project includes tests for:

Preprocessing
Prediction behavior
API endpoint functionality

Run the complete test suite with:

pytest -v

The tests help ensure that changes to the project do not silently break the preprocessing pipeline or API.

🐳 Docker

The application is containerized using Docker.

The Docker image contains:

Python runtime
Project dependencies
Source code
Trained model
FastAPI application

Build the image:

docker build -t customer-churn .

Run the container:

docker run -d \
  --name customer-churn \
  -p 8000:8000 \
  customer-churn

The API is then available at:

http://localhost:8000
🧩 Docker Compose

The project also includes a multi-service Docker Compose setup.

The stack contains:

FastAPI
Prometheus
Grafana

Start the complete stack:

docker compose up -d --build

Check running services:

docker ps

Stop the stack:

docker compose down
Service Endpoints
Service	URL
FastAPI	http://localhost:8002
FastAPI Docs	http://localhost:8002/docs
Prometheus	http://localhost:9090
Grafana	http://localhost:3000

The API container listens on port 8000, while Docker Compose exposes it on host port 8002.

📡 Monitoring

Application metrics are exposed through:

/metrics

Prometheus collects these metrics from the API service.

The monitoring stack tracks metrics such as:

API availability
Total HTTP requests
Request rate
Average response time
4xx client errors
5xx server errors

This provides basic observability for the deployed machine learning service.

📊 Grafana Dashboard

Grafana is used to visualize application metrics collected by Prometheus.

The dashboard includes panels for:

API Status

Shows whether the API service is available.

Total HTTP Requests

Tracks the total number of requests received by the API.

Request Rate

Shows the request rate over time.

Average Response Time

Tracks the average API response latency.

Client Errors

Tracks HTTP 4xx responses.

Server Errors

Tracks HTTP 5xx responses.

This allows the behavior of the deployed service to be monitored without inspecting application logs manually.

🔄 CI/CD

The project uses GitHub Actions to automate the software delivery workflow.

The CI/CD pipeline performs:

Git Push
   │
   ▼
Run Tests
   │
   ▼
Build Docker Image
   │
   ▼
Publish Image to GHCR
   │
   ▼
Deploy to Self-Hosted Runner
Continuous Integration

The test job:

Checks out the repository.
Sets up Python.
Installs dependencies.
Runs the complete pytest suite.
Container Publishing

After successful tests, the Docker image is built and published to:

GitHub Container Registry (GHCR)

The image is tagged using the Git commit SHA.

This provides immutable image versions for deployment.

Continuous Deployment

The deployment stage:

Logs in to GHCR.
Pulls the newly published image.
Stops the previous container.
Removes the old container.
Starts the new container.

The deployment runs on a self-hosted GitHub Actions runner.

📦 Project Structure
ml-customer-churn/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   └── logistic_regression_pipeline.joblib
│
├── notebooks/
│   └── customer_churn.ipynb
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model_metadata.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── prometheus.yml
├── pytest.ini
├── requirements.txt
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/RasulAramian/ml-customer-churn.git

Move into the project directory:

cd ml-customer-churn

Create a virtual environment:

python -m venv .venv

Activate it on Linux/macOS:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project Locally

Start the API:

uvicorn src.api:app --reload --port 8000

Open the interactive API documentation:

http://localhost:8000/docs

Run tests:

pytest -v
🐳 Running the Complete Stack

To start the API, Prometheus, and Grafana together:

docker compose up -d --build

Check the containers:

docker ps

Access the services:

API:
http://localhost:8002

API Documentation:
http://localhost:8002/docs

Prometheus:
http://localhost:9090

Grafana:
http://localhost:3000

Stop the services:

docker compose down
🔐 Configuration and Security

Environment-specific configuration is separated from the application code.

Example environment configuration:

.env.example

Sensitive configuration should not be committed to Git.

The project uses:

.env

for local environment-specific variables and excludes it through .gitignore.

The .env.example file provides a template for required configuration without exposing sensitive values.

🧠 Engineering Practices

This project demonstrates several practical machine learning engineering principles:

Reproducible preprocessing

The preprocessing logic is included inside the model pipeline.

Separation of concerns

Data processing, training, evaluation, prediction, and API logic are separated into different modules.

Automated testing

Core functionality is covered by pytest tests.

Containerization

The API and model are packaged into a Docker image.

Observability

Prometheus and Grafana provide basic application monitoring.

CI/CD automation

GitHub Actions automatically tests, builds, publishes, and deploys the application.

Versioned deployment artifacts

Docker images are tagged using Git commit SHA values.

📓 Notebook

The complete machine learning workflow is documented in:

notebooks/customer_churn.ipynb

The notebook covers:

Project Overview
Business Problem
Dataset Overview
Data Quality Assessment
Data Cleaning
Exploratory Data Analysis
Feature Preparation
Train/Test Split
Preprocessing Pipeline
Model Training
Model Evaluation
ROC-AUC Analysis
Precision-Recall Analysis
Threshold Analysis
Model Interpretation
Permutation Importance
Model Persistence
Prediction Example
Conclusions
Future Improvements

The notebook is designed to provide a clear analytical narrative rather than simply presenting a collection of code cells.

🚀 Future Improvements

Possible extensions include:

Hyperparameter optimization across all candidate models.
Cross-validation and more extensive model comparison.
Model calibration.
Experiment tracking with MLflow.
Data drift detection.
Model performance monitoring.
Automated retraining pipelines.
Cloud deployment.
Kubernetes-based deployment.
Authentication and authorization for the API.
API rate limiting.
Structured application logging.
Distributed monitoring and alerting.
📌 Key Takeaways

This project demonstrates the complete lifecycle of a machine learning application:

Data
 ↓
EDA
 ↓
Preprocessing
 ↓
Modeling
 ↓
Evaluation
 ↓
Model Persistence
 ↓
REST API
 ↓
Docker
 ↓
Monitoring
 ↓
Testing
 ↓
CI/CD
 ↓
Deployment

The main focus is not only on building a predictive model, but also on demonstrating how machine learning can be integrated into a reproducible, testable, deployable, and monitorable software system.

👨‍💻 Author

Rasul Aramian

PhD in Mathematics — Lie Algebra
MSc — Group Theory & Algebraic Graph Theory

Interests:

Machine Learning
Deep Learning
Mathematical Modeling
Machine Learning Engineering
AI Applications
Data Science
📄 License

This project is intended for educational and portfolio purposes.
