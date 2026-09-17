import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.data import load_data


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"


def evaluate_model():

    # -------------------------
    # 1. Load data
    # -------------------------
    df = load_data(DATA_PATH)

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = df["Churn"]

    # -------------------------
    # 2. Same train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # 3. Load trained model
    # -------------------------
    model = joblib.load(MODEL_PATH)

    # -------------------------
    # 4. Predictions
    # -------------------------
    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    # -------------------------
    # 5. Metrics
    # -------------------------
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Yes"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Yes"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Yes"
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba
    )

    # -------------------------
    # 6. Print results
    # -------------------------
    print("\nModel Evaluation")
    print("================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print("================")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=["No", "Yes"]
        )
    )

    print("\nClassification Report")
    print("====================")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


if __name__ == "__main__":
    evaluate_model()
