import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data import load_data
from src.preprocessing import create_preprocessor


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"


def train_model():

    df = load_data(DATA_PATH)

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = create_preprocessor()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print("Model saved successfully.")
    print(f"Model path: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
