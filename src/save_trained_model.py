from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "student_risk_model.joblib"

# Features used by the existing model
numeric_features = [
    "clicks_first_30_days",
    "num_of_prev_attempts",
    "studied_credits",
]

categorical_features = [
    "age_band",
    "highest_education",
    "code_module",
    "code_presentation",
]

features = numeric_features + categorical_features


def main():
    # Load the processed dataset
    data = pd.read_csv(DATA_PATH)

    X = data[features]
    y = data["at_risk"]

    # Preprocess categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    # Use the same Random Forest configuration
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )

    # Train the final model using the full dataset
    model.fit(X, y)

    # Save the complete preprocessing and prediction pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved successfully.")
    print(f"Training records: {len(data):,}")
    print(f"Model location: {MODEL_PATH}")


if __name__ == "__main__":
    main()