from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_dataset.csv"
)

data = pd.read_csv(DATA_PATH)

# Use only early engagement as the initial feature.
X = data[["clicks_first_30_days"]]
y = data["at_risk"]

# Keep the same risk-group proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Establish a baseline that always predicts the majority class.
model = DummyClassifier(strategy="most_frequent")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nBaseline model results")
print("----------------------")

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Precision:",
    round(precision_score(y_test, predictions, zero_division=0), 4),
)
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 score:", round(f1_score(y_test, predictions), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Not at risk", "At risk"],
        zero_division=0,
    )
)