from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

# Use only information available during the first 30 days.
X = data[["clicks_first_30_days"]]
y = data["at_risk"]

# Split the data while preserving the class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Standardise the feature and train logistic regression.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42),
)

model.fit(X_train, y_train)

# Generate predictions on the held-out test set.
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("LOGISTIC REGRESSION RESULTS")
print("=" * 35)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nEvaluation metrics:")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Precision:",
    round(precision_score(y_test, predictions, zero_division=0), 4),
)
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 score:", round(f1_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

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