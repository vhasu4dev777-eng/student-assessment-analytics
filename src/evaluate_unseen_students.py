
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Load the dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_dataset.csv"

data = pd.read_csv(DATA_PATH)

# 2. Select features and target
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

X = data[features]
y = data["at_risk"]
groups = data["id_student"]

# 3. Split by student ID, not by individual row
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_students = set(groups.iloc[train_indices])
test_students = set(groups.iloc[test_indices])

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training students:", len(train_students))
print("Testing students:", len(test_students))
print("Students appearing in both sets:", len(train_students & test_students))

# 4. Preprocess categorical features
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

# 5. Create the model
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

# 6. Train and evaluate
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nUNSEEN-STUDENT EVALUATION")
print("-" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        zero_division=0,
    )
)
