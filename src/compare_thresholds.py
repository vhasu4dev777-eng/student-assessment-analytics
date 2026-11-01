
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Load data
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_dataset.csv"

data = pd.read_csv(DATA_PATH)

# 2. Define features
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

# 3. Split by student
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# 4. Preprocessing
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

# 5. Build the model
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

# 6. Train the model
model.fit(X_train, y_train)

# 7. Get predicted probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# 8. Compare probability thresholds
results = []

for threshold in [0.50, 0.40, 0.30]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn,
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
    )

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print("-" * 100)
print(results_df.round(4).to_string(index=False))

# 9. Save results
output_path = (
    PROJECT_DIR / "data" / "processed" / "threshold_comparison.csv"
)

results_df.to_csv(output_path, index=False)

print(f"\nSaved results to: {output_path}")
