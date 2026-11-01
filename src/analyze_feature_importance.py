
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Load data
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_dataset.csv"

data = pd.read_csv(DATA_PATH)

# 2. Select features
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

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Preprocess categorical columns
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

# 5. Train the same Random Forest configuration
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

model.fit(X_train, y_train)

# 6. Get encoded feature names and importance scores
fitted_preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["classifier"]

encoded_names = fitted_preprocessor.get_feature_names_out()
importance_values = random_forest.feature_importances_

importance = pd.DataFrame(
    {
        "encoded_feature": encoded_names,
        "importance": importance_values,
    }
)

# 7. Map encoded columns back to their original features
def get_original_feature(encoded_feature):
    name = encoded_feature.split("__", 1)[-1]

    for feature in categorical_features:
        if name == feature or name.startswith(feature + "_"):
            return feature

    for feature in numeric_features:
        if name == feature:
            return feature

    return name


importance["original_feature"] = importance["encoded_feature"].apply(
    get_original_feature
)

# 8. Combine the importance of encoded categories
grouped_importance = (
    importance.groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
)

print("\nFEATURE IMPORTANCE")
print("-" * 40)

for feature, score in grouped_importance.items():
    print(f"{feature:25s} {score:.4f}")

# 9. Save the results
output_path = (
    PROJECT_DIR / "data" / "processed" / "feature_importance.csv"
)

grouped_importance.rename("importance").to_csv(output_path)

print(f"\nSaved feature importance to: {output_path}")