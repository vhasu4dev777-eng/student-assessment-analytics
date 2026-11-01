from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

STUDENT_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "oulad"
    / "studentInfo.csv"
)

ENGAGEMENT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "early_engagement.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_dataset.csv"
)

print("Loading student information...")
students = pd.read_csv(STUDENT_PATH)

print("Loading early engagement features...")
engagement = pd.read_csv(ENGAGEMENT_PATH)

# Join using all three identifiers to prevent mixing
# students from different modules or presentations.
model_data = students.merge(
    engagement,
    on=[
        "code_module",
        "code_presentation",
        "id_student",
    ],
    how="left",
    validate="one_to_one",
)

# Students without a recorded click during the first
# 30 days receive zero clicks.
model_data["clicks_first_30_days"] = (
    model_data["clicks_first_30_days"]
    .fillna(0)
    .astype("int64")
)

# Define the project's binary target.
model_data["at_risk"] = (
    model_data["final_result"]
    .isin(["Fail", "Withdrawn"])
    .astype("int64")
)

# Save the modelling dataset.
model_data.to_csv(OUTPUT_PATH, index=False)

print("\nDataset created successfully!")
print("Dataset shape:", model_data.shape)

print("\nAt-risk distribution:")
print(
    model_data["at_risk"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Not at risk", 1: "At risk"})
)

print("\nAt-risk percentage:")
print(round(model_data["at_risk"].mean() * 100, 2), "%")

print("\nMissing values in selected columns:")
print(
    model_data[
        [
            "id_student",
            "code_module",
            "code_presentation",
            "clicks_first_30_days",
            "final_result",
            "at_risk",
        ]
    ].isnull().sum()
)

print("\nFirst five records:")
print(model_data.head())

print("\nSaved to:", OUTPUT_PATH)