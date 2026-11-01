from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "model_dataset.csv"
)

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)

print("\nAll columns:")
for column in data.columns:
    print(f"- {column}")

print("\nData types:")
print(data.dtypes)

print("\nSelected feature summaries:")

candidate_columns = [
    "num_of_prev_attempts",
    "studied_credits",
    "age_band",
    "highest_education",
    "code_module",
    "code_presentation",
    "clicks_first_30_days",
]

for column in candidate_columns:
    if column in data.columns:
        print(f"\n{column}")
        print(data[column].value_counts().head(10))