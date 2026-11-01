from pathlib import Path
import pandas as pd

# Find the project and dataset directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "oulad"

# Inspect the six smaller CSV files first
files = [
    "courses.csv",
    "assessments.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "vle.csv",
]

for filename in files:
    file_path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"\nShape (rows, columns): {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())