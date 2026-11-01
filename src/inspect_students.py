from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "oulad"

# Load student information
df = pd.read_csv(DATA_DIR / "studentInfo.csv")

print("DATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nFIRST 5 STUDENTS")
print(df.head().to_string(index=False))

print("\nFINAL RESULT COUNTS")
print(df["final_result"].value_counts())

print("\nMISSING VALUES")
print(df.isnull().sum())