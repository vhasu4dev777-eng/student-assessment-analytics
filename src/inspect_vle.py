from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
VLE_PATH = PROJECT_DIR / "data" / "raw" / "oulad" / "studentVle.csv"

print("Checking whether the VLE dataset exists...")

if not VLE_PATH.exists():
    print(f"Dataset not found: {VLE_PATH}")
    raise SystemExit(1)

print(f"Dataset found: {VLE_PATH}")
print(f"File size: {VLE_PATH.stat().st_size / (1024 ** 2):.2f} MB")

# Read only the first 10,000 rows to avoid loading the entire file.
sample = pd.read_csv(VLE_PATH, nrows=10000)

print("\nSample shape:", sample.shape)

print("\nColumn names:")
print(sample.columns.tolist())

print("\nFirst five rows:")
print(sample.head())

print("\nMissing values:")
print(sample.isnull().sum())