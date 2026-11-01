from pathlib import Path
import sqlite3
import pandas as pd

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "oulad"
DB_DIR = PROJECT_DIR / "data" / "processed"
DB_PATH = DB_DIR / "oulad.db"

# Ensure the database directory exists
DB_DIR.mkdir(parents=True, exist_ok=True)

# Start with the smaller tables
files = [
    "courses.csv",
    "assessments.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "vle.csv",
]

with sqlite3.connect(DB_PATH) as connection:
    for filename in files:
        table_name = Path(filename).stem

        print(f"Loading {filename}...")

        df = pd.read_csv(DATA_DIR / filename)

        df.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
        )

        print(f"Loaded {len(df):,} rows into {table_name}")

print(f"\nDatabase created at: {DB_PATH}")