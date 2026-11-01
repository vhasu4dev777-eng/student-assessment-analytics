from pathlib import Path
import sqlite3
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "processed" / "oulad.db"
SQL_DIR = PROJECT_DIR / "sql"

if len(sys.argv) != 2:
    print("Usage: python src/run_sql.py <sql_filename>")
    raise SystemExit(1)

sql_file = SQL_DIR / sys.argv[1]

if not sql_file.is_file() or sql_file.parent != SQL_DIR:
    print("Error: SQL file not found in the sql folder.")
    raise SystemExit(1)

query = sql_file.read_text(encoding="utf-8")

with sqlite3.connect(DB_PATH) as connection:
    results = pd.read_sql_query(query, connection)

print(f"\nRESULTS: {sql_file.name}")
print(results.to_string(index=False))