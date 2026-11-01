from pathlib import Path
import sqlite3

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "data" / "processed" / "oulad.db"

if not DB_PATH.exists():
    print("ERROR: Database file not found.")
    raise SystemExit(1)

with sqlite3.connect(DB_PATH) as connection:
    cursor = connection.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
    """)

    tables = cursor.fetchall()

    print("TABLES IN DATABASE")
    for (table_name,) in tables:
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cursor.fetchone()[0]
        print(f"{table_name}: {row_count:,} rows")