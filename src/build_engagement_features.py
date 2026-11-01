from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

VLE_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "oulad"
    / "studentVle.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "early_engagement.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Store aggregated clicks for each student and course presentation.
engagement_parts = []

# Process the large CSV in manageable chunks.
chunk_size = 100_000

print("Processing studentVle.csv...")
print("Counting clicks recorded during days 0–30.")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        VLE_PATH,
        usecols=[
            "code_module",
            "code_presentation",
            "id_student",
            "date",
            "sum_click",
        ],
        chunksize=chunk_size,
    ),
    start=1,
):
    # Keep interactions recorded from day 0 through day 30.
    early_activity = chunk[
        chunk["date"].between(0, 30)
    ]

    # Aggregate clicks by student, module and presentation.
    if not early_activity.empty:
        grouped = (
            early_activity
            .groupby(
                [
                    "code_module",
                    "code_presentation",
                    "id_student",
                ],
                as_index=False,
            )["sum_click"]
            .sum()
            .rename(
                columns={
                    "sum_click": "clicks_first_30_days"
                }
            )
        )

        engagement_parts.append(grouped)

    if chunk_number % 50 == 0:
        print(f"Processed {chunk_number} chunks...")

if not engagement_parts:
    raise RuntimeError(
        "No interactions were found during days 0–30."
    )

# Combine partial aggregates and aggregate again because a
# student may appear in multiple CSV chunks.
engagement = pd.concat(
    engagement_parts,
    ignore_index=True,
)

engagement = (
    engagement
    .groupby(
        [
            "code_module",
            "code_presentation",
            "id_student",
        ],
        as_index=False,
    )["clicks_first_30_days"]
    .sum()
)

engagement.to_csv(OUTPUT_PATH, index=False)

print("\nFeature generation complete.")
print("Number of student-course records:", len(engagement))
print("Saved to:", OUTPUT_PATH)

print("\nFeature summary:")
print(
    engagement["clicks_first_30_days"].describe().round(2)
)

print("\nFirst five records:")
print(engagement.head())