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

print("Dataset loaded successfully.")
print("Total records:", len(data))

# Summarise early engagement for each risk group.
print("\nEarly engagement by risk group:")

engagement_summary = (
    data.groupby("at_risk")["clicks_first_30_days"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
    .rename(index={
        0: "Not at risk",
        1: "At risk",
    })
)

print(engagement_summary)

# Create engagement groups to make the results easier to interpret.
data["engagement_group"] = pd.cut(
    data["clicks_first_30_days"],
    bins=[-1, 0, 100, 300, 1000, float("inf")],
    labels=[
        "No clicks",
        "1–100 clicks",
        "101–300 clicks",
        "301–1000 clicks",
        "1000+ clicks",
    ],
)

print("\nAt-risk rate by engagement group:")

risk_by_engagement = (
    data.groupby("engagement_group", observed=False)
    .agg(
        total_students=("at_risk", "size"),
        at_risk_students=("at_risk", "sum"),
        at_risk_rate=("at_risk", "mean"),
    )
)

risk_by_engagement["at_risk_rate"] = (
    risk_by_engagement["at_risk_rate"] * 100
).round(2)

print(risk_by_engagement)