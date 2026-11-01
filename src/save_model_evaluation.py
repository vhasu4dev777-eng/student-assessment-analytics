
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

threshold_path = OUTPUT_DIR / "threshold_comparison.csv"
importance_path = OUTPUT_DIR / "feature_importance.csv"

thresholds = pd.read_csv(threshold_path)
importance = pd.read_csv(importance_path)

# Select the threshold with the highest F1 score
best_row = thresholds.loc[thresholds["f1_score"].idxmax()]

print("\nFINAL MODEL EVALUATION SUMMARY")
print("=" * 45)

print("Model: Random Forest")
print("Evaluation: Student-level train/test split")
print("Selected threshold:", best_row["threshold"])
print(f"Accuracy:  {best_row['accuracy']:.4f}")
print(f"Precision: {best_row['precision']:.4f}")
print(f"Recall:    {best_row['recall']:.4f}")
print(f"F1 score:  {best_row['f1_score']:.4f}")
print(f"ROC-AUC:   {best_row['roc_auc']:.4f}")

print("\nConfusion matrix counts:")
print("True negatives:", int(best_row["true_negatives"]))
print("False positives:", int(best_row["false_positives"]))
print("False negatives:", int(best_row["false_negatives"]))
print("True positives:", int(best_row["true_positives"]))

print("\nFeature importance:")
print(importance.to_string(index=False))
