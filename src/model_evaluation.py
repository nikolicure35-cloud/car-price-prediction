"""Print evaluation metrics and example predictions from saved training reports."""
from __future__ import annotations
import json
import pandas as pd


def main():
    summary = json.load(open("reports/training_summary.json", encoding="utf-8"))
    print(f"Best model: {summary['best_model']}")
    for key, value in summary["best_metrics"].items():
        if key != "model": print(f"{key}: {value:,.3f}")
    print("\nExamples (first rows of hold-out set):")
    print(pd.read_csv("reports/sample_predictions.csv").head(10).round(0).to_string(index=False))

if __name__ == "__main__": main()
