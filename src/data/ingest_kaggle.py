"""Load and filter Kaggle data for years 1960-2020."""

from pathlib import Path

import pandas as pd


def run(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    filtered = df[(df["year"] >= 1960) & (df["year"] <= 2020)].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)


if __name__ == "__main__":
    raise SystemExit("Use this module from a pipeline script.")
