"""Ingest Kaggle Spotify dataset (1960-2020)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = [
    "artists",
    "name",
    "duration_ms",
    "year",
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "popularity",
]


def load_and_prepare_kaggle_data(
    csv_path: str | Path,
    start_year: int = 1980,
    end_year: int = 2020,
) -> pd.DataFrame:
    """Load, filter, and align Kaggle CSV to required columns."""
    csv_path = Path(csv_path)
    logger.info("Loading Kaggle CSV: %s", csv_path)

    df = pd.read_csv(csv_path, low_memory=False)

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    elif "release_date" in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    else:
        raise ValueError("Dataset must have 'year' or 'release_date' column")

    df = df[df["year"].notna()].copy()
    df = df[
        (df["year"].astype(int) >= start_year)
        & (df["year"].astype(int) <= end_year)
    ].copy()
    logger.info("Filtered to %d rows in range %d-%d", len(df), start_year, end_year)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    kaggle_df = df[REQUIRED_COLUMNS].copy()
    kaggle_df["year"] = kaggle_df["year"].astype("Int64")

    logger.info("Kaggle prepared: %d rows", len(kaggle_df))
    return kaggle_df


def run(
    input_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    start_year: int = 1980,
    end_year: int = 2020,
) -> pd.DataFrame:
    """Load and save Kaggle data."""
    logging.basicConfig(level=logging.INFO)

    if input_csv is None:
        input_csv = PROJECT_ROOT / "data/interim/kaggle_1960_2020.csv"

    kaggle_df = load_and_prepare_kaggle_data(input_csv, start_year, end_year)

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kaggle_df.to_csv(output_path, index=False)
        logger.info("Saved to %s", output_path)

    return kaggle_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Kaggle Spotify dataset")
    parser.add_argument("--input-csv", type=str, help="Input Kaggle CSV path")
    parser.add_argument("--output-csv", type=str, help="Output CSV path")
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2020)
    args = parser.parse_args()

    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        start_year=args.start_year,
        end_year=args.end_year,
    )