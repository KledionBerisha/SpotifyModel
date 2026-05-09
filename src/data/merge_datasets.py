"""Merge Kaggle and Spotify API datasets into unified table."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def merge_datasets(
    kaggle_df: pd.DataFrame,
    api_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Kaggle and Spotify API datasets."""
    logger.info("Merging: Kaggle (%d rows) + API (%d rows)", len(kaggle_df), len(api_df))
    merged = pd.concat([kaggle_df, api_df], ignore_index=True)
    logger.info("Merged: %d total rows", len(merged))
    return merged


def run(
    kaggle_csv: str | Path | None = None,
    spotify_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run full merge pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if kaggle_csv is None:
        kaggle_csv = PROJECT_ROOT / "data/interim/kaggle_prepared.csv"
        
    if spotify_csv is None:
        spotify_csv = PROJECT_ROOT / "data/interim/spotify_api_2021_2025.csv"

    if output_csv is None:
        output_csv = PROJECT_ROOT / "data/processed/Spotify_1980_2025_Final.csv"

    logger.info("=" * 80)
    logger.info("STEP 1: Loading Kaggle data...")
    logger.info("=" * 80)
    kaggle_df = pd.read_csv(kaggle_csv, low_memory=False)

    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: Loading Spotify API data...")
    logger.info("=" * 80)
    try:
        api_df = pd.read_csv(spotify_csv, low_memory=False)
    except Exception as exc:
        logger.error("Could not load spotify data: %s", exc)
        logger.error("Continuing with Kaggle-only dataset.")
        api_df = pd.DataFrame(columns=kaggle_df.columns)

    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 3: Merging datasets...")
    logger.info("=" * 80)
    merged_df = merge_datasets(kaggle_df, api_df)

    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 4: Saving merged dataset...")
    logger.info("=" * 80)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    logger.info("Saved to %s", output_path)
    logger.info("Final shape: %d rows x %d columns", len(merged_df), len(merged_df.columns))

    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Kaggle and Spotify datasets")
    parser.add_argument("--kaggle-csv", type=str, help="Kaggle CSV path")
    parser.add_argument("--spotify-csv", type=str, help="Spotify API CSV path")
    parser.add_argument("--output-csv", type=str, help="Output CSV path")
    args = parser.parse_args()

    run(
        kaggle_csv=args.kaggle_csv,
        spotify_csv=args.spotify_csv,
        output_csv=args.output_csv,
    )
