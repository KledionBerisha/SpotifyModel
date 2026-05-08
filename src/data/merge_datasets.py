"""Merge Kaggle and Spotify API datasets into unified table."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import ingest_kaggle, ingest_spotify

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


def merge_datasets(
    kaggle_df: pd.DataFrame,
    api_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Kaggle and Spotify API datasets."""
    logger.info(f"Merging: Kaggle ({len(kaggle_df)} rows) + API ({len(api_df)} rows)")
    merged = pd.concat([kaggle_df, api_df], ignore_index=True)
    logger.info(f"Merged: {len(merged)} total rows")
    return merged


def run(
    kaggle_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    api_start_year: int = 2021,
    api_end_year: int = 2025,
    tracks_per_year: int = 7500,
) -> pd.DataFrame:
    """
    Run full merge pipeline.
    
    Args:
        kaggle_csv: Path to Kaggle CSV
        output_csv: Path to save merged output
        api_start_year: API fetch start year
        api_end_year: API fetch end year
        tracks_per_year: Tracks to fetch per year
        
    Returns:
        Merged DataFrame
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    if kaggle_csv is None:
        kaggle_csv = PROJECT_ROOT / "data/interim/kaggle_1960_2020.csv"
    
    if output_csv is None:
        output_csv = PROJECT_ROOT / "data/processed/Spotify_1960_2026_Final.csv"
    
    # Step 1: Load Kaggle
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Kaggle data...")
    logger.info("=" * 80)
    kaggle_df = ingest_kaggle.load_and_prepare_kaggle_data(kaggle_csv)
    
    # Step 2: Fetch Spotify API
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: Fetching Spotify API data...")
    logger.info("=" * 80)
    api_df = ingest_spotify.build_api_dataframe(
        api_start_year=api_start_year,
        api_end_year=api_end_year,
        tracks_per_year=tracks_per_year,
    )
    
    # Step 3: Merge
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 3: Merging datasets...")
    logger.info("=" * 80)
    merged_df = merge_datasets(kaggle_df, api_df)
    
    # Step 4: Save
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 4: Saving merged dataset...")
    logger.info("=" * 80)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"✓ Final shape: {len(merged_df)} rows × {len(merged_df.columns)} columns")
    
    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Kaggle and Spotify datasets")
    parser.add_argument("--kaggle-csv", type=str, help="Kaggle CSV path")
    parser.add_argument("--output-csv", type=str, help="Output CSV path")
    parser.add_argument("--api-start-year", type=int, default=2021)
    parser.add_argument("--api-end-year", type=int, default=2025)
    parser.add_argument("--tracks-per-year", type=int, default=7500)
    args = parser.parse_args()
    
    run(
        kaggle_csv=args.kaggle_csv,
        output_csv=args.output_csv,
        api_start_year=args.api_start_year,
        api_end_year=args.api_end_year,
        tracks_per_year=args.tracks_per_year,
    )