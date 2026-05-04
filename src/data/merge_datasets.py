"""Merge Kaggle and Spotify datasets into one unified table."""

import logging
from pathlib import Path

from . import merge_spotify_trends

logger = logging.getLogger(__name__)


def run(
    kaggle_csv: str | None = None,
    output_csv: str | None = None,
    api_start_year: int = 2021,
    api_end_year: int = 2025,
) -> None:
    """
    Run the merge pipeline.
    
    Args:
        kaggle_csv: Path to Kaggle CSV (defaults to data/interim/kaggle_1960_2020.csv)
        output_csv: Output path (defaults to data/processed/Spotify_1960_2026_Final.csv)
        api_start_year: Start year for API fetch (default 2021)
        api_end_year: End year for API fetch (default 2025)
    """
    project_root = Path(__file__).resolve().parents[2]
    
    if kaggle_csv is None:
        kaggle_csv = str(project_root / "data/interim/kaggle_1960_2020.csv")
    
    if output_csv is None:
        output_csv = str(project_root / "data/processed/Spotify_1960_2026_Final.csv")
    
    logger.info("Starting dataset merge...")
    merge_spotify_trends.build_merged_dataset(
        kaggle_csv=kaggle_csv,
        output_csv=output_csv,
        api_start_year=api_start_year,
        api_end_year=api_end_year,
    )
    logger.info("Dataset merge completed.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run()