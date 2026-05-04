"""Fetch Spotify data for years 2021-2025 and store raw files.

Note: This module is handled by merge_spotify_trends.py which combines
Kaggle + Spotify API data in one step. For standalone API-only fetches,
use the build_api_dataframe_for_year() function from merge_spotify_trends.
"""

import logging

logger = logging.getLogger(__name__)


def run() -> None:
    """
    Placeholder for standalone Spotify API ingest.
    
    For now, use: src/data/merge_spotify_trends.py
    which handles both Kaggle ingestion and Spotify API fetching.
    """
    raise NotImplementedError(
        "Spotify API ingestion is integrated into merge_spotify_trends.py.\n"
        "Use: python -m src.data.merge_spotify_trends --help"
    )


if __name__ == "__main__":
    run()