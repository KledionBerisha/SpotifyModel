import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Columns required for the final merged dataset.
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
]

logger = logging.getLogger(__name__)


def authenticate_spotify(client_id: str, client_secret: str) -> spotipy.Spotify:
    """Authenticate with Spotify using Client Credentials flow."""
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        cache_handler=None,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def safe_artists_to_string(artists_payload: List[Dict]) -> str:
    """Convert Spotify artists payload to a comma-separated artist string."""
    if not artists_payload:
        return "Unknown"
    return ", ".join(artist.get("name", "Unknown") for artist in artists_payload)


def fetch_year_track_candidates(
    sp: spotipy.Spotify,
    year: int,
    market: str = "US",
    max_search_results_per_query: int = 200,
    query_seeds: List[str] | None = None,
) -> List[Dict]:
    """Fetch track candidates for a given year using Spotify Search."""
    candidates: List[Dict] = []
    limit = 10
    seeds = query_seeds or [""]

    for seed in seeds:
        query = f"year:{year} {seed}".strip()
        offset = 0
        while offset < max_search_results_per_query:
            try:
                response = sp.search(
                    q=query,
                    type="track",
                    market=market,
                    limit=limit,
                    offset=offset,
                )
            except SpotifyException as exc:
                if exc.http_status == 429:
                    logger.warning(f"Rate limited on query '{query}'. Skipping remaining pages.")
                    break
                raise

            items = response.get("tracks", {}).get("items", [])
            if not items:
                break

            candidates.extend(items)
            offset += limit
            time.sleep(0.05 + random.uniform(0.0, 0.05))

    return candidates


def fetch_audio_features_map(sp: spotipy.Spotify, track_ids: List[str]) -> Dict[str, Dict]:
    """Fetch audio features in batches and map them by track id."""
    features_by_id: Dict[str, Dict] = {}
    batch_size = 100

    for i in range(0, len(track_ids), batch_size):
        batch_ids = track_ids[i : i + batch_size]
        try:
            features_batch = sp.audio_features(batch_ids)
        except SpotifyException as exc:
            if exc.http_status == 403:
                logger.warning("audio-features endpoint returned 403. Continuing with missing values.")
                return features_by_id
            raise
        for feature in features_batch:
            if feature and feature.get("id"):
                features_by_id[feature["id"]] = feature
        time.sleep(0.05)

    return features_by_id


def build_api_dataframe_for_year(
    sp: spotipy.Spotify,
    year: int,
    tracks_per_year: int,
    market: str,
    max_search_results_per_query: int,
    query_seeds: List[str],
) -> pd.DataFrame:
    """Build a DataFrame of top tracks for one year from Spotify API."""
    candidates = fetch_year_track_candidates(
        sp=sp,
        year=year,
        market=market,
        max_search_results_per_query=max_search_results_per_query,
        query_seeds=query_seeds,
    )

    unique_tracks: Dict[str, Dict] = {}
    for track in candidates:
        track_id = track.get("id")
        if track_id and track_id not in unique_tracks:
            unique_tracks[track_id] = track

    ranked_tracks = sorted(
        unique_tracks.values(),
        key=lambda t: t.get("popularity", 0),
        reverse=True,
    )[:tracks_per_year]

    track_ids = [track["id"] for track in ranked_tracks if track.get("id")]
    audio_features_map = fetch_audio_features_map(sp, track_ids)

    rows = []
    for track in ranked_tracks:
        track_id = track.get("id")
        feature = audio_features_map.get(track_id, {})
        rows.append(
            {
                "artists": safe_artists_to_string(track.get("artists", [])),
                "name": track.get("name"),
                "duration_ms": track.get("duration_ms"),
                "year": year,
                "acousticness": feature.get("acousticness"),
                "danceability": feature.get("danceability"),
                "energy": feature.get("energy"),
                "instrumentalness": feature.get("instrumentalness"),
                "liveness": feature.get("liveness"),
                "loudness": feature.get("loudness"),
                "speechiness": feature.get("speechiness"),
                "tempo": feature.get("tempo"),
                "valence": feature.get("valence"),
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_and_prepare_kaggle_data(
    csv_path: str,
    start_year: int = 1980,
    end_year: int = 2020,
) -> pd.DataFrame:
    """Load Kaggle CSV, keep years in range, and align columns to REQUIRED_COLUMNS."""
    kaggle_df = pd.read_csv(csv_path)
    kaggle_df["year"] = pd.to_numeric(kaggle_df["year"], errors="coerce")
    kaggle_df = kaggle_df[(kaggle_df["year"] >= start_year) & (kaggle_df["year"] <= end_year)].copy()

    for col in REQUIRED_COLUMNS:
        if col not in kaggle_df.columns:
            kaggle_df[col] = np.nan

    kaggle_df = kaggle_df[REQUIRED_COLUMNS].copy()
    kaggle_df["year"] = kaggle_df["year"].astype("Int64")
    return kaggle_df


def infer_tracks_per_year(kaggle_df: pd.DataFrame, default_value: int = 2000) -> int:
    """Infer the target top-track count from Kaggle data using the mode of yearly counts."""
    yearly_counts = kaggle_df.groupby("year").size()
    if yearly_counts.empty:
        return default_value
    return int(yearly_counts.mode().iloc[0])


def clean_merged_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged dataset: remove duplicates, handle missing values, fill numeric columns."""
    cleaned = df.copy()

    cleaned.drop_duplicates(subset=["artists", "name", "year"], inplace=True)

    for text_col in ["artists", "name"]:
        cleaned[text_col] = cleaned[text_col].fillna("Unknown")

    numeric_cols = [
        "duration_ms",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ]

    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce").astype("Int64")
    cleaned = cleaned[REQUIRED_COLUMNS].reset_index(drop=True)
    return cleaned

def build_merged_dataset(
    kaggle_csv: str,
    output_csv: str,
    api_start_year: int = 2021,
    api_end_year: int = 2025,
    kaggle_start_year: int = 1980,
    kaggle_end_year: int = 2020,
    client_id: str | None = None,
    client_secret: str | None = None,
    market: str = "US",
    max_search_results_per_query: int = 200,
    tracks_per_year: int | None = None,
    query_seeds: str | None = None,
) -> pd.DataFrame:
    """
    Build merged dataset from Kaggle and Spotify API.
    
    Returns the final cleaned DataFrame and saves to output_csv.
    """
    # Load and prepare Kaggle data
    kaggle_df = load_and_prepare_kaggle_data(
        csv_path=kaggle_csv,
        start_year=kaggle_start_year,
        end_year=kaggle_end_year,
    )
    logger.info(f"Loaded Kaggle data: {len(kaggle_df)} rows")

    # Infer tracks per year
    inferred_tracks = infer_tracks_per_year(kaggle_df)
    target_tracks_per_year = tracks_per_year or inferred_tracks
    logger.info(f"Using tracks_per_year={target_tracks_per_year}")

    # Parse query seeds
    default_seeds = ",a,e,i,o,u,the,love,feat,remix,radio,live"
    seeds_str = query_seeds or default_seeds
    query_seeds_list = [seed.strip() for seed in seeds_str.split(",") if seed.strip() != ""]
    query_seeds_list = [""] + query_seeds_list

    # Fetch Spotify API data
    sp = authenticate_spotify(client_id or os.getenv("SPOTIFY_CLIENT_ID"), 
                               client_secret or os.getenv("SPOTIFY_CLIENT_SECRET"))

    api_frames = []
    for year in range(api_start_year, api_end_year + 1):
        logger.info(f"Fetching Spotify API tracks for year {year}...")
        year_df = build_api_dataframe_for_year(
            sp=sp,
            year=year,
            tracks_per_year=target_tracks_per_year,
            market=market,
            max_search_results_per_query=max_search_results_per_query,
            query_seeds=query_seeds_list,
        )
        logger.info(f"  -> collected {len(year_df)} rows for {year}")
        api_frames.append(year_df)

    api_df = pd.concat(api_frames, ignore_index=True) if api_frames else pd.DataFrame(columns=REQUIRED_COLUMNS)
    logger.info(f"Total API data: {len(api_df)} rows")

    # Merge and clean
    merged_df = pd.concat([kaggle_df, api_df], ignore_index=True)
    final_df = clean_merged_data(merged_df)

    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged dataset: {output_path}")
    logger.info(f"Final shape: {final_df.shape[0]} rows x {final_df.shape[1]} columns")

    return final_df


def build_argument_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments for easy reuse and reproducibility."""
    parser = argparse.ArgumentParser(description="Merge Kaggle Spotify history with Spotify API recent years.")
    parser.add_argument(
        "--kaggle-csv",
        type=str,
        default=str(PROJECT_ROOT / "data/interim/kaggle_1960_2020.csv"),
        help="Path to Kaggle data.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "data/processed/Spotify_1960_2026_Final.csv"),
        help="Output CSV filename",
    )
    parser.add_argument("--client-id", default=os.getenv("SPOTIFY_CLIENT_ID"), help="Spotify Client ID")
    parser.add_argument("--client-secret", default=os.getenv("SPOTIFY_CLIENT_SECRET"), help="Spotify Client Secret")
    parser.add_argument("--kaggle-start-year", type=int, default=1980, help="Kaggle data lower year bound")
    parser.add_argument("--kaggle-end-year", type=int, default=2020, help="Kaggle data upper year bound")
    parser.add_argument("--api-start-year", type=int, default=2021, help="Spotify API lower year bound")
    parser.add_argument("--api-end-year", type=int, default=2025, help="Spotify API upper year bound")
    parser.add_argument("--market", default="US", help="Spotify market code for search")
    parser.add_argument(
        "--max-search-results-per-query",
        type=int,
        default=200,
        help="Maximum Spotify search results to collect per query seed",
    )
    parser.add_argument(
        "--tracks-per-year",
        type=int,
        default=None,
        help="Optional override for number of top tracks collected per API year",
    )
    parser.add_argument(
        "--query-seeds",
        default=None,
        help="Comma-separated query seed tokens combined with year search",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    args = build_argument_parser().parse_args()

    if not args.client_id or not args.client_secret:
        raise ValueError(
            "Missing Spotify credentials. Provide --client-id/--client-secret "
            "or set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables."
        )

    build_merged_dataset(
        kaggle_csv=args.kaggle_csv,
        output_csv=args.output_csv,
        api_start_year=args.api_start_year,
        api_end_year=args.api_end_year,
        kaggle_start_year=args.kaggle_start_year,
        kaggle_end_year=args.kaggle_end_year,
        client_id=args.client_id,
        client_secret=args.client_secret,
        market=args.market,
        max_search_results_per_query=args.max_search_results_per_query,
        tracks_per_year=args.tracks_per_year,
        query_seeds=args.query_seeds,
    )


if __name__ == "__main__":
    main()
