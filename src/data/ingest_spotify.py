"""Fetch Spotify API data for recent years (2021-2025)."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_MAX_TRACKS_PER_YEAR = 200
API_MAX_SEARCH_RESULTS_PER_QUERY = 50

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


class SpotifyAccessBlocked(RuntimeError):
    """Raised when Spotify blocks audio-features access with a 403."""


def authenticate_spotify(
    client_id: str | None = None,
    client_secret: str | None = None,
) -> spotipy.Spotify:
    """Authenticate with Spotify using Client Credentials flow."""
    client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in environment or passed as args"
        )

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        cache_handler=None,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def safe_artists_to_string(artists_payload: List[Dict]) -> str:
    """Convert Spotify artists payload to a comma-separated string."""
    if not artists_payload:
        return "Unknown"
    return ", ".join(artist.get("name", "Unknown") for artist in artists_payload)


def fetch_year_track_candidates(
    sp: spotipy.Spotify,
    year: int,
    market: str = "US",
    max_search_results_per_query: int = API_MAX_SEARCH_RESULTS_PER_QUERY,
    query_seeds: List[str] | None = None,
) -> List[Dict]:
    """Fetch track candidates for a given year using Spotify search."""
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
                    logger.warning("Rate limited on search query '%s'. Skipping remaining pages.", query)
                    break
                raise

            items = response.get("tracks", {}).get("items", [])
            if not items:
                break

            candidates.extend(items)
            offset += limit
            time.sleep(0.1)

    return candidates


def fetch_audio_features_with_retry(
    sp: spotipy.Spotify,
    track_ids: List[str],
    max_retries: int = 5,
    base_wait: int = 60,
) -> Dict[str, Dict]:
    """
    Fetch audio features with retry logic.

    On 429, back off and retry.
    On 403, stop the run by raising SpotifyAccessBlocked.
    """
    features_by_id: Dict[str, Dict] = {}
    batch_size = 20

    for i in range(0, len(track_ids), batch_size):
        batch_ids = track_ids[i : i + batch_size]
        features_batch = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    "Fetching %d features (attempt %d/%d)",
                    len(batch_ids),
                    attempt,
                    max_retries,
                )
                features_batch = sp.audio_features(batch_ids)
                break
            except SpotifyException as exc:
                status = getattr(exc, "http_status", None)

                if status == 429:
                    retry_after = base_wait * (2 ** (attempt - 1))
                    try:
                        retry_after = int(exc.headers.get("Retry-After", retry_after))
                    except Exception:
                        pass
                    logger.warning(
                        "Rate limited (attempt %d/%d). Waiting %ss...",
                        attempt,
                        max_retries,
                        retry_after,
                    )
                    time.sleep(retry_after + random.uniform(1.0, 5.0))
                    continue

                if status in (500, 502, 503, 504):
                    wait = 5.0 * attempt
                    logger.warning(
                        "Server error %s (attempt %d). Retrying in %.1fs...",
                        status,
                        attempt,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                if status == 403:
                    raise SpotifyAccessBlocked(
                        "Spotify returned 403 on audio_features. Stopping API fetch for this run."
                    ) from exc

                raise

        if features_batch is None:
            logger.warning("Failed after %d attempts. Treating batch as missing.", max_retries)
            features_batch = [None] * len(batch_ids)

        for fid, feature in zip(batch_ids, features_batch):
            if feature and feature.get("id"):
                features_by_id[feature["id"]] = feature

        time.sleep(0.5)

    return features_by_id


def build_api_dataframe_for_year(
    sp: spotipy.Spotify,
    year: int,
    tracks_per_year: int,
    market: str = "US",
    max_search_results_per_query: int = API_MAX_SEARCH_RESULTS_PER_QUERY,
    query_seeds: List[str] | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of top tracks for one year from Spotify API."""
    logger.info("Fetching tracks for year %d", year)

    candidates = fetch_year_track_candidates(
        sp=sp,
        year=year,
        market=market,
        max_search_results_per_query=max_search_results_per_query,
        query_seeds=query_seeds,
    )
    logger.info("  Found %d candidates", len(candidates))

    unique_tracks: Dict[str, Dict] = {}
    for track in candidates:
        track_id = track.get("id")
        if track_id and track_id not in unique_tracks:
            unique_tracks[track_id] = track

    top_n = min(tracks_per_year, API_MAX_TRACKS_PER_YEAR)
    ranked_tracks = sorted(
        unique_tracks.values(),
        key=lambda t: t.get("popularity", 0),
        reverse=True,
    )[:top_n]

    track_ids = [t["id"] for t in ranked_tracks if t.get("id")]
    logger.info("  Ranking top %d by popularity", len(track_ids))

    if not track_ids:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    audio_features_map = fetch_audio_features_with_retry(sp, track_ids)
    logger.info("  Got audio features for %d/%d tracks", len(audio_features_map), len(track_ids))

    rows = []
    for track in ranked_tracks:
        track_id = track.get("id")
        feature = audio_features_map.get(track_id)

        if not feature:
            continue

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
                "popularity": track.get("popularity"),
            }
        )

    logger.info("  Built %d rows for %d", len(rows), year)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def build_api_dataframe(
    api_start_year: int = 2021,
    api_end_year: int = 2025,
    tracks_per_year: int = 7500,
    market: str = "US",
    max_search_results_per_query: int = API_MAX_SEARCH_RESULTS_PER_QUERY,
    query_seeds: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> pd.DataFrame:
    """Fetch and build complete API dataset for a year range."""
    sp = authenticate_spotify(client_id, client_secret)

    default_seeds = ",a,e,i,o,u,the,love,feat,remix,radio,live"
    seeds_str = query_seeds or default_seeds
    query_seeds_list = [seed.strip() for seed in seeds_str.split(",") if seed.strip()]
    query_seeds_list = [""] + query_seeds_list

    api_frames = []
    for year in range(api_start_year, api_end_year + 1):
        try:
            year_df = build_api_dataframe_for_year(
                sp=sp,
                year=year,
                tracks_per_year=tracks_per_year,
                market=market,
                max_search_results_per_query=max_search_results_per_query,
                query_seeds=query_seeds_list,
            )
        except SpotifyAccessBlocked as exc:
            logger.error("%s", exc)
            logger.error("Stopping API ingestion early at year %d.", year)
            break

        api_frames.append(year_df)

    api_df = (
        pd.concat(api_frames, ignore_index=True)
        if api_frames
        else pd.DataFrame(columns=REQUIRED_COLUMNS)
    )
    logger.info("Total API data: %d rows", len(api_df))
    return api_df


def run(
    output_csv: str | Path | None = None,
    api_start_year: int = 2021,
    api_end_year: int = 2025,
    tracks_per_year: int = 7500,
) -> pd.DataFrame:
    """Fetch and save Spotify API dataset."""
    logging.basicConfig(level=logging.INFO)

    api_df = build_api_dataframe(
        api_start_year=api_start_year,
        api_end_year=api_end_year,
        tracks_per_year=tracks_per_year,
    )

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        api_df.to_csv(output_path, index=False)
        logger.info("Saved to %s", output_path)

    return api_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Spotify API dataset (2021-2025)")
    parser.add_argument("--output-csv", type=str, help="Output CSV path")
    parser.add_argument("--api-start-year", type=int, default=2021)
    parser.add_argument("--api-end-year", type=int, default=2025)
    parser.add_argument("--tracks-per-year", type=int, default=7500)
    args = parser.parse_args()

    run(
        output_csv=args.output_csv,
        api_start_year=args.api_start_year,
        api_end_year=args.api_end_year,
        tracks_per_year=args.tracks_per_year,
    )