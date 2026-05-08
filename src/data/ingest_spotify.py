"""Fetch Spotify API data for recent years (2021-2025)."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
import requests
import spotipy
from dotenv import load_dotenv
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

TRACKS_BATCH_SIZE = 50
AUDIO_FEATURES_BATCH_SIZE = 100
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
    """Raised when Spotify blocks access with a 403."""


def chunked(items: Sequence[str], chunk_size: int) -> Iterator[list[str]]:
    """Yield chunks from a sequence."""
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def authenticate_spotify(
    client_id: str | None = None,
    client_secret: str | None = None,
) -> spotipy.Spotify:
    """Authenticate with Spotify.

    Defaults to client-credentials flow for public read access.
    OAuth is optional and can be enabled with SPOTIFY_USE_OAUTH=true.
    """
    client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
    use_oauth = os.getenv("SPOTIFY_USE_OAUTH", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not client_id or not client_secret:
        raise ValueError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in environment or passed as args"
        )

    if use_oauth:
        redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
        if not redirect_uri:
            raise ValueError("SPOTIFY_REDIRECT_URI must be set when SPOTIFY_USE_OAUTH is enabled")

        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-read-private user-read-email",
            cache_path=str(PROJECT_ROOT / ".spotify-cache"),
            open_browser=False,
        )
        return spotipy.Spotify(auth_manager=auth_manager)

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        cache_handler=None,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def safe_artists_to_string(artists_payload: list[dict]) -> str:
    """Convert Spotify artists payload to a comma-separated string."""
    if not artists_payload:
        return "Unknown"
    return ", ".join(artist.get("name", "Unknown") for artist in artists_payload)


def fetch_year_track_candidates(
    sp: spotipy.Spotify,
    year: int,
    market: str | None = None,
    max_search_results_per_query: int = API_MAX_SEARCH_RESULTS_PER_QUERY,
    query_seeds: list[str] | None = None,
) -> list[dict]:
    """Fetch track candidates for a given year using Spotify search."""
    candidates: list[dict] = []
    limit = 10
    seeds = query_seeds or [""]
    max_net_retries = 5

    for seed in seeds:
        query = f"year:{year} {seed}".strip()
        offset = 0

        while offset < max_search_results_per_query:
            search_kwargs = {
                "q": query,
                "type": "track",
                "limit": limit,
                "offset": offset,
            }
            if market:
                search_kwargs["market"] = market

            response = None
            for attempt in range(1, max_net_retries + 1):
                try:
                    response = sp.search(**search_kwargs)
                    break
                except SpotifyException as exc:
                    if exc.http_status == 429:
                        logger.warning("Rate limited on search query '%s'. Skipping remaining pages.", query)
                        response = None
                        break
                    if exc.http_status == 403:
                        raise SpotifyAccessBlocked("Spotify returned 403 on search. Stopping API fetch.") from exc
                    raise
                except requests.exceptions.RequestException as exc:
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        "Network error on search attempt %d/%d for query '%s': %s. Retrying in %.1fs",
                        attempt,
                        max_net_retries,
                        query,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    continue
            else:
                logger.warning("Search failed after %d network retries; skipping this query page.", max_net_retries)

            if not response:
                break

            items = response.get("tracks", {}).get("items", [])
            if not items:
                break

            candidates.extend(items)
            offset += limit
            time.sleep(0.1)

    return candidates


def fetch_tracks_with_retry(
    sp: spotipy.Spotify,
    track_ids: Sequence[str],
    max_retries: int = 5,
    base_wait: int = 60,
    market: str | None = None,
    skip_forbidden: bool = True,
) -> dict[str, dict]:
    """Fetch track metadata in batches using Spotify's plural tracks endpoint.

    On a 403 for a full batch, this function will attempt to isolate forbidden IDs
    by sub-batching and, if necessary, requesting individual IDs. If `skip_forbidden`
    is True (default), forbidden IDs are skipped so ingestion can continue.
    """
    tracks_by_id: dict[str, dict] = {}

    for batch_ids in chunked([track_id for track_id in track_ids if track_id], TRACKS_BATCH_SIZE):
        batch_tracks = None

        for attempt in range(1, max_retries + 1):
            try:
                if market:
                    response = sp.tracks(batch_ids, market=market)
                else:
                    response = sp.tracks(batch_ids)
                batch_tracks = response.get("tracks", []) if isinstance(response, dict) else response
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
                        "Rate limited on tracks batch (attempt %d/%d). Waiting %ss...",
                        attempt,
                        max_retries,
                        retry_after,
                    )
                    time.sleep(retry_after + random.uniform(1.0, 5.0))
                    continue

                if status in (500, 502, 503, 504):
                    wait = 5.0 * attempt
                    logger.warning(
                        "Server error %s on tracks batch (attempt %d). Retrying in %.1fs...",
                        status,
                        attempt,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                if status == 403:
                    logger.error(
                        "403 Forbidden on tracks batch. batch_size=%d, ids=%s",
                        len(batch_ids),
                        ",".join(batch_ids),
                    )
                    if not skip_forbidden:
                        raise SpotifyAccessBlocked(
                            "Spotify returned 403 on tracks endpoint. This can indicate app permissions, blocked access, or a restricted batch."
                        ) from exc

                    # Try to isolate the problem by sub-batching; fall back to individual ids.
                    logger.info("Attempting to isolate forbidden IDs by sub-batching (size=%d).", len(batch_ids))

                    # If batch is already size 1, skip it.
                    if len(batch_ids) == 1:
                        logger.warning("Skipping forbidden track id=%s", batch_ids[0])
                        batch_tracks = [None]
                        break

                    # First try halves, then quarters, then individuals as needed.
                    try:
                        sub_batch_size = max(1, len(batch_ids) // 2)
                        for sub_ids in chunked(batch_ids, sub_batch_size):
                            try:
                                if market:
                                    sub_resp = sp.tracks(sub_ids, market=market)
                                else:
                                    sub_resp = sp.tracks(sub_ids)
                                sub_tracks = sub_resp.get("tracks", []) if isinstance(sub_resp, dict) else sub_resp
                                for t in sub_tracks:
                                    if t and t.get("id"):
                                        tracks_by_id[t["id"]] = t
                                time.sleep(0.1)
                            except SpotifyException as sub_exc:
                                s_status = getattr(sub_exc, "http_status", None)
                                # If sub-batch also 403 and is larger than 1, drill down to individuals
                                if s_status == 403 and len(sub_ids) > 1:
                                    logger.info("Sub-batch of size %d returned 403; drilling down.", len(sub_ids))
                                    for tid in sub_ids:
                                        try:
                                            if market:
                                                single_resp = sp.tracks([tid], market=market)
                                            else:
                                                single_resp = sp.tracks([tid])
                                            single_tracks = single_resp.get("tracks", []) if isinstance(single_resp, dict) else single_resp
                                            if single_tracks and single_tracks[0] and single_tracks[0].get("id"):
                                                tracks_by_id[single_tracks[0]["id"]] = single_tracks[0]
                                        except SpotifyException as single_exc:
                                            ss = getattr(single_exc, "http_status", None)
                                            if ss == 403:
                                                logger.warning("Skipping forbidden track id=%s", tid)
                                                continue
                                            raise
                                elif s_status == 403:
                                    # sub_ids length is 1, skip it
                                    for tid in sub_ids:
                                        logger.warning("Skipping forbidden track id=%s", tid)
                                    continue
                                else:
                                    raise
                    except Exception:
                        logger.exception("Error while isolating forbidden IDs; continuing by skipping problematic items.")
                    # We've attempted to collect what we could via sub-requests; mark batch handled.
                    batch_tracks = []
                    break
                raise
            except requests.exceptions.RequestException as exc:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Network error fetching tracks batch (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue

        if batch_tracks is None:
            logger.warning("Failed after %d attempts on tracks batch. Treating as missing.", max_retries)
            batch_tracks = [None] * len(batch_ids)

        for track in batch_tracks:
            if track and track.get("id"):
                tracks_by_id[track["id"]] = track

        time.sleep(0.25)

    return tracks_by_id


def fetch_audio_features_with_retry(
    sp: spotipy.Spotify,
    track_ids: Sequence[str],
    max_retries: int = 5,
    base_wait: int = 60,
) -> dict[str, dict]:
    """Fetch audio features in batches using Spotify's plural endpoint."""
    features_by_id: dict[str, dict] = {}

    for batch_ids in chunked([track_id for track_id in track_ids if track_id], AUDIO_FEATURES_BATCH_SIZE):
        features_batch = None

        for attempt in range(1, max_retries + 1):
            try:
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
                        "Rate limited on audio_features batch (attempt %d/%d). Waiting %ss...",
                        attempt,
                        max_retries,
                        retry_after,
                    )
                    time.sleep(retry_after + random.uniform(1.0, 5.0))
                    continue

                if status in (500, 502, 503, 504):
                    wait = 5.0 * attempt
                    logger.warning(
                        "Server error %s on audio_features batch (attempt %d). Retrying in %.1fs...",
                        status,
                        attempt,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                if status == 403:
                    logger.error(
                        "403 Forbidden on audio_features batch. batch_size=%d, ids=%s",
                        len(batch_ids),
                        ",".join(batch_ids),
                    )
                    raise SpotifyAccessBlocked(
                        "Spotify returned 403 on audio_features. This can indicate app permissions, blocked access, or a restricted batch."
                    ) from exc

                raise
            except requests.exceptions.RequestException as exc:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Network error fetching audio_features (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue

        if features_batch is None:
            logger.warning("Failed after %d attempts on audio_features batch. Treating as missing.", max_retries)
            features_batch = [None] * len(batch_ids)

        for track_id, feature in zip(batch_ids, features_batch):
            if feature and feature.get("id"):
                features_by_id[feature["id"]] = feature

        time.sleep(0.25)

    return features_by_id


def build_api_dataframe_for_year(
    sp: spotipy.Spotify,
    year: int,
    tracks_per_year: int,
    market: str | None = None,
    max_search_results_per_query: int = API_MAX_SEARCH_RESULTS_PER_QUERY,
    query_seeds: list[str] | None = None,
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

    unique_tracks: dict[str, dict] = {}
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

    track_ids = [track["id"] for track in ranked_tracks if track.get("id")]
    logger.info("  Ranking top %d by popularity", len(track_ids))

    if not track_ids:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    track_details_map = fetch_tracks_with_retry(sp, track_ids, market=market)
    audio_features_map = fetch_audio_features_with_retry(sp, track_ids)

    logger.info(
        "  Got track details for %d/%d and audio features for %d/%d tracks",
        len(track_details_map),
        len(track_ids),
        len(audio_features_map),
        len(track_ids),
    )

    rows = []
    for track_id in track_ids:
        track = track_details_map.get(track_id)
        feature = audio_features_map.get(track_id)

        if not track or not feature:
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
    market: str | None = None,
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
    market: str | None = None,
) -> pd.DataFrame:
    """Fetch and save Spotify API dataset."""
    logging.basicConfig(level=logging.INFO)

    api_df = build_api_dataframe(
        api_start_year=api_start_year,
        api_end_year=api_end_year,
        tracks_per_year=tracks_per_year,
        market=market,
    )

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        api_df.to_csv(output_path, index=False)
        logger.info("Saved to %s", output_path)

    return api_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Spotify API dataset (2021-2025)")
    parser.add_argument("--output-csv", type=str, help="Output CSV path")
    parser.add_argument("--api-start-year", type=int, default=2021)
    parser.add_argument("--api-end-year", type=int, default=2025)
    parser.add_argument("--tracks-per-year", type=int, default=7500)
    parser.add_argument("--market", type=str, default=None, help="Optional Spotify market code")
    args = parser.parse_args()

    run(
        output_csv=args.output_csv,
        api_start_year=args.api_start_year,
        api_end_year=args.api_end_year,
        tracks_per_year=args.tracks_per_year,
        market=args.market,
    )