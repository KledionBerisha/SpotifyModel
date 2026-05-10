#!/usr/bin/env python3
"""
Fetch Spotify API data for recent years (2021-2025) and build Kaggle-style datasets.
Also supports fixing missing track popularity in existing CSVs.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load variables from .env file automatically
load_dotenv(PROJECT_ROOT / ".env")

TRACK_COLUMNS = [
    "id", "name", "popularity", "duration_ms", "explicit", "artists",
    "id_artists", "release_date", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature"
]

ARTIST_COLUMNS = ["id", "followers", "genres", "name", "popularity"]

AUDIO_COLUMNS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature"
]

class SpotifyApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

def literal_list(values: list[str]) -> str:
    return repr(values)

def cache_load(path: Path) -> Any | None:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

def cache_save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def batch_cache_path(cache_dir: Path, namespace: str, ids: list[str]) -> Path:
    digest = hashlib.sha1(",".join(ids).encode("utf-8")).hexdigest()
    return cache_dir / namespace / f"{digest}.json"

@dataclass
class SpotifyClient:
    client_id: str
    client_secret: str
    token: str | None = None
    token_expires_at: float = 0

    def authenticate(self) -> None:
        auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {auth}"},
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        if response.status_code >= 400:
            raise SpotifyApiError(f"Spotify auth failed: {response.status_code} {response.text}", response.status_code)
        payload = response.json()
        self.token = payload["access_token"]
        self.token_expires_at = time.time() + int(payload.get("expires_in", 3600)) - 60

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.token or time.time() >= self.token_expires_at:
            self.authenticate()

        url = f"https://api.spotify.com/v1/{path.lstrip('/')}"
        for attempt in range(8):
            response = requests.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=45,
            )

            if response.status_code == 401 and attempt == 0:
                self.authenticate()
                continue

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "2"))
                logger.warning(f"Rate limited. Waiting {retry_after}s.")
                time.sleep(retry_after + 1)
                continue

            if response.status_code in {500, 502, 503, 504}:
                time.sleep(2**attempt)
                continue

            if response.status_code >= 400:
                raise SpotifyApiError(f"Spotify request failed for {path}: {response.status_code} {response.text}", response.status_code)

            return response.json()
        raise SpotifyApiError(f"Spotify request failed after retries for {path}")

def search_tracks_for_year(sp: SpotifyClient, year: int, market: str, max_search_results: int, cache_dir: Path) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"search_{year}_{market}_{max_search_results}.json"
    cached = cache_load(cache_path)
    if cached is not None:
        return cached

    tracks: dict[str, dict[str, Any]] = {}
    limit = 10
    max_offset = min(max_search_results, 1000)

    for offset in range(0, max_offset, limit):
        payload = sp.get("search", {"q": f"year:{year}", "type": "track", "limit": limit, "offset": offset, "market": market})
        items = payload.get("tracks", {}).get("items", [])
        if not items:
            break
        for item in items:
            release_date = item.get("album", {}).get("release_date", "")
            if release_date.startswith(str(year)):
                tracks[item["id"]] = item
        time.sleep(0.15)

    result = sorted(tracks.values(), key=lambda t: t.get("popularity", 0), reverse=True)
    cache_save(cache_path, result)
    return result

def hydrate_track_details(sp: SpotifyClient, tracks: list[dict[str, Any]], market: str, max_candidates: int, cache_dir: Path) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    needs_hydration = any(track.get("popularity") is None for track in tracks)
    if not needs_hydration:
        return tracks

    tracks_to_hydrate = tracks[:max_candidates]
    for track in tracks_to_hydrate:
        track_id = track["id"]
        cache_path = cache_dir / "tracks" / f"{track_id}.json"
        cached = cache_load(cache_path)
        if cached is None:
            try:
                cached = sp.get(f"tracks/{track_id}", {"market": market})
                cache_save(cache_path, cached)
                time.sleep(0.05)
            except SpotifyApiError:
                cached = track
        hydrated.append(cached or track)

    return sorted(hydrated, key=lambda t: t.get("popularity", 0) or 0, reverse=True)

def fetch_artists(sp: SpotifyClient, artist_ids: list[str], cache_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for batch in batched(sorted(set(artist_ids)), 50):
        cache_path = batch_cache_path(cache_dir, "artists", batch)
        cached = cache_load(cache_path)
        if cached is None:
            payload = sp.get("artists", {"ids": ",".join(batch)})
            cached = payload.get("artists", [])
            cache_save(cache_path, cached)
            time.sleep(0.15)
        for artist in cached:
            if artist:
                result[artist["id"]] = artist
    return result

def fetch_artists_individually(sp: SpotifyClient, artist_ids: list[str], cache_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for artist_id in sorted(set(artist_ids)):
        cache_path = cache_dir / "artists_individual" / f"{artist_id}.json"
        cached = cache_load(cache_path)
        if cached is None:
            try:
                cached = sp.get(f"artists/{artist_id}")
                cache_save(cache_path, cached)
                time.sleep(0.05)
            except SpotifyApiError:
                cached = None
        if cached:
            result[artist_id] = cached
    return result

def fetch_spotify_audio_features(sp: SpotifyClient, track_ids: list[str], cache_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for batch in batched(track_ids, 100):
        cache_path = batch_cache_path(cache_dir, "spotify_audio_features", batch)
        cached = cache_load(cache_path)
        if cached is None:
            payload = sp.get("audio-features", {"ids": ",".join(batch)})
            cached = payload.get("audio_features", [])
            cache_save(cache_path, cached)
            time.sleep(0.15)
        for features in cached:
            if features:
                result[features["id"]] = features
    return result

def fetch_reccobeats_audio_features(track_ids: list[str], cache_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for batch in batched(track_ids, 40):
        cache_path = batch_cache_path(cache_dir, "reccobeats_audio_features", batch)
        cached = cache_load(cache_path)
        if cached is None:
            response = requests.get(
                "https://api.reccobeats.com/v1/audio-features",
                params={"ids": ",".join(batch)},
                headers={"Accept": "application/json"},
                timeout=45,
            )
            if response.status_code == 429:
                time.sleep(int(response.headers.get("Retry-After", "2")) + 1)
                response = requests.get(
                    "https://api.reccobeats.com/v1/audio-features",
                    params={"ids": ",".join(batch)},
                    headers={"Accept": "application/json"},
                    timeout=45,
                )
            response.raise_for_status()
            cached = response.json().get("content", [])
            cache_save(cache_path, cached)
            time.sleep(1.0)
        for features in cached:
            marker = "/track/"
            href = features.get("href", "")
            if marker in href:
                spotify_id = href.split(marker, 1)[1].split("?", 1)[0].strip()
                if spotify_id:
                    result[spotify_id] = features
    return result

def normalize_audio_features(features: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if not features:
        return {column: None for column in AUDIO_COLUMNS}
    normalized = {column: features.get(column) for column in AUDIO_COLUMNS}
    if source == "reccobeats" and normalized.get("time_signature") is None:
        normalized["time_signature"] = 4
    return normalized

def build_track_row(track: dict[str, Any], audio_features: dict[str, Any]) -> dict[str, Any]:
    artists = track.get("artists", [])
    row = {
        "id": track.get("id"),
        "name": track.get("name"),
        "popularity": int(track.get("popularity") or 0),
        "duration_ms": int(track.get("duration_ms") or 0),
        "explicit": int(bool(track.get("explicit"))),
        "artists": literal_list([artist.get("name", "") for artist in artists]),
        "id_artists": literal_list([artist.get("id", "") for artist in artists]),
        "release_date": track.get("album", {}).get("release_date", ""),
    }
    row.update(audio_features)
    return {column: row.get(column) for column in TRACK_COLUMNS}

def artist_stubs_from_tracks(tracks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stubs: dict[str, dict[str, Any]] = {}
    for track in tracks:
        for artist in track.get("artists", []):
            artist_id = artist.get("id")
            if artist_id:
                stubs.setdefault(artist_id, {"id": artist_id, "followers": {"total": 0}, "genres": [], "name": artist.get("name", ""), "popularity": 0})
    return stubs

def build_dataset(args, sp: SpotifyClient):
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_tracks: list[dict[str, Any]] = []

    for year in range(args.api_start_year, args.api_end_year + 1):
        candidates = search_tracks_for_year(sp, year, args.market, args.max_search_results, args.cache_dir)
        candidates = hydrate_track_details(sp, candidates, args.market, args.max_hydrate_candidates, args.cache_dir)
        chosen = candidates[: args.top_n]
        logger.info(f"{year}: selected {len(chosen)} tracks from {len(candidates)} candidates")
        selected_tracks.extend(chosen)

    track_ids = [track["id"] for track in selected_tracks]
    audio_source_used = args.audio_source

    if args.audio_source in {"auto", "spotify"}:
        try:
            audio = fetch_spotify_audio_features(sp, track_ids, args.cache_dir)
            audio_source_used = "spotify"
        except SpotifyApiError as exc:
            if args.audio_source == "spotify" or exc.status_code != 403:
                raise
            logger.info("Spotify audio-features returned 403; falling back to ReccoBeats.")
            audio = fetch_reccobeats_audio_features(track_ids, args.cache_dir)
            audio_source_used = "reccobeats"
    else:
        audio = fetch_reccobeats_audio_features(track_ids, args.cache_dir)

    track_rows = []
    for track in selected_tracks:
        features = normalize_audio_features(audio.get(track["id"]), audio_source_used)
        track_rows.append(build_track_row(track, features))

    tracks_df = pd.DataFrame(track_rows, columns=TRACK_COLUMNS)
    tracks_df.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote tracks to {args.output_csv}")

    artist_ids = []
    for value in tracks_df["id_artists"]:
        artist_ids.extend([str(item) for item in ast.literal_eval(value)])

    artist_stubs = artist_stubs_from_tracks(selected_tracks)
    try:
        artist_data = fetch_artists(sp, artist_ids, args.cache_dir)
        artist_data = {**artist_stubs, **artist_data}
    except SpotifyApiError as exc:
        if exc.status_code != 403:
            raise
        logger.info("Spotify batch artists returned 403; trying individuals.")
        artist_data = {**artist_stubs, **fetch_artists_individually(sp, artist_ids, args.cache_dir)}

    artist_rows = sorted([{"id": a.get("id"), "followers": float(a.get("followers", {}).get("total") or 0), "genres": literal_list(a.get("genres") or []), "name": a.get("name"), "popularity": int(a.get("popularity") or 0)} for a in artist_data.values()], key=lambda row: row["id"] or "")
    
    artists_df = pd.DataFrame(artist_rows, columns=ARTIST_COLUMNS)
    artists_path = args.output_csv.parent / f"artists_{args.api_start_year}_{args.api_end_year}.csv"
    artists_df.to_csv(artists_path, index=False)
    logger.info(f"Wrote artists to {artists_path}")

def fix_popularity(args, sp: SpotifyClient):
    if not args.output_csv.exists():
        logger.error(f"Cannot fix popularity. File {args.output_csv} does not exist.")
        return
    
    df = pd.read_csv(args.output_csv)
    fixed = 0
    missing = []
    
    for i, track_id in enumerate(df["id"].astype(str).tolist()):
        if i % 25 == 0:
            logger.info(f"Processing {i}/{len(df)} tracks")

        cache_path = args.cache_dir / "tracks" / f"{track_id}.json"
        track = cache_load(cache_path)
        if track is None:
            try:
                track = sp.get(f"tracks/{track_id}", {"market": args.market})
                cache_save(cache_path, track)
                time.sleep(0.12)
            except SpotifyApiError:
                missing.append(track_id)
                continue
        
        popularity = track.get("popularity")
        if popularity is not None:
            df.loc[i, "popularity"] = int(popularity)
            fixed += 1
        else:
            missing.append(track_id)

    df["popularity"] = df["popularity"].fillna(0).astype(int)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Updated popularity for {fixed} tracks in {args.output_csv}")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Fetch or fix Spotify API dataset.")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "data/raw/spotify/tracks_2021_2025.csv")
    parser.add_argument("--api-start-year", type=int, default=2021)
    parser.add_argument("--api-end-year", type=int, default=2025)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--market", default="US")
    parser.add_argument("--max-search-results", type=int, default=1000)
    parser.add_argument("--max-hydrate-candidates", type=int, default=250)
    parser.add_argument("--audio-source", choices=["auto", "spotify", "reccobeats"], default="auto")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache_spotify_dataset")
    parser.add_argument("--fix-csv", action="store_true", help="Fix track popularity in output-csv instead of fetching new data.")
    args = parser.parse_args()

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env or environment.")

    sp = SpotifyClient(client_id=client_id, client_secret=client_secret)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.fix_csv:
        fix_popularity(args, sp)
    else:
        build_dataset(args, sp)

if __name__ == "__main__":
    main()
