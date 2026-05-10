#!/usr/bin/env python3
"""
Finalize cleaning for processed Spotify CSV.

Usage:
  python src/data/finalize_cleaning.py --input data/processed/Spotify_1980_2025_Final.csv --output data/processed/Spotify_1980_2025_Final_clean.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

NUMERIC_COLS = [
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
TEXT_COLS = ["artists", "name"]
YEAR_MIN = 1980
YEAR_MAX = 2025


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Year handling: coerce and filter range
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df = df[df["year"].notna()]
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    # Drop exact duplicates
    if {"artists", "name", "year"}.issubset(df.columns):
        df.drop_duplicates(subset=["artists", "name", "year"], inplace=True)
    else:
        df = df.drop_duplicates().copy()

    # Text cols
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str).str.strip()
        else:
            df[c] = "Unknown"

    # Try to normalize artist lists like "['A', 'B']" -> "A, B"
    import ast
    import re

    def parse_artists(val):
        if pd.isna(val):
            return "Unknown"
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = ast.literal_eval(s)
                if isinstance(arr, (list, tuple)):
                    return ", ".join(str(x) for x in arr)
            except Exception:
                pass
        # remove surrounding brackets/quotes if present
        s2 = re.sub(r"^[\[\]'\"]+|[\[\]'\"]+$", "", s)
        return s2 or "Unknown"

    if "artists" in df.columns:
        df["artists"] = df["artists"].apply(parse_artists)
    
    api_mask = df["year"].between(2021, YEAR_MAX, inclusive="both")
    
    # Numeric columns: coerce and impute only for pre-2021 rows.
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        median = df.loc[~api_mask, c].median(skipna=True)
        if pd.isna(median):
            median = df[c].median(skipna=True)
        if pd.isna(median):
            median = 0.0
        df.loc[~api_mask, c] = df.loc[~api_mask, c].fillna(median)

    # Drop tracks that are still missing audio features (e.g. the 33 API tracks that couldn't fetch features)
    df = df.dropna(subset=NUMERIC_COLS).copy()

    # Sanity bounds and clipping
    zero_one_cols = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "valence",
    ]
    for c in zero_one_cols:
        if c in df.columns:
            df[c] = df[c].clip(0.0, 1.0)

    if "loudness" in df.columns:
        df["loudness"] = df["loudness"].clip(-60.0, 0.0)

    if "tempo" in df.columns:
        tempo_med = df.loc[~api_mask, "tempo"].median(skipna=True)
        if pd.isna(tempo_med):
            tempo_med = df["tempo"].median(skipna=True)
        if pd.isna(tempo_med):
            tempo_med = 0.0
        df.loc[~api_mask, "tempo"] = df.loc[~api_mask, "tempo"].fillna(tempo_med)
        df.loc[(~api_mask) & ((df["tempo"] < 30) | (df["tempo"] > 250)), "tempo"] = tempo_med

    if "duration_ms" in df.columns:
        dur_med = df.loc[~api_mask, "duration_ms"].median(skipna=True)
        if pd.isna(dur_med):
            dur_med = df["duration_ms"].median(skipna=True)
        if pd.isna(dur_med):
            dur_med = 0.0
        df.loc[~api_mask, "duration_ms"] = df.loc[~api_mask, "duration_ms"].fillna(dur_med)
        df.loc[(~api_mask) & ((df["duration_ms"] < 10000) | (df["duration_ms"] > 3_600_000)), "duration_ms"] = dur_med

    # Handle popularity: coerce and impute for pre-2021 rows only
    if "popularity" not in df.columns:
        df["popularity"] = np.nan
    else:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    
    # Impute popularity for Kaggle rows (pre-2021) only
    pop_median = df.loc[~api_mask, "popularity"].median(skipna=True)
    if pd.isna(pop_median):
        pop_median = df["popularity"].median(skipna=True)
    if not pd.isna(pop_median):
        df.loc[~api_mask, "popularity"] = df.loc[~api_mask, "popularity"].fillna(pop_median)
    
    # Clip popularity to valid range
    if "popularity" in df.columns:
        df["popularity"] = df["popularity"].clip(0, 100)

    # Reorder columns for consistency
    desired = [
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
    cols = [c for c in desired if c in df.columns] + [c for c in df.columns if c not in desired]
    df = df[cols].reset_index(drop=True)

    return df


def summary(df: pd.DataFrame) -> str:
    miss = df.isna().sum().to_dict()
    stats = df[NUMERIC_COLS].describe().loc[["mean", "50%", "std"]]
    lines = [f"Rows: {len(df)}", f"Missing per column: {miss}", "Numeric summary (mean/median/std):"]
    for c in NUMERIC_COLS:
        if c in stats.columns:
            mean = stats.loc["mean", c]
            median = stats.loc["50%", c]
            std = stats.loc["std", c]
            lines.append(f"  {c}: mean={mean:.4f} median={median:.4f} std={std:.4f}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Finalize cleaning for processed Spotify CSV")
    p.add_argument("--input", required=True, help="Input merged CSV path")
    p.add_argument("--output", required=True, help="Output cleaned CSV path")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp, low_memory=False)
    cleaned = clean_df(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    print(summary(cleaned))


if __name__ == "__main__":
    main()