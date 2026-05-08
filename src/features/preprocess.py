"""Common preprocessing logic for feature engineering and modeling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data/processed/Spotify_1980_2025_Final_clean.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data/processed/model_input.csv"

AUDIO_FEATURE_COLUMNS = [
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
MODEL_FEATURE_COLUMNS = AUDIO_FEATURE_COLUMNS + ["duration_ms", "year"]
TARGET_COLUMN = "popularity"
IDENTITY_COLUMNS = ["artists", "name"]
NUMERIC_COLUMNS = MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]

logger = logging.getLogger(__name__)


def load_clean_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned Spotify dataset and coerce core numeric columns."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_INPUT_CSV
    df = pd.read_csv(path, low_memory=False)
    return coerce_numeric_columns(df, NUMERIC_COLUMNS)


def coerce_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert columns to numeric values when possible."""
    coerced = df.copy()
    for column in columns:
        if column not in coerced.columns:
            coerced[column] = pd.NA
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def build_model_frame(df: pd.DataFrame, drop_missing_target: bool = True) -> pd.DataFrame:
    """Create a model-ready frame with the standard feature set."""
    frame = coerce_numeric_columns(df, NUMERIC_COLUMNS)

    if drop_missing_target and TARGET_COLUMN in frame.columns:
        frame = frame.dropna(subset=[TARGET_COLUMN])

    frame = frame.dropna(subset=MODEL_FEATURE_COLUMNS).copy()

    for column in IDENTITY_COLUMNS:
        if column not in frame.columns:
            frame[column] = "Unknown"

    selected_columns = IDENTITY_COLUMNS + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]
    return frame[selected_columns].reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a model frame into features and target."""
    features = df[MODEL_FEATURE_COLUMNS].copy()
    target = df[TARGET_COLUMN].copy()
    return features, target


def scale_feature_frame(
    train_features: pd.DataFrame,
    *other_feature_frames: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.DataFrame], StandardScaler]:
    """Scale feature frames using a StandardScaler fit on the training frame."""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=train_features.columns,
        index=train_features.index,
    )

    scaled_others: list[pd.DataFrame] = []
    for feature_frame in other_feature_frames:
        scaled_others.append(
            pd.DataFrame(
                scaler.transform(feature_frame),
                columns=feature_frame.columns,
                index=feature_frame.index,
            )
        )

    return train_scaled, scaled_others, scaler


def run(input_csv: str | None = None, output_csv: str | None = None) -> pd.DataFrame:
    """Build and save a model-ready training dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    source_path = Path(input_csv) if input_csv is not None else DEFAULT_INPUT_CSV
    destination_path = Path(output_csv) if output_csv is not None else DEFAULT_OUTPUT_CSV

    df = load_clean_dataset(source_path)
    model_frame = build_model_frame(df, drop_missing_target=True)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame.to_csv(destination_path, index=False)

    logger.info("Loaded %d rows from %s", len(df), source_path)
    logger.info("Prepared %d rows with non-missing popularity", len(model_frame))
    logger.info("Saved model input to %s", destination_path)
    return model_frame


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare feature tables for modeling and clustering.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to the cleaned Spotify dataset (default: data/processed/Spotify_1980_2025_Final_clean.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to the model-ready CSV (default: data/processed/model_input.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(input_csv=args.input_csv, output_csv=args.output_csv)