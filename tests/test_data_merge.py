"""Tests for data merge pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import merge_spotify_trends


@pytest.fixture
def sample_kaggle_csv():
    """Create a minimal Kaggle CSV for testing."""
    data = {
        "artists": ["Artist A", "Artist B", "Artist C"],
        "name": ["Song 1", "Song 2", "Song 3"],
        "duration_ms": [200000, 180000, 220000],
        "year": [2000, 2010, 2015],
        "acousticness": [0.5, 0.6, 0.4],
        "danceability": [0.7, 0.6, 0.8],
        "energy": [0.6, 0.7, 0.5],
        "instrumentalness": [0.0, 0.1, 0.05],
        "liveness": [0.1, 0.2, 0.15],
        "loudness": [-5.0, -6.0, -4.0],
        "speechiness": [0.03, 0.04, 0.02],
        "tempo": [120.0, 100.0, 130.0],
        "valence": [0.7, 0.6, 0.8],
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame(data)
        df.to_csv(f.name, index=False)
        yield f.name
    
    Path(f.name).unlink()


def test_load_and_prepare_kaggle_data(sample_kaggle_csv):
    """Test loading and preparing Kaggle data."""
    df = merge_spotify_trends.load_and_prepare_kaggle_data(
        csv_path=sample_kaggle_csv,
        start_year=2000,
        end_year=2020,
    )
    
    assert len(df) == 3
    assert list(df.columns) == merge_spotify_trends.REQUIRED_COLUMNS
    assert df["year"].dtype == "Int64"


def test_infer_tracks_per_year(sample_kaggle_csv):
    """Test inferring tracks per year from Kaggle data."""
    df = merge_spotify_trends.load_and_prepare_kaggle_data(sample_kaggle_csv)
    
    tracks_per_year = merge_spotify_trends.infer_tracks_per_year(df, default_value=100)
    
    # Each year appears once, so mode should return 1
    assert tracks_per_year == 1


def test_clean_merged_data(sample_kaggle_csv):
    """Test cleaning merged data."""
    df = merge_spotify_trends.load_and_prepare_kaggle_data(sample_kaggle_csv)
    
    # Add a duplicate row
    df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    
    cleaned = merge_spotify_trends.clean_merged_data(df_with_dup)
    
    # Should have removed the duplicate
    assert len(cleaned) == len(df)
    # All required columns should be present
    assert list(cleaned.columns) == merge_spotify_trends.REQUIRED_COLUMNS


def test_safe_artists_to_string():
    """Test artists payload conversion."""
    artists_payload = [
        {"name": "Artist 1"},
        {"name": "Artist 2"},
    ]
    
    result = merge_spotify_trends.safe_artists_to_string(artists_payload)
    assert result == "Artist 1, Artist 2"
    
    # Test empty case
    result = merge_spotify_trends.safe_artists_to_string([])
    assert result == "Unknown"


def test_required_columns_constant():
    """Verify REQUIRED_COLUMNS is properly defined."""
    expected_cols = [
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
    assert merge_spotify_trends.REQUIRED_COLUMNS == expected_cols