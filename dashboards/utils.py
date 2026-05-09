"""Utility functions for Spotify dashboard."""

import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_spotify_data(path: str = None) -> pd.DataFrame:
    """Load Spotify dataset with caching for performance."""
    if path is None:
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / "data/processed/Spotify_1980_2025_Final_clean.csv"
    
    df = pd.read_csv(path)
    # Ensure year is numeric
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    return df


@st.cache_data
def compute_yearly_averages(df: pd.DataFrame, audio_features: list) -> pd.DataFrame:
    """Compute yearly averages for audio features."""
    return df.groupby('year')[audio_features].mean().reset_index()


def get_audio_features() -> list:
    """Return list of audio features to analyze."""
    return ['danceability', 'energy', 'valence', 'acousticness', 'liveness', 'speechiness']


def get_comparison_metrics() -> list:
    """Return list of metrics for comparison plots."""
    return ['popularity', 'duration_ms', 'tempo', 'loudness']
