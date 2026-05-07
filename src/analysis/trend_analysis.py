"""Analyze long-term trends in Spotify audio characteristics (1980-2025)."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Project root for relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_END_YEAR = 2025
BUCKET_SIZE_YEARS = 5

logger = logging.getLogger(__name__)

# Audio features to analyze
AUDIO_FEATURES = [
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


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate merged Spotify dataset."""
    df = pd.read_csv(csv_path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    
    # Ensure all audio features are numeric
    for col in AUDIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["year"].between(1980, ANALYSIS_END_YEAR)].copy()

    logger.info(f"Loaded {len(df)} tracks from {csv_path}")
    return df

def add_five_year_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bucket_start = (df["year"] // BUCKET_SIZE_YEARS) * BUCKET_SIZE_YEARS
    bucket_end = np.minimum(bucket_start + (BUCKET_SIZE_YEARS - 1), ANALYSIS_END_YEAR)

    df["bucket_start"] = bucket_start.astype(int)
    df["bucket_label"] = (
        bucket_start.astype(int).astype(str)
        + "-"
        + bucket_end.astype(int).astype(str)
    )
    return df

def compute_period_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5-year period statistics for all audio features.
    """
    bucketed = add_five_year_bucket(df)

    period_stats = bucketed.groupby(
        ["bucket_start", "bucket_label"],
        as_index=False
    )[AUDIO_FEATURES].agg([
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
    ])

    period_stats.columns = [
        "_".join(col).strip() if col[1] else col[0]
        for col in period_stats.columns
    ]

    period_stats.rename(
        columns={
            "bucket_start_": "bucket_start",
            "bucket_label_": "bucket_label",
        },
        inplace=True,
    )

    period_counts = bucketed.groupby(
        ["bucket_start", "bucket_label"]
    ).size().reset_index(name="track_count")

    period_stats = period_stats.merge(
        period_counts,
        on=["bucket_start", "bucket_label"]
    ).sort_values("bucket_start")

    # Keep the old "year" field so the rest of the code can still work
    period_stats["year"] = period_stats["bucket_start"]

    logger.info(f"Computed statistics for {len(period_stats)} 5-year periods")
    return period_stats


def test_trend(years: np.ndarray, values: np.ndarray) -> Tuple[float, float, float]:
    """
    Test for linear trend using linear regression.
    
    Returns:
        (slope, p_value, r_squared)
    """
    # Remove NaN values
    mask = ~(np.isnan(years) | np.isnan(values))
    years_clean = years[mask]
    values_clean = values[mask]
    
    if len(years_clean) < 2:
        return np.nan, np.nan, np.nan
    
    # Linear regression: y ~ x
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_clean, values_clean)
    
    return slope, p_value, r_value ** 2


def analyze_trends(yearly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze trends for each audio feature.
    
    Returns:
        DataFrame with columns: feature, slope, p_value, r_squared, trend_direction
    """
    years = yearly_stats["year"].values
    trends = []
    
    for feature in AUDIO_FEATURES:
        col_mean = f"{feature}_mean"
        if col_mean not in yearly_stats.columns:
            continue
        
        values = yearly_stats[col_mean].values
        slope, p_value, r_squared = test_trend(years, values)
        
        # Determine trend direction (significant if p < 0.05)
        if p_value < 0.05:
            direction = "↑ Increasing" if slope > 0 else "↓ Decreasing"
        else:
            direction = "→ No significant trend"
        
        trends.append({
            "feature": feature,
            "slope": slope,
            "p_value": p_value,
            "r_squared": r_squared,
            "trend_direction": direction,
        })
    
    trends_df = pd.DataFrame(trends)
    logger.info(f"Analyzed trends for {len(trends_df)} features")
    return trends_df


def plot_trends(period_stats: pd.DataFrame, output_dir: Path) -> None:
    """Create line plot of all audio features over 5-year periods."""
    x = period_stats["bucket_start"].values
    labels = period_stats["bucket_label"].values

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), facecolor="white")
    axes = axes.flatten()

    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]
        col_mean = f"{feature}_mean"
        col_std = f"{feature}_std"

        if col_mean not in period_stats.columns:
            continue

        means = period_stats[col_mean].values
        stds = period_stats[col_std].values if col_std in period_stats.columns else None

        ax.plot(x, means, linewidth=2.5, label=feature, color="steelblue")

        if stds is not None:
            ax.fill_between(x, means - stds, means + stds, alpha=0.3, color="steelblue")

        mask = ~(np.isnan(means))
        if mask.sum() >= 2:
            z = np.polyfit(x[mask], means[mask], 1)
            p = np.poly1d(z)
            ax.plot(x[mask], p(x[mask]), "--", linewidth=2, color="red", alpha=0.7, label="Trend")

        ax.set_xlabel("5-Year Period")
        ax.set_ylabel(feature.capitalize())
        ax.set_title(f"{feature.capitalize()} Trend (1980-2025)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)

    for idx in range(len(AUDIO_FEATURES), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "trend_analysis_all_features.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: trend_analysis_all_features.png")
    plt.close(fig)

def generate_report(
    trends_df: pd.DataFrame,
    period_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path = output_dir / "trend_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPOTIFY AUDIO TRENDS ANALYSIS (1980-2025, 5-YEAR PERIODS)\n")
        f.write("=" * 80 + "\n\n")

        f.write("FEATURE RANGES (FIRST vs LAST 5-YEAR PERIOD):\n")
        first_period = period_stats.iloc[0]
        last_period = period_stats.iloc[-1]

        for feature in AUDIO_FEATURES:
            col_mean = f"{feature}_mean"
            if col_mean in period_stats.columns:
                first_avg = first_period[col_mean]
                last_avg = last_period[col_mean]
                change_pct = ((last_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0

                f.write(f"  • {feature.upper()}\n")
                f.write(f"    {first_period['bucket_label']} average: {first_avg:.4f}\n")
                f.write(f"    {last_period['bucket_label']} average: {last_avg:.4f}\n")
                f.write(f"    Change: {change_pct:+.1f}%\n\n")

def save_statistics(yearly_stats: pd.DataFrame, output_dir: Path) -> None:
    """Save yearly statistics to CSV."""
    output_path = output_dir / "trend_analysis_yearly_stats.csv"
    yearly_stats.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")


def run(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run trend analysis pipeline.
    
    Returns:
        (yearly_stats, trends_df)
    """
    if input_csv is None:
        input_csv = str(PROJECT_ROOT / "data/processed/Spotify_1980_2025_Final.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "reports")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_csv)
    
    period_stats = compute_period_statistics(df)
    trends_df = analyze_trends(period_stats)
    plot_trends(period_stats, output_path)
    save_statistics(period_stats, output_path)
    generate_report(trends_df, period_stats, output_path)
    return period_stats, trends_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze long-term trends in Spotify audio characteristics."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=f"Path to merged Spotify dataset (default: data/processed/Spotify_1980_2025_Final.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for results (default: reports/)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    args = parse_args()
    run(input_csv=args.input_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

