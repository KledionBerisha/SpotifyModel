"""Analyze long-term trends in Spotify audio characteristics (1960-2026)."""

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
    
    logger.info(f"Loaded {len(df)} tracks from {csv_path}")
    return df


def compute_yearly_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly statistics for all audio features.
    
    Returns:
        DataFrame with columns: year, feature_mean, feature_median, feature_std, feature_count
    """
    yearly_stats = df.groupby("year", as_index=False)[AUDIO_FEATURES].agg([
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
    ])
    
    # Flatten multi-level columns
    yearly_stats.columns = ["_".join(col).strip() if col[1] else col[0] 
                             for col in yearly_stats.columns]
    yearly_stats.rename(columns={"year_": "year"}, inplace=True)
    
    # Add count of tracks per year
    yearly_counts = df.groupby("year").size().reset_index(name="track_count")
    yearly_stats = yearly_stats.merge(yearly_counts, on="year")
    
    logger.info(f"Computed yearly statistics for {len(yearly_stats)} years")
    return yearly_stats


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


def plot_trends(yearly_stats: pd.DataFrame, output_dir: Path) -> None:
    """Create line plot of all audio features over time."""
    years = yearly_stats["year"].values
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), facecolor="white")
    axes = axes.flatten()
    
    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]
        col_mean = f"{feature}_mean"
        col_std = f"{feature}_std"
        
        if col_mean not in yearly_stats.columns:
            continue
        
        means = yearly_stats[col_mean].values
        stds = yearly_stats[col_std].values if col_std in yearly_stats.columns else None
        
        ax.plot(years, means, linewidth=2.5, label=feature, color="steelblue")
        
        if stds is not None:
            ax.fill_between(years, means - stds, means + stds, alpha=0.3, color="steelblue")
        
        # Add linear trend line
        mask = ~(np.isnan(means))
        if mask.sum() >= 2:
            z = np.polyfit(years[mask], means[mask], 1)
            p = np.poly1d(z)
            ax.plot(years[mask], p(years[mask]), "--", linewidth=2, color="red", alpha=0.7, label="Trend")
        
        ax.set_xlabel("Year")
        ax.set_ylabel(feature.capitalize())
        ax.set_title(f"{feature.capitalize()} Trend (1960-2026)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(AUDIO_FEATURES), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / "trend_analysis_all_features.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: trend_analysis_all_features.png")
    plt.close(fig)


def generate_report(
    trends_df: pd.DataFrame,
    yearly_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate a text report summarizing key findings."""
    report_path = output_dir / "trend_analysis_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPOTIFY AUDIO TRENDS ANALYSIS (1960-2026)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("How have the audio characteristics of popular songs evolved over time?\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n\n")
        
        # Significant trends
        sig_trends = trends_df[trends_df["p_value"] < 0.05].sort_values("p_value")
        if len(sig_trends) > 0:
            f.write(f"STATISTICALLY SIGNIFICANT TRENDS (p < 0.05):\n")
            for _, row in sig_trends.iterrows():
                f.write(f"  • {row['feature'].upper()}\n")
                f.write(f"    Trend: {row['trend_direction']}\n")
                f.write(f"    Slope: {row['slope']:.6f}\n")
                f.write(f"    P-value: {row['p_value']:.4f}\n")
                f.write(f"    R²: {row['r_squared']:.4f}\n\n")
        else:
            f.write("No statistically significant trends found (α = 0.05).\n\n")
        
        # Feature ranges over time
        f.write("FEATURE RANGES (1960 vs 2026):\n")
        early_years = yearly_stats[yearly_stats["year"] <= 1970]
        recent_years = yearly_stats[yearly_stats["year"] >= 2020]
        
        if len(early_years) > 0 and len(recent_years) > 0:
            for feature in AUDIO_FEATURES:
                col_mean = f"{feature}_mean"
                if col_mean in yearly_stats.columns:
                    early_avg = early_years[col_mean].mean()
                    recent_avg = recent_years[col_mean].mean()
                    change_pct = ((recent_avg - early_avg) / early_avg * 100) if early_avg != 0 else 0
                    
                    f.write(f"  • {feature.upper()}\n")
                    f.write(f"    1960s average: {early_avg:.4f}\n")
                    f.write(f"    2020s average: {recent_avg:.4f}\n")
                    f.write(f"    Change: {change_pct:+.1f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generated by trend_analysis.py\n")
    
    logger.info(f"Saved: {report_path}")


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
        input_csv = str(PROJECT_ROOT / "data/processed/Spotify_1960_2026_Final.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "reports")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_csv)
    
    # Compute statistics
    yearly_stats = compute_yearly_statistics(df)
    
    # Analyze trends
    trends_df = analyze_trends(yearly_stats)
    
    # Generate outputs
    plot_trends(yearly_stats, output_path)
    save_statistics(yearly_stats, output_path)
    generate_report(trends_df, yearly_stats, output_path)
    
    logger.info(f"Trend analysis complete. Results saved to: {output_path}")
    
    return yearly_stats, trends_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze long-term trends in Spotify audio characteristics."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=f"Path to merged Spotify dataset (default: data/processed/Spotify_1960_2026_Final.csv)",
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