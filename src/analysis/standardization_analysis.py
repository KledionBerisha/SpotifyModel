"""Analyze music standardization: Are modern songs becoming more uniform?"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_END_YEAR = 2025

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

    for col in AUDIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["year"].between(1980, ANALYSIS_END_YEAR)].copy()

    logger.info(f"Loaded {len(df)} tracks from {csv_path}")
    return df


def add_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Add decade column."""
    df = df.copy()
    bucket_start = (df["year"] // 5) * 5
    bucket_end = np.minimum(bucket_start + 4, ANALYSIS_END_YEAR)
    df["decade"] = bucket_start.astype(int)
    df["decade_label"] = bucket_start.astype(int).astype(str) + "-" + bucket_end.astype(int).astype(str)
    return df


def compute_variance_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance, std, IQR, and coefficient of variation by decade.
    
    Returns:
        DataFrame with variance statistics per feature per decade
    """
    results = []
    
    for decade in sorted(df["decade"].unique()):
        decade_data = df[df["decade"] == decade]
        decade_label = f"{int(decade)}s"
        track_count = len(decade_data)
        
        for feature in AUDIO_FEATURES:
            values = decade_data[feature].dropna()
            
            if len(values) < 2:
                continue
            
            var = values.var()
            std = values.std()
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            mean = values.mean()
            cv = (std / mean) if mean != 0 else 0  # Coefficient of variation
            
            results.append({
                "decade": decade,
                "decade_label": decade_label,
                "feature": feature,
                "variance": var,
                "std": std,
                "iqr": iqr,
                "cv": cv,
                "mean": mean,
                "track_count": track_count,
            })
    
    results_df = pd.DataFrame(results)
    logger.info(f"Computed variance statistics for {len(results_df)} feature-decade pairs")
    return results_df


def levene_test_by_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Levene's test for variance equality across decades for each feature.
    
    Returns:
        DataFrame with test statistics and p-values
    """
    results = []
    
    for feature in AUDIO_FEATURES:
        # Group by decade
        decade_groups = []
        for decade in sorted(df["decade"].unique()):
            decade_data = df[df["decade"] == decade][feature].dropna()
            if len(decade_data) > 1:
                decade_groups.append(decade_data.values)
        
        if len(decade_groups) < 2:
            continue
        
        # Levene's test
        stat, p_value = stats.levene(*decade_groups)
        
        # Determine if variance is changing
        variance_changing = "Yes (p < 0.05)" if p_value < 0.05 else "No (p ≥ 0.05)"
        
        results.append({
            "feature": feature,
            "levene_statistic": stat,
            "p_value": p_value,
            "variance_changing": variance_changing,
        })
    
    results_df = pd.DataFrame(results)
    logger.info(f"Levene's test completed for {len(results_df)} features")
    return results_df


def plot_variance_trends(variance_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot variance/IQR trends across decades."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 10), facecolor="white")
    axes = axes.flatten()
    
    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]
        feature_data = variance_df[variance_df["feature"] == feature].sort_values("decade")
        
        decades = feature_data["decade_label"].values
        variances = feature_data["variance"].values
        iqrs = feature_data["iqr"].values
        
        ax.plot(decades, variances, marker="o", linewidth=2.5, markersize=8, 
                label="Variance", color="steelblue")
        ax.set_title(f"{feature.capitalize()} Variance Trend", fontweight="bold")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Variance")
        ax.set_xlim(-0.5, len(decades) - 0.5)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    
    # Remove extra subplot
    axes[-1].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / "standardization_variance_trends.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: standardization_variance_trends.png")
    plt.close(fig)


def plot_violin_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Create violin plots comparing feature distributions by decade."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), facecolor="white")
    axes = axes.flatten()
    
    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]
        
        # Prepare data for violin plot
        plot_data = df[[feature, "decade_label"]].dropna()
        
        decade_order = sorted(
            plot_data["decade_label"].unique(),
            key=lambda x: int(str(x).split("-")[0]),
        )
        
        sns.violinplot(data=plot_data, x="decade_label", y=feature, 
                       order=decade_order, ax=ax, palette="Set2")
        
        ax.set_title(f"{feature.capitalize()}", fontweight="bold")
        ax.set_xlabel("Decade")
        ax.set_ylabel(feature.capitalize())
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    
    axes[-1].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / "standardization_violin_distributions.png", 
                dpi=300, bbox_inches="tight")
    logger.info("Saved: standardization_violin_distributions.png")
    plt.close(fig)


def plot_iqr_trends(variance_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot IQR (interquartile range) trends to show distribution narrowing."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), facecolor="white")
    axes = axes.flatten()
    
    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]
        feature_data = variance_df[variance_df["feature"] == feature].sort_values("decade")
        
        decades = feature_data["decade_label"].values
        iqrs = feature_data["iqr"].values
        
        ax.bar(range(len(decades)), iqrs, color="coral", alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(decades)))
        ax.set_xticklabels(decades, rotation=45)
        ax.set_title(f"{feature.capitalize()} IQR by Decade", fontweight="bold")
        ax.set_ylabel("IQR (Interquartile Range)")
        ax.set_xlabel("Decade")
        ax.set_xlim(-0.5, len(decades) - 0.5)
        ax.grid(axis="y", alpha=0.3)
    
    axes[-1].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / "standardization_iqr_trends.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: standardization_iqr_trends.png")
    plt.close(fig)


def generate_report(
    variance_df: pd.DataFrame,
    levene_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate standardization analysis report."""
    report_path = output_dir / "standardization_analysis_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPOTIFY MUSIC STANDARDIZATION ANALYSIS (1980-2025)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("Is modern music becoming more homogenized/standardized?\n")
        f.write("(Do songs have less variance in audio characteristics over time?)\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Compute variance and IQR by decade for each feature\n")
        f.write("2. Test for variance equality using Levene's test\n")
        f.write("3. Visualize distributions with violin plots\n")
        f.write("4. Analyze trends: Is variance decreasing?\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n\n")
        
        # Levene's test results
        f.write("LEVENE'S TEST FOR VARIANCE EQUALITY (across decades):\n")
        f.write("H0: Variance is equal across all decades\n")
        f.write("Result (α = 0.05): Reject H0 if p < 0.05\n\n")
        
        sig_features = levene_df[levene_df["p_value"] < 0.05].sort_values("p_value")
        if len(sig_features) > 0:
            f.write(f"SIGNIFICANT VARIANCE CHANGES ({len(sig_features)} features):\n")
            for _, row in sig_features.iterrows():
                f.write(f"  • {row['feature'].upper()}\n")
                f.write(f"    P-value: {row['p_value']:.6f}\n")
                f.write(f"    Interpretation: Variance changes significantly across decades\n\n")
        
        no_sig = levene_df[levene_df["p_value"] >= 0.05]
        if len(no_sig) > 0:
            f.write(f"NO SIGNIFICANT VARIANCE CHANGES ({len(no_sig)} features):\n")
            for _, row in no_sig.iterrows():
                f.write(f"  • {row['feature'].upper()} (p = {row['p_value']:.4f})\n")
            f.write("\n")
        
        # Variance trend analysis
        f.write("VARIANCE TRENDS (1980 → 2025) (5-Year Periods):\n")
        f.write("-" * 80 + "\n\n")
        
        for feature in AUDIO_FEATURES:
            feature_var = variance_df[variance_df["feature"] == feature].sort_values("decade")
            
            if len(feature_var) < 2:
                continue
            
            var_1980s = feature_var.iloc[0]["variance"]
            var_2025 = feature_var.iloc[-1]["variance"]
            change_pct = ((var_2025 - var_1980s) / var_1980s * 100) if var_1980s != 0 else 0
            
            trend = "↓ DECREASING (more standardized)" if var_2025 < var_1980s else "↑ INCREASING (more diverse)"
            
            f.write(f"{feature.upper()}:\n")
            f.write(f"  1980s Variance: {var_1980s:.6f}\n")
            f.write(f"  2020s Variance: {var_2025:.6f}\n")
            f.write(f"  Change: {change_pct:+.1f}%\n")
            f.write(f"  Trend: {trend}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CONCLUSION:\n")
        f.write("-" * 80 + "\n")
        f.write("If variance is DECREASING over time → Music is becoming MORE STANDARDIZED\n")
        f.write("If variance is INCREASING over time → Music is becoming MORE DIVERSE\n\n")
        f.write("Report generated by standardization_analysis.py\n")
    
    logger.info(f"Saved: {report_path}")


def save_statistics(variance_df: pd.DataFrame, levene_df: pd.DataFrame, output_dir: Path) -> None:
    """Save variance statistics and test results to CSV."""
    variance_df.to_csv(output_dir / "standardization_variance_stats.csv", index=False)
    levene_df.to_csv(output_dir / "standardization_levene_test.csv", index=False)
    logger.info("Saved: standardization_variance_stats.csv, standardization_levene_test.csv")


def run(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run standardization analysis pipeline.
    
    Returns:
        (variance_df, levene_df)
    """
    if input_csv is None:
        input_csv = str(PROJECT_ROOT / "data/processed/Spotify_1980_2025_Final.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "reports")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_csv)
    df = add_decade(df)
    
    # Compute variance by decade
    variance_df = compute_variance_by_decade(df)
    
    # Levene's test
    levene_df = levene_test_by_feature(df)
    
    # Generate outputs
    plot_variance_trends(variance_df, output_path)
    plot_violin_distributions(df, output_path)
    plot_iqr_trends(variance_df, output_path)
    save_statistics(variance_df, levene_df, output_path)
    generate_report(variance_df, levene_df, output_path)
    
    logger.info(f"Standardization analysis complete. Results saved to: {output_path}")
    
    return variance_df, levene_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze whether modern music is becoming more standardized."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to merged Spotify dataset (default: data/processed/Spotify_1980_2025_Final.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: reports/)",
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
