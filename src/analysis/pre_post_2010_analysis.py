"""Analyze impact of streaming era (2010+) on music characteristics."""

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

# Cutoff year
STREAMING_ERA_START = 2010


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


def split_by_streaming_era(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into pre-streaming (1980-2009) and post-streaming (2010+)."""
    pre = df[df["year"] < STREAMING_ERA_START].copy()
    post = df[df["year"] >= STREAMING_ERA_START].copy()
    
    logger.info(f"Pre-streaming (1980-2009): {len(pre)} tracks")
    logger.info(f"Post-streaming (2010+): {len(post)} tracks")
    
    return pre, post


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (group1.mean() - group2.mean()) / pooled_std if pooled_std != 0 else 0
    return d


def perform_statistical_tests(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """
    Perform t-tests, Mann-Whitney U tests, and calculate effect sizes.
    
    Returns:
        DataFrame with test results
    """
    results = []
    
    for feature in AUDIO_FEATURES:
        pre_values = pre[feature].dropna().values
        post_values = post[feature].dropna().values
        
        # Check normality (Shapiro-Wilk for subsample)
        sample_size = min(5000, len(pre_values), len(post_values))
        pre_sample = np.random.choice(pre_values, sample_size, replace=False)
        post_sample = np.random.choice(post_values, sample_size, replace=False)
        
        _, p_shapiro_pre = stats.shapiro(pre_sample)
        _, p_shapiro_post = stats.shapiro(post_sample)
        normal = (p_shapiro_pre > 0.05) and (p_shapiro_post > 0.05)
        
        # t-test
        t_stat, t_pval = stats.ttest_ind(pre_values, post_values)
        
        # Mann-Whitney U (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(pre_values, post_values, alternative="two-sided")
        
        # Levene's test for variance
        lev_stat, lev_pval = stats.levene(pre_values, post_values)
        
        # Cohen's d
        d = cohens_d(pre_values, post_values)
        
        # Effect size interpretation
        if abs(d) < 0.2:
            effect = "Negligible"
        elif abs(d) < 0.5:
            effect = "Small"
        elif abs(d) < 0.8:
            effect = "Medium"
        else:
            effect = "Large"
        
        results.append({
            "feature": feature,
            "pre_mean": pre_values.mean(),
            "post_mean": post_values.mean(),
            "mean_change_pct": ((post_values.mean() - pre_values.mean()) / pre_values.mean() * 100),
            "pre_std": pre_values.std(),
            "post_std": post_values.std(),
            "pre_median": np.median(pre_values),
            "post_median": np.median(post_values),
            "t_statistic": t_stat,
            "t_pvalue": t_pval,
            "u_statistic": u_stat,
            "u_pvalue": u_pval,
            "cohens_d": d,
            "effect_size": effect,
            "lev_pvalue": lev_pval,
            "variance_changed": "Yes" if lev_pval < 0.05 else "No",
            "is_normal": normal,
        })
    
    results_df = pd.DataFrame(results)
    logger.info(f"Statistical tests completed for {len(results_df)} features")
    return results_df


def plot_distributions(pre: pd.DataFrame, post: pd.DataFrame, output_dir: Path) -> None:
    """Plot box plots comparing pre and post streaming era."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), facecolor="white")
    axes = axes.flatten()
    
    for idx, feature in enumerate(AUDIO_FEATURES):
        ax = axes[idx]
        
        # Prepare data for box plot
        pre_vals = pre[feature].dropna()
        post_vals = post[feature].dropna()
        
        data_to_plot = [pre_vals, post_vals]
        
        bp = ax.boxplot(
            data_to_plot,
            labels=["Pre-Streaming\n(1980-2009)", "Post-Streaming\n(2010-2025)"],
            patch_artist=True,
            showmeans=True,
        )
        
        # Color boxes
        colors = ["lightblue", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(feature.capitalize(), fontweight="bold")
        ax.set_title(f"{feature.capitalize()}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    
    axes[-1].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / "pre_post_2010_boxplots.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: pre_post_2010_boxplots.png")
    plt.close(fig)


def plot_effect_sizes(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Cohen's d effect sizes."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    
    features = results_df["feature"].values
    cohens_d_vals = results_df["cohens_d"].values
    
    colors = ["red" if x < 0 else "green" for x in cohens_d_vals]
    ax.barh(range(len(features)), cohens_d_vals, color=colors, alpha=0.7, edgecolor="black")
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.capitalize() for f in features])
    ax.set_xlabel("Cohen's d (Effect Size)", fontweight="bold")
    ax.set_title("Streaming Era Impact: Effect Sizes (Pre vs Post 2010)", fontweight="bold", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=-0.2, color="gray", linestyle="--", alpha=0.5, label="Small effect")
    ax.axvline(x=0.2, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / "pre_post_2010_effect_sizes.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: pre_post_2010_effect_sizes.png")
    plt.close(fig)


def plot_mean_changes(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot percentage changes in means."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    
    features = results_df["feature"].values
    changes = results_df["mean_change_pct"].values
    
    colors = ["red" if x < 0 else "green" for x in changes]
    ax.barh(range(len(features)), changes, color=colors, alpha=0.7, edgecolor="black")
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.capitalize() for f in features])
    ax.set_xlabel("Change in Mean (%)", fontweight="bold")
    ax.set_title("Streaming Era Impact: Percentage Change in Feature Means", fontweight="bold", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(changes):
        ax.text(v + (1 if v > 0 else -1), i, f"{v:+.1f}%", va="center", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / "pre_post_2010_mean_changes.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: pre_post_2010_mean_changes.png")
    plt.close(fig)


def plot_variance_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot variance comparison (standard deviation)."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    
    features = results_df["feature"].values
    x = np.arange(len(features))
    width = 0.35
    
    pre_stds = results_df["pre_std"].values
    post_stds = results_df["post_std"].values
    
    ax.bar(x - width/2, pre_stds, width, label="Pre-Streaming", alpha=0.7, color="steelblue", edgecolor="black")
    ax.bar(x + width/2, post_stds, width, label="Post-Streaming", alpha=0.7, color="coral", edgecolor="black")
    
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Standard Deviation", fontweight="bold")
    ax.set_title("Variance Comparison: Pre vs Post Streaming Era", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in features], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "pre_post_2010_variance_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: pre_post_2010_variance_comparison.png")
    plt.close(fig)


def generate_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate detailed streaming era impact report."""
    report_path = output_dir / "pre_post_2010_analysis_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPOTIFY MUSIC: STREAMING ERA IMPACT ANALYSIS (Pre-2010 vs Post-2010)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("Did the rise of streaming platforms (2010+) fundamentally change music characteristics?\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("Pre-Streaming Era: 1980-2009 (traditional radio/physical media era)\n")
        f.write("Post-Streaming Era: 2010-2025 (Spotify/Apple Music/YouTube Music)\n\n")
        f.write("Statistical Tests:\n")
        f.write("  • t-test (assumes normality)\n")
        f.write("  • Mann-Whitney U test (non-parametric, rank-based)\n")
        f.write("  • Levene's test for variance equality\n")
        f.write("  • Cohen's d for effect size\n\n")
        
        f.write("RESULTS:\n")
        f.write("=" * 80 + "\n\n")
        
        # Significant changes
        sig_results = results_df[results_df["u_pvalue"] < 0.05].sort_values("cohens_d", ascending=False)
        
        f.write(f"STATISTICALLY SIGNIFICANT CHANGES (Mann-Whitney U, α=0.05):\n")
        f.write(f"Found {len(sig_results)} out of {len(results_df)} features with significant changes\n\n")
        
        for _, row in sig_results.iterrows():
            f.write(f"{row['feature'].upper()}:\n")
            f.write(f"  Pre-Streaming Mean:  {row['pre_mean']:.4f} (±{row['pre_std']:.4f})\n")
            f.write(f"  Post-Streaming Mean: {row['post_mean']:.4f} (±{row['post_std']:.4f})\n")
            f.write(f"  Change: {row['mean_change_pct']:+.1f}%\n")
            f.write(f"  Cohen's d: {row['cohens_d']:.4f} ({row['effect_size']} effect)\n")
            f.write(f"  p-value (Mann-Whitney): {row['u_pvalue']:.6f}\n")
            f.write(f"  Variance Changed: {row['variance_changed']} (p={row['lev_pvalue']:.4f})\n\n")
        
        # No significant changes
        no_sig = results_df[results_df["u_pvalue"] >= 0.05]
        if len(no_sig) > 0:
            f.write(f"\nNO SIGNIFICANT CHANGES ({len(no_sig)} features):\n")
            for _, row in no_sig.iterrows():
                f.write(f"  • {row['feature'].upper()} (p={row['u_pvalue']:.4f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        f.write("Large Cohen's d (|d| > 0.8): Streaming fundamentally changed this feature\n")
        f.write("Medium Cohen's d (0.5-0.8): Moderate streaming era effect\n")
        f.write("Small Cohen's d (0.2-0.5): Minor streaming era effect\n")
        f.write("Negligible Cohen's d (< 0.2): Minimal or no streaming era effect\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generated by pre_post_2010_analysis.py\n")
    
    logger.info(f"Saved: {report_path}")


def save_statistics(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Save test results to CSV."""
    results_df.to_csv(output_dir / "pre_post_2010_statistical_tests.csv", index=False)
    logger.info("Saved: pre_post_2010_statistical_tests.csv")


def run(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run pre/post-2010 analysis pipeline.
    
    Returns:
        results_df
    """
    if input_csv is None:
        input_csv = str(PROJECT_ROOT / "data/processed/Spotify_1980_2025_Final.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "reports")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    df = load_data(input_csv)
    pre, post = split_by_streaming_era(df)
    
    # Statistical tests
    results_df = perform_statistical_tests(pre, post)
    
    # Generate outputs
    plot_distributions(pre, post, output_path)
    plot_effect_sizes(results_df, output_path)
    plot_mean_changes(results_df, output_path)
    plot_variance_comparison(results_df, output_path)
    save_statistics(results_df, output_path)
    generate_report(results_df, output_path)
    
    logger.info(f"Pre/post-2010 analysis complete. Results saved to: {output_path}")
    
    return results_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze streaming era (2010+) impact on music characteristics."
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
