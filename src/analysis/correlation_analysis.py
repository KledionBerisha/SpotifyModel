"""Analyze correlations between audio features across time."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Project root
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

    for col in AUDIO_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["year"].between(1980, ANALYSIS_END_YEAR)].copy()
    logger.info(f"Loaded {len(df)} tracks from {csv_path}")
    return df


def add_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5-year period columns."""
    df = df.copy()
    bucket_start = (df["year"] // BUCKET_SIZE_YEARS) * BUCKET_SIZE_YEARS
    bucket_end = np.minimum(bucket_start + (BUCKET_SIZE_YEARS - 1), ANALYSIS_END_YEAR)
    df["period_start"] = bucket_start.astype(int)
    df["period_label"] = bucket_start.astype(int).astype(str) + "-" + bucket_end.astype(int).astype(str)
    return df


def compute_global_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson and Spearman correlations for entire dataset.

    Returns:
        (pearson_corr, spearman_corr)
    """
    features_df = df[AUDIO_FEATURES].dropna()
    pearson_corr = features_df.corr(method="pearson")
    spearman_corr = features_df.corr(method="spearman")

    logger.info("Computed global Pearson and Spearman correlations")
    return pearson_corr, spearman_corr


def compute_period_correlations(df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Compute Pearson and Spearman correlations for each complete 5-year period."""
    period_corrs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for period_start in sorted(df["period_start"].unique()):
        period_df = df[df["period_start"] == period_start].copy()
        period_label = f"{int(period_start)}-{min(int(period_start) + 4, ANALYSIS_END_YEAR)}"

        # Skip incomplete tail periods (for example, 2025-2025).
        if period_df["year"].nunique() < BUCKET_SIZE_YEARS:
            continue

        period_data = period_df[AUDIO_FEATURES].dropna()
        if len(period_data) < 10:
            continue

        pearson = period_data.corr(method="pearson")
        spearman = period_data.corr(method="spearman")
        period_corrs[period_label] = (pearson, spearman)

    logger.info(f"Computed correlations for {len(period_corrs)} complete 5-year periods")
    return period_corrs


def plot_global_heatmap(pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame, output_dir: Path) -> None:
    """Plot global correlation heatmaps (Pearson and Spearman)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=axes[0],
    )
    axes[0].set_title("Pearson Correlation (All Data 1980-2025)", fontweight="bold", fontsize=12)

    sns.heatmap(
        spearman_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=axes[1],
    )
    axes[1].set_title("Spearman Correlation (All Data 1980-2025)", fontweight="bold", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "correlation_global_heatmap.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: correlation_global_heatmap.png")
    plt.close(fig)


def plot_period_heatmaps(period_corrs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], output_dir: Path) -> None:
    """Plot 5-year period correlation heatmaps (Pearson only for clarity)."""
    periods = sorted(period_corrs.keys(), key=lambda x: int(x.split("-")[0]))
    n_periods = len(periods)

    n_cols = 4
    n_rows = (n_periods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), facecolor="white")
    axes = axes.flatten()

    for idx, period in enumerate(periods):
        ax = axes[idx]
        pearson_corr, _ = period_corrs[period]

        sns.heatmap(
            pearson_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.3,
            cbar=False,
            ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title(period, fontweight="bold", fontsize=10)

    for idx in range(n_periods, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(output_dir / "correlation_period_heatmaps.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: correlation_period_heatmaps.png")
    plt.close(fig)


def extract_top_correlations(pearson_corr: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extract top positive and negative correlations (excluding diagonal)."""
    correlations = []

    for i in range(len(pearson_corr.columns)):
        for j in range(i + 1, len(pearson_corr.columns)):
            feat1 = pearson_corr.columns[i]
            feat2 = pearson_corr.columns[j]
            corr_value = pearson_corr.iloc[i, j]
            correlations.append(
                {
                    "feature_1": feat1,
                    "feature_2": feat2,
                    "correlation": corr_value,
                }
            )

    corr_df = pd.DataFrame(correlations)
    top_positive = corr_df.nlargest(top_n, "correlation")
    top_negative = corr_df.nsmallest(top_n, "correlation")
    return pd.concat([top_positive, top_negative], ignore_index=True)


def plot_strongest_correlations(pearson_corr: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Plot bar chart of strongest positive and negative correlations."""
    top_corrs = extract_top_correlations(pearson_corr, top_n=top_n)
    top_corrs["pair"] = top_corrs["feature_1"] + " ↔ " + top_corrs["feature_2"]
    top_corrs = top_corrs.sort_values("correlation")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    colors = ["red" if x < 0 else "green" for x in top_corrs["correlation"]]
    ax.barh(range(len(top_corrs)), top_corrs["correlation"], color=colors, alpha=0.7, edgecolor="black")
    ax.set_yticks(range(len(top_corrs)))
    ax.set_yticklabels(top_corrs["pair"], fontsize=10)
    ax.set_xlabel("Correlation Coefficient", fontweight="bold")
    ax.set_title(f"Top {top_n} Strongest Correlations (Positive & Negative)", fontweight="bold", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "correlation_strongest_pairs.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: correlation_strongest_pairs.png")
    plt.close(fig)


def analyze_correlation_changes(
    global_pearson: pd.DataFrame,
    period_corrs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Analyze how specific correlations change across 5-year periods."""
    key_pairs = [
        ("energy", "loudness"),
        ("energy", "acousticness"),
        ("danceability", "tempo"),
        ("valence", "energy"),
        ("acousticness", "liveness"),
    ]

    results = []
    periods = sorted(period_corrs.keys(), key=lambda x: int(x.split("-")[0]))

    for feat1, feat2 in key_pairs:
        row = {"feature_pair": f"{feat1} ↔ {feat2}"}
        for period in periods:
            pearson, _ = period_corrs[period]
            if feat1 in pearson.columns and feat2 in pearson.columns:
                row[period] = pearson.loc[feat1, feat2]
        results.append(row)

    return pd.DataFrame(results)


def plot_correlation_changes(changes_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot how key correlations change over time."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    periods = [col for col in changes_df.columns if col != "feature_pair"]

    for _, row in changes_df.iterrows():
        pair = row["feature_pair"]
        values = [row[p] for p in periods]
        ax.plot(range(len(periods)), values, marker="o", linewidth=2.5, label=pair, markersize=8)

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45)
    ax.set_ylabel("Correlation Coefficient", fontweight="bold")
    ax.set_xlabel("5-Year Period", fontweight="bold")
    ax.set_title("How Key Feature Correlations Change Over Time", fontweight="bold", fontsize=12)
    ax.set_xlim(-0.5, len(periods) - 0.5)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "correlation_changes_over_time.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: correlation_changes_over_time.png")
    plt.close(fig)


def generate_report(
    pearson_corr: pd.DataFrame,
    period_corrs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    changes_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate correlation analysis report."""
    report_path = output_dir / "correlation_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPOTIFY AUDIO FEATURE CORRELATIONS ANALYSIS (1980-2025, 5-YEAR PERIODS)\n")
        f.write("=" * 80 + "\n\n")

        f.write("RESEARCH QUESTION:\n")
        f.write("Which audio features correlate with each other, and how do these relationships change over time?\n\n")

        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Compute Pearson (linear) and Spearman (rank) correlations\n")
        f.write("2. Global correlation matrix for entire dataset\n")
        f.write("3. 5-year-period-specific correlations to track changes\n")
        f.write("4. Identify strongest positive/negative relationships\n\n")

        f.write("STRONGEST CORRELATIONS (ALL DATA):\n")
        f.write("-" * 80 + "\n\n")

        top_corrs = extract_top_correlations(pearson_corr, top_n=10)

        positive = top_corrs[top_corrs["correlation"] > 0].nlargest(5, "correlation")
        f.write("STRONGEST POSITIVE CORRELATIONS:\n")
        for _, row in positive.iterrows():
            f.write(f"  • {row['feature_1'].upper()} ↔ {row['feature_2'].upper()}\n")
            f.write(f"    Correlation: {row['correlation']:.4f}\n")
            f.write("    Interpretation: These features tend to increase/decrease together\n\n")

        negative = top_corrs[top_corrs["correlation"] < 0].nsmallest(5, "correlation")
        f.write("STRONGEST NEGATIVE CORRELATIONS:\n")
        for _, row in negative.iterrows():
            f.write(f"  • {row['feature_1'].upper()} ↔ {row['feature_2'].upper()}\n")
            f.write(f"    Correlation: {row['correlation']:.4f}\n")
            f.write("    Interpretation: These features move in opposite directions\n\n")

        f.write("CORRELATION CHANGES ACROSS 5-YEAR PERIODS:\n")
        f.write("-" * 80 + "\n\n")

        for _, row in changes_df.iterrows():
            pair = row["feature_pair"]
            periods = [col for col in changes_df.columns if col != "feature_pair"]
            first_val = row[periods[0]]
            last_val = row[periods[-1]]
            change = last_val - first_val

            f.write(f"{pair.upper()}:\n")
            f.write(f"  {periods[0]}: {first_val:.4f}\n")
            f.write(f"  {periods[-1]}: {last_val:.4f}\n")
            f.write(f"  Change: {change:+.4f}\n")
            if abs(change) > 0.2:
                f.write("  ** SIGNIFICANT CHANGE **\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Report generated by correlation_analysis.py\n")

    logger.info(f"Saved: {report_path}")


def save_statistics(
    pearson_corr: pd.DataFrame,
    spearman_corr: pd.DataFrame,
    changes_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save correlation matrices and changes to CSV."""
    pearson_corr.to_csv(output_dir / "correlation_pearson_global.csv")
    spearman_corr.to_csv(output_dir / "correlation_spearman_global.csv")
    changes_df.to_csv(output_dir / "correlation_changes_by_5year_period.csv", index=False)
    logger.info("Saved: correlation CSV files")


def run(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run correlation analysis pipeline.

    Returns:
        (global_pearson, period_corrs_dict)
    """
    if input_csv is None:
        input_csv = str(PROJECT_ROOT / "data/processed/Spotify_1960_2026_Final.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "reports")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_data(input_csv)
    df = add_period(df)

    pearson_corr, spearman_corr = compute_global_correlations(df)
    period_corrs = compute_period_correlations(df)
    changes_df = analyze_correlation_changes(pearson_corr, period_corrs)

    plot_global_heatmap(pearson_corr, spearman_corr, output_path)
    plot_period_heatmaps(period_corrs, output_path)
    plot_strongest_correlations(pearson_corr, output_path)
    plot_correlation_changes(changes_df, output_path)
    save_statistics(pearson_corr, spearman_corr, changes_df, output_path)
    generate_report(pearson_corr, period_corrs, changes_df, output_path)

    logger.info(f"Correlation analysis complete. Results saved to: {output_path}")
    return pearson_corr, period_corrs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Analyze correlations between audio features.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to merged Spotify dataset (default: data/processed/Spotify_1960_2026_Final.csv)",
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