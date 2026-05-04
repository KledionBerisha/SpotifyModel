"""Analyze correlations between audio features across time."""

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
    
    logger.info(f"Loaded {len(df)} tracks from {csv_path}")
    return df


def add_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Add decade column."""
    df = df.copy()
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["decade_label"] = df["decade"].astype(str) + "s"
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


def compute_decade_correlations(df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Compute Pearson and Spearman correlations for each decade.
    
    Returns:
        Dict mapping decade_label -> (pearson_corr, spearman_corr)
    """
    decade_corrs = {}
    
    for decade in sorted(df["decade"].unique()):
        decade_data = df[df["decade"] == decade][AUDIO_FEATURES].dropna()
        decade_label = f"{int(decade)}s"
        
        if len(decade_data) < 10:
            continue
        
        pearson = decade_data.corr(method="pearson")
        spearman = decade_data.corr(method="spearman")
        
        decade_corrs[decade_label] = (pearson, spearman)
    
    logger.info(f"Computed correlations for {len(decade_corrs)} decades")
    return decade_corrs


def plot_global_heatmap(pearson_corr: pd.DataFrame, spearman_corr: pd.DataFrame, output_dir: Path) -> None:
    """Plot global correlation heatmaps (Pearson and Spearman)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    
    # Pearson
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
    axes[0].set_title("Pearson Correlation (All Data 1960-2026)", fontweight="bold", fontsize=12)
    
    # Spearman
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
    axes[1].set_title("Spearman Correlation (All Data 1960-2026)", fontweight="bold", fontsize=12)
    
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_global_heatmap.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: correlation_global_heatmap.png")
    plt.close(fig)


def plot_decade_heatmaps(decade_corrs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], output_dir: Path) -> None:
    """Plot decade-specific correlation heatmaps (Pearson only for clarity)."""
    decades = sorted(decade_corrs.keys(), key=lambda x: int(x[:-1]))
    n_decades = len(decades)
    
    # Arrange in 2 rows
    n_cols = 4
    n_rows = (n_decades + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), facecolor="white")
    axes = axes.flatten()
    
    for idx, decade in enumerate(decades):
        ax = axes[idx]
        pearson_corr, _ = decade_corrs[decade]
        
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
        ax.set_title(f"{decade}", fontweight="bold", fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_decades, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_decade_heatmaps.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: correlation_decade_heatmaps.png")
    plt.close(fig)


def extract_top_correlations(pearson_corr: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Extract top positive and negative correlations (excluding diagonal).
    
    Returns:
        DataFrame with feature pairs and correlation values
    """
    correlations = []
    
    for i in range(len(pearson_corr.columns)):
        for j in range(i + 1, len(pearson_corr.columns)):
            feat1 = pearson_corr.columns[i]
            feat2 = pearson_corr.columns[j]
            corr_value = pearson_corr.iloc[i, j]
            
            correlations.append({
                "feature_1": feat1,
                "feature_2": feat2,
                "correlation": corr_value,
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # Top positive
    top_positive = corr_df.nlargest(top_n, "correlation")
    # Top negative
    top_negative = corr_df.nsmallest(top_n, "correlation")
    
    return pd.concat([top_positive, top_negative], ignore_index=True)


def plot_strongest_correlations(pearson_corr: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    """Plot bar chart of strongest positive and negative correlations."""
    top_corrs = extract_top_correlations(pearson_corr, top_n=top_n)
    
    # Create labels
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
    decade_corrs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Analyze how specific correlations change across decades.
    
    Returns:
        DataFrame with correlation changes over time
    """
    # Key pairs to track
    key_pairs = [
        ("energy", "loudness"),
        ("energy", "acousticness"),
        ("danceability", "tempo"),
        ("valence", "energy"),
        ("acousticness", "liveness"),
    ]
    
    results = []
    decades = sorted(decade_corrs.keys(), key=lambda x: int(x[:-1]))
    
    for feat1, feat2 in key_pairs:
        row = {"feature_pair": f"{feat1} ↔ {feat2}"}
        
        for decade in decades:
            pearson, _ = decade_corrs[decade]
            if feat1 in pearson.columns and feat2 in pearson.columns:
                corr = pearson.loc[feat1, feat2]
                row[decade] = corr
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_correlation_changes(changes_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot how key correlations change over time."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    
    # Extract decades for x-axis
    decades = [col for col in changes_df.columns if col != "feature_pair"]
    
    for _, row in changes_df.iterrows():
        pair = row["feature_pair"]
        values = [row[d] for d in decades]
        ax.plot(range(len(decades)), values, marker="o", linewidth=2.5, label=pair, markersize=8)
    
    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels(decades, rotation=45)
    ax.set_ylabel("Correlation Coefficient", fontweight="bold")
    ax.set_xlabel("Decade", fontweight="bold")
    ax.set_title("How Key Feature Correlations Change Over Time", fontweight="bold", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)
    
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_changes_over_time.png", dpi=300, bbox_inches="tight")
    logger.info("Saved: correlation_changes_over_time.png")
    plt.close(fig)


def generate_report(
    pearson_corr: pd.DataFrame,
    decade_corrs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    changes_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate correlation analysis report."""
    report_path = output_dir / "correlation_analysis_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SPOTIFY AUDIO FEATURE CORRELATIONS ANALYSIS (1960-2026)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RESEARCH QUESTION:\n")
        f.write("Which audio features correlate with each other, and how do these relationships change over time?\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Compute Pearson (linear) and Spearman (rank) correlations\n")
        f.write("2. Global correlation matrix for entire dataset\n")
        f.write("3. Decade-specific correlations to track changes\n")
        f.write("4. Identify strongest positive/negative relationships\n\n")
        
        f.write("STRONGEST CORRELATIONS (ALL DATA):\n")
        f.write("-" * 80 + "\n\n")
        
        top_corrs = extract_top_correlations(pearson_corr, top_n=10)
        
        # Positive
        positive = top_corrs[top_corrs["correlation"] > 0].nlargest(5, "correlation")
        f.write("STRONGEST POSITIVE CORRELATIONS:\n")
        for _, row in positive.iterrows():
            f.write(f"  • {row['feature_1'].upper()} ↔ {row['feature_2'].upper()}\n")
            f.write(f"    Correlation: {row['correlation']:.4f}\n")
            f.write(f"    Interpretation: These features tend to increase/decrease together\n\n")
        
        # Negative
        negative = top_corrs[top_corrs["correlation"] < 0].nsmallest(5, "correlation")
        f.write("STRONGEST NEGATIVE CORRELATIONS:\n")
        for _, row in negative.iterrows():
            f.write(f"  • {row['feature_1'].upper()} ↔ {row['feature_2'].upper()}\n")
            f.write(f"    Correlation: {row['correlation']:.4f}\n")
            f.write(f"    Interpretation: These features move in opposite directions\n\n")
        
        f.write("CORRELATION CHANGES ACROSS DECADES:\n")
        f.write("-" * 80 + "\n\n")
        
        for _, row in changes_df.iterrows():
            pair = row["feature_pair"]
            decades = [col for col in changes_df.columns if col != "feature_pair"]
            first_val = row[decades[0]]
            last_val = row[decades[-1]]
            change = last_val - first_val
            
            f.write(f"{pair.upper()}:\n")
            f.write(f"  {decades[0]}: {first_val:.4f}\n")
            f.write(f"  {decades[-1]}: {last_val:.4f}\n")
            f.write(f"  Change: {change:+.4f}\n")
            if abs(change) > 0.2:
                f.write(f"  ** SIGNIFICANT CHANGE **\n")
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
    changes_df.to_csv(output_dir / "correlation_changes_by_decade.csv", index=False)
    logger.info("Saved: correlation CSV files")


def run(
    input_csv: str | None = None,
    output_dir: str | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run correlation analysis pipeline.
    
    Returns:
        (global_pearson, decade_corrs_dict)
    """
    if input_csv is None:
        input_csv = str(PROJECT_ROOT / "data/processed/Spotify_1960_2026_Final.csv")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "reports")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_csv)
    df = add_decade(df)
    
    # Global correlations
    pearson_corr, spearman_corr = compute_global_correlations(df)
    
    # Decade correlations
    decade_corrs = compute_decade_correlations(df)
    
    # Correlation changes
    changes_df = analyze_correlation_changes(pearson_corr, decade_corrs)
    
    # Generate outputs
    plot_global_heatmap(pearson_corr, spearman_corr, output_path)
    plot_decade_heatmaps(decade_corrs, output_path)
    plot_strongest_correlations(pearson_corr, output_path)
    plot_correlation_changes(changes_df, output_path)
    save_statistics(pearson_corr, spearman_corr, changes_df, output_path)
    generate_report(pearson_corr, decade_corrs, changes_df, output_path)
    
    logger.info(f"Correlation analysis complete. Results saved to: {output_path}")
    
    return pearson_corr, decade_corrs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze correlations between audio features."
    )
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