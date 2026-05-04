import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

# Project root for relative paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse script arguments for flexible input/output paths."""
    parser = argparse.ArgumentParser(
        description="Create visualizations for Spotify_1960_2026_Final.csv"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(PROJECT_ROOT / "data/processed/Spotify_1960_2026_Final.csv"),
        help="Path to merged Spotify dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports/figures"),
        help="Directory where all charts will be saved.",
    )
    return parser.parse_args()


def add_figure_caption(fig: plt.Figure, caption_text: str) -> None:
    """Add a figure caption below the image (for report template compatibility)."""
    fig.text(0.5, 0.01, caption_text, ha="center", va="bottom", fontsize=11)



def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert selected columns to numeric type safely."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def create_line_trend_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Figura 1: Long-term trends in loudness, danceability, and energy.
    Shows how production intensity and rhythmic characteristics evolved over time.
    """
    yearly = (
        df.groupby("year", as_index=False)[["loudness", "danceability", "energy"]]
        .mean()
        .sort_values("year")
    )

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")

    sns.lineplot(data=yearly, x="year", y="loudness", label="Loudness", linewidth=2.2, ax=ax)
    sns.lineplot(data=yearly, x="year", y="danceability", label="Danceability", linewidth=2.2, ax=ax)
    sns.lineplot(data=yearly, x="year", y="energy", label="Energy", linewidth=2.2, ax=ax)

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Feature Value")
    ax.legend(title="Audio Features")
    ax.grid(alpha=0.25)

    add_figure_caption(
        fig,
        "Figura 1. Evolución anual de loudness, danceability y energy (1960–2026).",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_dir / "figura_1_line_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: figura_1_line_trends.png")


def create_decade_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Figura 2: Danceability distribution by decade.
    Evaluates whether music became more standardized across decades.
    """
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    df["decade_label"] = df["decade"].astype("Int64").astype(str) + "s"

    decade_order = sorted(df["decade_label"].dropna().unique(), key=lambda x: int(x[:-1]))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")
    sns.boxplot(
        data=df,
        x="decade_label",
        y="danceability",
        order=decade_order,
        ax=ax,
        color="#8ecae6",
        fliersize=2,
    )

    ax.set_xlabel("Decade")
    ax.set_ylabel("Danceability")
    ax.grid(axis="y", alpha=0.2)

    add_figure_caption(
        fig,
        "Figura 2. Distribución de danceability por década (análisis de estandarización).",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_dir / "figura_2_danceability_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: figura_2_danceability_boxplot.png")


def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Figura 3: Correlations among audio features.
    Reveals which musical attributes tend to increase or decrease together.
    """
    corr_features = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "duration_ms",
        "year",
    ]
    corr_features = [c for c in corr_features if c in df.columns]

    corr_matrix = df[corr_features].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    add_figure_caption(
        fig,
        "Figura 3. Matriz de correlación entre variables de audio.",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_dir / "figura_3_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: figura_3_correlation_heatmap.png")


def create_interactive_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Figura 4 (BONUS): Interactive scatter of year vs energy, colored by loudness.
    Point size uses popularity; hover shows song and artist.
    """
    viz_df = df.copy()

    if "popularity" not in viz_df.columns:
        viz_df["popularity"] = 50

    fig = px.scatter(
        viz_df,
        x="year",
        y="energy",
        size="popularity",
        color="loudness",
        hover_data=["name", "artists"],
        title="Figura 4. Dispersión interactiva de energy por año (color: loudness, tamaño: popularity).",
        color_continuous_scale="Viridis",
        opacity=0.7,
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(output_dir / "figura_4_interactive_scatter.html", include_plotlyjs="cdn")
    logger.info("Saved: figura_4_interactive_scatter.html")


def main() -> None:
    """Main pipeline: load data, ensure numeric columns, create visualizations."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    logger.info(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    numeric_columns = [
        "year",
        "loudness",
        "danceability",
        "energy",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
        "valence",
        "duration_ms",
        "popularity",
    ]
    df = ensure_numeric(df, numeric_columns)

    # Keep target timeline
    df = df[(df["year"] >= 1960) & (df["year"] <= 2026)].copy()
    logger.info(f"Filtered to {len(df)} rows for years 1960-2026")

    create_line_trend_plot(df, output_dir)
    create_decade_boxplot(df, output_dir)
    create_correlation_heatmap(df, output_dir)
    create_interactive_scatter(df, output_dir)

    logger.info(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()