"""Cluster tracks by audio profile and analyze cluster shifts by decade."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features.preprocess import AUDIO_FEATURE_COLUMNS, load_clean_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data/processed/Spotify_1980_2025_Final_clean.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_CLUSTERED_CSV = DEFAULT_REPORT_DIR / "clustered_tracks.csv"
DEFAULT_CLUSTER_SUMMARY_CSV = DEFAULT_REPORT_DIR / "cluster_summary.csv"
DEFAULT_CLUSTER_DECADE_CSV = DEFAULT_REPORT_DIR / "cluster_decade_distribution.csv"
DEFAULT_CLUSTER_SCORE_CSV = DEFAULT_REPORT_DIR / "cluster_candidate_scores.csv"
DEFAULT_CLUSTER_PLOT = DEFAULT_REPORT_DIR / "cluster_pca.png"
DEFAULT_ALT_CLUSTER_CSV = DEFAULT_REPORT_DIR / "cluster_alternative_labels.csv"

logger = logging.getLogger(__name__)


def build_cluster_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a cluster-ready frame using the audio feature set."""
    frame = df.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    for column in AUDIO_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["year"] + AUDIO_FEATURE_COLUMNS).copy()
    frame["decade"] = (frame["year"].astype(int) // 10) * 10
    frame["decade_label"] = frame["decade"].astype(str) + "s"
    return frame.reset_index(drop=True)


def scale_features(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize the audio feature matrix."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def evaluate_kmeans_candidates(features: pd.DataFrame, min_k: int = 2, max_k: int = 8) -> pd.DataFrame:
    """Evaluate KMeans candidates using inertia and silhouette score."""
    sample = features
    if len(sample) > 10000:
        sample = sample.sample(n=10000, random_state=42)

    scaled, _ = scale_features(sample)

    records: list[dict[str, float]] = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels) if len(set(labels)) > 1 else float("nan")
        records.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(score),
            }
        )

    return pd.DataFrame(records)


def fit_kmeans(features: pd.DataFrame, k: int) -> tuple[KMeans, np.ndarray, StandardScaler]:
    """Fit KMeans on standardized audio features."""
    scaled, scaler = scale_features(features)
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(scaled)
    return model, labels, scaler


def fit_alternative_model(features: pd.DataFrame, k: int) -> tuple[str, np.ndarray]:
    """Fit an alternative clustering model for comparison."""
    scaled, _ = scale_features(features)

    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm_labels = gmm.fit_predict(scaled)
    if len(set(gmm_labels)) > 1:
        gmm_score = silhouette_score(scaled, gmm_labels)
    else:
        gmm_score = float("nan")

    dbscan = DBSCAN(eps=1.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(scaled)
    if len(set(dbscan_labels)) > 1 and len(set(dbscan_labels)) > 1:
        dbscan_score = silhouette_score(scaled, dbscan_labels)
    else:
        dbscan_score = float("nan")

    alt_df = pd.DataFrame(
        [
            {"method": "gmm", "k": k, "silhouette": float(gmm_score)},
            {"method": "dbscan", "k": -1, "silhouette": float(dbscan_score)},
        ]
    )
    alt_df.to_csv(DEFAULT_ALT_CLUSTER_CSV, index=False)

    best_method = "gmm"
    best_labels = gmm_labels
    if not np.isnan(dbscan_score) and (np.isnan(gmm_score) or dbscan_score > gmm_score):
        best_method = "dbscan"
        best_labels = dbscan_labels

    return best_method, best_labels


def plot_pca_clusters(features: pd.DataFrame, labels: np.ndarray, output_path: Path) -> None:
    """Plot a 2D PCA projection of the clustered tracks."""
    scaled, _ = scale_features(features)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    plot_df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "cluster": labels.astype(int),
        }
    )

    if len(plot_df) > 5000:
        plot_df = plot_df.sample(n=5000, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    scatter = ax.scatter(
        plot_df["pc1"],
        plot_df["pc2"],
        c=plot_df["cluster"],
        cmap="tab10",
        s=14,
        alpha=0.75,
    )
    ax.set_title("Spotify Audio Clusters in PCA Space")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean feature values by cluster."""
    summary = df.groupby("cluster")[AUDIO_FEATURE_COLUMNS].mean().reset_index()
    counts = df.groupby("cluster").size().reset_index(name="track_count")
    return summary.merge(counts, on="cluster").sort_values("cluster").reset_index(drop=True)


def summarize_clusters_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the cluster share by decade."""
    counts = df.groupby(["decade_label", "cluster"]).size().reset_index(name="track_count")
    totals = counts.groupby("decade_label")["track_count"].transform("sum")
    counts["share"] = counts["track_count"] / totals
    return counts.sort_values(["decade_label", "cluster"]).reset_index(drop=True)


def run(
    input_csv: str | None = None,
    report_dir: str | None = None,
    min_k: int = 2,
    max_k: int = 8,
) -> dict[str, object]:
    """Fit clusters, save summaries, and create plots."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    source_path = Path(input_csv) if input_csv is not None else DEFAULT_INPUT_CSV
    output_dir = Path(report_dir) if report_dir is not None else DEFAULT_REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_dataset(source_path)
    cluster_frame = build_cluster_frame(df)

    if cluster_frame.empty:
        raise ValueError("No rows with complete audio features were found for clustering.")

    features = cluster_frame[AUDIO_FEATURE_COLUMNS]
    score_df = evaluate_kmeans_candidates(features, min_k=min_k, max_k=max_k)
    best_row = score_df.dropna(subset=["silhouette"]).sort_values(["silhouette", "inertia"], ascending=[False, True]).iloc[0]
    best_k = int(best_row["k"])

    kmeans_model, labels, _ = fit_kmeans(features, best_k)
    cluster_frame = cluster_frame.copy()
    cluster_frame["cluster"] = labels

    cluster_summary = summarize_clusters(cluster_frame)
    cluster_decade = summarize_clusters_by_decade(cluster_frame)

    best_alt_method, alt_labels = fit_alternative_model(features, best_k)

    score_df.to_csv(DEFAULT_CLUSTER_SCORE_CSV, index=False)
    cluster_frame.to_csv(DEFAULT_CLUSTERED_CSV, index=False)
    cluster_summary.to_csv(DEFAULT_CLUSTER_SUMMARY_CSV, index=False)
    cluster_decade.to_csv(DEFAULT_CLUSTER_DECADE_CSV, index=False)
    plot_pca_clusters(features, labels, DEFAULT_CLUSTER_PLOT)

    alt_output = cluster_frame[["artists", "name", "year"]].copy()
    alt_output["alternative_method"] = best_alt_method
    alt_output["alternative_cluster"] = alt_labels
    alt_output.to_csv(DEFAULT_ALT_CLUSTER_CSV, index=False)

    logger.info("Loaded %d rows from %s", len(df), source_path)
    logger.info("Clustered %d rows with best k=%d", len(cluster_frame), best_k)
    logger.info("Alternative clustering method selected: %s", best_alt_method)
    logger.info("Saved clustered tracks to %s", DEFAULT_CLUSTERED_CSV)

    return {
        "best_k": best_k,
        "silhouette_scores": score_df,
        "cluster_frame": cluster_frame,
        "cluster_summary": cluster_summary,
        "cluster_decade": cluster_decade,
        "alternative_method": best_alt_method,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Cluster Spotify tracks by audio profile.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to the cleaned Spotify dataset (default: data/processed/Spotify_1980_2025_Final_clean.csv)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for clustering outputs (default: reports/)",
    )
    parser.add_argument("--min-k", type=int, default=2, help="Minimum number of clusters to evaluate")
    parser.add_argument("--max-k", type=int, default=8, help="Maximum number of clusters to evaluate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(input_csv=args.input_csv, report_dir=args.report_dir, min_k=args.min_k, max_k=args.max_k)