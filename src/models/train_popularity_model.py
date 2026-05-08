"""Train and evaluate a model to predict track popularity."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.preprocess import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_model_frame,
    load_clean_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data/processed/Spotify_1980_2025_Final_clean.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "popularity_model.joblib"
DEFAULT_METRICS_PATH = DEFAULT_REPORT_DIR / "popularity_model_metrics.csv"
DEFAULT_IMPORTANCE_PATH = DEFAULT_REPORT_DIR / "popularity_feature_importance.csv"
DEFAULT_SHAP_PATH = DEFAULT_REPORT_DIR / "popularity_shap_values.csv"
DEFAULT_REPORT_PATH = DEFAULT_REPORT_DIR / "popularity_model_report.txt"

logger = logging.getLogger(__name__)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a frame into train, validation, and test sets by time."""
    ordered_years = sorted(df["year"].dropna().astype(int).unique().tolist())
    if len(ordered_years) < 3:
        ordered = df.sort_values("year").reset_index(drop=True)
        n_rows = len(ordered)
        train_end = max(1, int(round(n_rows * 0.6)))
        val_end = max(train_end + 1, int(round(n_rows * 0.8)))
        return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:val_end].copy(), ordered.iloc[val_end:].copy()

    test_year_count = max(1, int(round(len(ordered_years) * 0.2)))
    val_year_count = max(1, int(round(len(ordered_years) * 0.15)))
    if test_year_count + val_year_count >= len(ordered_years):
        test_year_count = 1
        val_year_count = 1

    train_years = ordered_years[: len(ordered_years) - test_year_count - val_year_count]
    val_years = ordered_years[len(ordered_years) - test_year_count - val_year_count : len(ordered_years) - test_year_count]
    test_years = ordered_years[len(ordered_years) - test_year_count :]

    if not train_years:
        ordered = df.sort_values("year").reset_index(drop=True)
        n_rows = len(ordered)
        train_end = max(1, int(round(n_rows * 0.6)))
        val_end = max(train_end + 1, int(round(n_rows * 0.8)))
        return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:val_end].copy(), ordered.iloc[val_end:].copy()

    train_df = df[df["year"].isin(train_years)].copy()
    val_df = df[df["year"].isin(val_years)].copy()
    test_df = df[df["year"].isin(test_years)].copy()
    return train_df, val_df, test_df


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Evaluate a fitted model."""
    predictions = model.predict(features)
    return regression_metrics(target, predictions)


def build_models() -> dict[str, object]:
    """Build the required baseline and stronger models."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }


def fit_models(train_df: pd.DataFrame) -> dict[str, object]:
    """Fit all candidate models."""
    models = build_models()
    for model in models.values():
        model.fit(train_df[MODEL_FEATURE_COLUMNS], train_df[TARGET_COLUMN])
    return models


def extract_feature_importance(model) -> pd.DataFrame:
    """Extract feature importance or coefficient magnitude."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float))
    else:
        values = np.zeros(len(MODEL_FEATURE_COLUMNS), dtype=float)

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)
    return importance.reset_index(drop=True)


def compute_shap_values(model, background: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP values for tree-based models when possible."""
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_df = pd.DataFrame(shap_values, columns=MODEL_FEATURE_COLUMNS, index=sample.index)
    return shap_df


def build_report(
    best_model_name: str,
    best_val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_rows: int,
    val_rows: int,
    test_rows: int,
) -> str:
    """Build a human-readable report."""
    return "\n".join(
        [
            "SPOTIFY POPULARITY MODEL REPORT",
            "=" * 80,
            f"Best model: {best_model_name}",
            f"Train rows: {train_rows}",
            f"Validation rows: {val_rows}",
            f"Test rows: {test_rows}",
            "",
            f"Validation MAE: {best_val_metrics['mae']:.4f}",
            f"Validation RMSE: {best_val_metrics['rmse']:.4f}",
            f"Validation R2: {best_val_metrics['r2']:.4f}",
            "",
            f"Test MAE: {test_metrics['mae']:.4f}",
            f"Test RMSE: {test_metrics['rmse']:.4f}",
            f"Test R2: {test_metrics['r2']:.4f}",
            "",
            "Features used:",
            ", ".join(MODEL_FEATURE_COLUMNS),
            "",
            "Baseline and stronger models used:",
            "Linear Regression, Random Forest Regressor, Gradient Boosting Regressor",
        ]
    )


def run(
    input_csv: str | None = None,
    model_dir: str | None = None,
    report_dir: str | None = None,
) -> dict[str, object]:
    """Train, evaluate, and save the popularity model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    source_path = Path(input_csv) if input_csv is not None else DEFAULT_INPUT_CSV
    model_path = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    output_report_dir = Path(report_dir) if report_dir is not None else DEFAULT_REPORT_DIR
    model_path.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_dataset(source_path)
    model_frame = build_model_frame(df, drop_missing_target=True)

    if model_frame.empty:
        raise ValueError("No labeled rows were found. You need rows with non-missing popularity before training.")

    train_df, val_df, test_df = time_split(model_frame)
    if val_df.empty or test_df.empty:
        raise ValueError("Not enough year coverage to build validation and test splits.")

    fitted_models = fit_models(train_df)

    metrics_records: list[dict[str, object]] = []
    validation_scores: dict[str, dict[str, float]] = {}
    test_scores: dict[str, dict[str, float]] = {}

    for model_name, model in fitted_models.items():
        val_metrics = evaluate_model(model, val_df[MODEL_FEATURE_COLUMNS], val_df[TARGET_COLUMN])
        test_metrics = evaluate_model(model, test_df[MODEL_FEATURE_COLUMNS], test_df[TARGET_COLUMN])

        validation_scores[model_name] = val_metrics
        test_scores[model_name] = test_metrics

        for split_name, metrics in (("validation", val_metrics), ("test", test_metrics)):
            metrics_records.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                }
            )

    metrics_df = pd.DataFrame(metrics_records)
    best_model_name = (
        metrics_df[metrics_df["split"] == "validation"]
        .sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
        .iloc[0]["model"]
    )
    best_model = fitted_models[best_model_name]
    best_validation_metrics = validation_scores[best_model_name]
    best_test_metrics = test_scores[best_model_name]

    importance_df = extract_feature_importance(best_model)
    shap_df = compute_shap_values(best_model, train_df[MODEL_FEATURE_COLUMNS], test_df[MODEL_FEATURE_COLUMNS])
    report_text = build_report(
        best_model_name,
        best_validation_metrics,
        best_test_metrics,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    joblib.dump(best_model, DEFAULT_MODEL_PATH)
    metrics_df.to_csv(DEFAULT_METRICS_PATH, index=False)
    importance_df.to_csv(DEFAULT_IMPORTANCE_PATH, index=False)
    if not shap_df.empty:
        shap_df.to_csv(DEFAULT_SHAP_PATH, index=False)
    DEFAULT_REPORT_PATH.write_text(report_text, encoding="utf-8")

    logger.info("Loaded %d labeled rows from %s", len(model_frame), source_path)
    logger.info("Train/validation/test rows: %d / %d / %d", len(train_df), len(val_df), len(test_df))
    logger.info("Best model: %s", best_model_name)
    logger.info("Saved model to %s", DEFAULT_MODEL_PATH)
    logger.info("Saved metrics to %s", DEFAULT_METRICS_PATH)
    logger.info("Saved feature importance to %s", DEFAULT_IMPORTANCE_PATH)

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "metrics": metrics_df,
        "feature_importance": importance_df,
        "shap_values": shap_df,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a Spotify popularity model.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to the cleaned Spotify dataset (default: data/processed/Spotify_1980_2025_Final_clean.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory where the trained model will be saved (default: models/)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for metrics and reports (default: reports/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(input_csv=args.input_csv, model_dir=args.model_dir, report_dir=args.report_dir)