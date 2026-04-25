"""Download and process the Kaggle Spotify dataset for years 1960-2020."""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATASET = "yamaerenay/spotify-dataset-19212020-600k-tracks"


def _find_csv_file(dataset_dir: Path) -> Path:
    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")
    # Prefer files that look like the main tracks table.
    for candidate in csv_files:
        name = candidate.name.lower()
        if "track" in name or "spotify" in name:
            return candidate
    return csv_files[0]


def run(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    if "year" not in df.columns:
        raise ValueError("Input dataset must contain a 'year' column.")

    filtered = df[(df["year"] >= 1960) & (df["year"] <= 2020)].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)


def run_from_kaggle(output_csv: Path, dataset_ref: str = DEFAULT_DATASET) -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is not installed. Install it with: pip install kagglehub"
        ) from exc

    dataset_path = Path(kagglehub.dataset_download(dataset_ref))
    input_csv = _find_csv_file(dataset_path)
    run(input_csv=input_csv, output_csv=output_csv)
    print(f"Downloaded dataset to: {dataset_path}")
    print(f"Using input file: {input_csv}")
    print(f"Saved filtered output: {output_csv}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Kaggle Spotify dataset for years 1960-2020."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/interim/kaggle_1960_2020.csv"),
        help="Output CSV path for filtered dataset.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional local input CSV. If omitted, data is downloaded from Kaggle.",
    )
    parser.add_argument(
        "--dataset-ref",
        type=str,
        default=DEFAULT_DATASET,
        help="Kaggle dataset reference for kagglehub download.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.input_csv is not None:
        run(input_csv=args.input_csv, output_csv=args.output_csv)
        print(f"Processed local input: {args.input_csv}")
        print(f"Saved filtered output: {args.output_csv}")
    else:
        run_from_kaggle(output_csv=args.output_csv, dataset_ref=args.dataset_ref)
