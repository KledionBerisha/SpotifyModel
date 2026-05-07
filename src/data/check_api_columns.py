# File: src/data/check_api_columns.py
import pandas as pd

INPUT = "data/processed/Spotify_1980_2025_Final.csv"
API_YEARS = range(2021, 2026)
COLUMNS = [
    "artists","name","duration_ms","year","acousticness","danceability","energy",
    "instrumentalness","liveness","loudness","speechiness","tempo","valence","popularity"
]

def main():
    parts = []
    for chunk in pd.read_csv(INPUT, usecols=lambda c: c in COLUMNS, chunksize=200000, low_memory=False):
        chunk["year"] = pd.to_numeric(chunk.get("year"), errors="coerce")
        part = chunk[(chunk["year"] >= 2021) & (chunk["year"] <= 2025)]
        if not part.empty:
            parts.append(part)
    if not parts:
        print("NO_API_ROWS")
        return
    df = pd.concat(parts, ignore_index=True)
    print("API_ROWS", len(df))
    for c in COLUMNS:
        if c not in df.columns:
            print(f"{c}: MISSING")
            continue
        s = df[c].dropna()
        unique = s.nunique(dropna=True)
        print(f"{c}: nonnull={len(s)} unique={unique}")
        if unique == 0:
            continue
        if unique == 1:
            print("  SINGLE_VALUE:", s.unique()[0])
        else:
            top = s.value_counts().head(5)
            print("  TOP_FREQS:")
            for val, cnt in top.items():
                print(f"    {cnt:6d}  {repr(val)}")
    print("\nSAMPLE ROWS (first 10):")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()