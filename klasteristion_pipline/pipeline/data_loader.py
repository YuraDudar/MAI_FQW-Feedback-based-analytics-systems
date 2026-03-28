"""
Data Ingestion: CSV loading, deduplication, basic statistics.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from config import (
    CSV_DIR,
    TEXT_FIELDS,
    TAGS_FIELD,
    RATING_FIELD,
    REVIEW_ID_FIELD,
)

log = logging.getLogger(__name__)


def find_latest_csv(csv_dir: Path = CSV_DIR) -> Path:
    """Return the most recently modified CSV in *csv_dir*."""
    csvs = sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    log.info("Using CSV: %s", csvs[0].name)
    return csvs[0]


def load_reviews(
    csv_path: str | Path | None = None,
    *,
    deduplicate: bool = True,
    print_stats: bool = True,
) -> pd.DataFrame:
    """
    Load reviews from CSV, deduplicate, and cast types.

    Returns a DataFrame with cleaned column types ready for preprocessing.
    """
    if csv_path is None:
        csv_path = find_latest_csv()
    csv_path = Path(csv_path)

    log.info("Loading reviews from %s …", csv_path.name)
    df = pd.read_csv(csv_path, dtype=str)
    n_raw = len(df)

    if REVIEW_ID_FIELD in df.columns and deduplicate:
        df = df.drop_duplicates(subset=[REVIEW_ID_FIELD], keep="first")
    n_dedup = len(df)

    if RATING_FIELD in df.columns:
        df[RATING_FIELD] = pd.to_numeric(df[RATING_FIELD], errors="coerce").fillna(0).astype(int)

    for col in TEXT_FIELDS + [TAGS_FIELD]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            df.loc[df[col].isin(["nan", "None", "none", ""]), col] = ""

    if print_stats:
        _print_stats(df, n_raw, n_dedup)

    return df


def _print_stats(df: pd.DataFrame, n_raw: int, n_dedup: int) -> None:
    print("\n" + "=" * 60)
    print("  DATA LOADING SUMMARY")
    print("=" * 60)
    print(f"  Raw rows        : {n_raw:,}")
    print(f"  After dedup     : {n_dedup:,}  (removed {n_raw - n_dedup:,})")
    print()

    rows = []
    for col in TEXT_FIELDS + [TAGS_FIELD]:
        if col not in df.columns:
            continue
        filled = (df[col] != "").sum()
        lengths = df.loc[df[col] != "", col].str.len()
        rows.append({
            "field": col,
            "filled": filled,
            "fill_%": f"{filled / len(df) * 100:.1f}%",
            "avg_len": f"{lengths.mean():.0f} ch" if len(lengths) else "—",
            "max_len": f"{lengths.max():.0f} ch" if len(lengths) else "—",
        })
    print(tabulate(rows, headers="keys", tablefmt="simple"))

    if RATING_FIELD in df.columns:
        print("\n  RATING DISTRIBUTION")
        print("  " + "-" * 40)
        for star in range(1, 6):
            cnt = (df[RATING_FIELD] == star).sum()
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 2)
            print(f"  {star}★  {cnt:>5}  ({pct:5.1f}%)  {bar}")
        neg = (df[RATING_FIELD] <= 3).sum()
        pos = (df[RATING_FIELD] >= 4).sum()
        print(f"\n  Negative (1-3★): {neg}  ({neg / len(df) * 100:.1f}%)")
        print(f"  Positive (4-5★): {pos}  ({pos / len(df) * 100:.1f}%)")

    print("=" * 60 + "\n")
