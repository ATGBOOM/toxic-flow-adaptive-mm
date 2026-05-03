# pipeline.py

from pathlib import Path

import pandas as pd

from loader import load_trades

RAW_TRADES = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'trades'
PROCESSED_ROOT = Path(__file__).parent.parent.parent / 'data' / 'processed'

WEEK_RANGES = {
    'week1': ('2024-09-09', '2024-09-15'),
    'week2': ('2024-10-28', '2024-11-03'),
    'week3': ('2025-02-24', '2025-03-02'),
}


def date_to_week(date_str: str) -> str | None:
    """Map a date string to the corresponding week label.

    Args:
        date_str: Date in 'YYYY-MM-DD' format.

    Returns:
        Week label (e.g. 'week1') if the date falls within a known range,
        otherwise None.
    """
    for week, (start, end) in WEEK_RANGES.items():
        if start <= date_str <= end:
            return week
    return None


def process_all() -> None:
    """Process all raw trade files and write per-asset per-week parquets.

    Iterates over every asset directory under RAW_TRADES, groups daily
    .csv.gz files into week buckets, concatenates them, and writes the
    result to PROCESSED_ROOT/<asset>/<week>.parquet.

    Args:
        None

    Returns:
        None
    """
    for asset_dir in sorted(RAW_TRADES.iterdir()):
        if not asset_dir.is_dir():
            continue
        asset = asset_dir.name  # e.g. "BTCUSDT"

        week_frames: dict[str, list] = {'week1': [], 'week2': [], 'week3': []}

        for file in sorted(asset_dir.iterdir()):
            if not file.name.endswith('.csv.gz'):
                continue
            # filename format: BTCUSDT2024-09-09.csv.gz
            date_str = file.name.replace(asset, '').replace('.csv.gz', '')
            week = date_to_week(date_str)
            if week is None:
                print(f"  Skipping {file.name} — no matching week")
                continue
            df = load_trades(file)
            week_frames[week].append(df)

        for week, frames in week_frames.items():
            if not frames:
                print(f"  No data for {asset}/{week}")
                continue
            combined = pd.concat(frames, ignore_index=True)
            out = PROCESSED_ROOT / asset / f"{week}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(out)
            print(f"  Saved {asset}/{week}: {len(combined):,} rows")


if __name__ == "__main__":
    process_all()
