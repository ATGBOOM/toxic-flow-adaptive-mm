# pipeline.py

import pandas as pd
from pathlib import Path
from loader import load_trades

RAW_TRADES = Path('data/raw/trades')
PROCESSED_ROOT = Path('data/processed')

WEEK_RANGES = {
    'week1': ('2024-09-09', '2024-09-15'),
    'week2': ('2024-10-28', '2024-11-03'),
    'week3': ('2025-02-24', '2025-03-02'),
}

def date_to_week(date_str):
    for week, (start, end) in WEEK_RANGES.items():
        if start <= date_str <= end:
            return week
    return None

def process_all():
    for asset_dir in sorted(RAW_TRADES.iterdir()):
        if not asset_dir.is_dir():
            continue
        asset = asset_dir.name  # e.g. "BTCUSDT"
        
        week_frames = {'week1': [], 'week2': [], 'week3': []}
        
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