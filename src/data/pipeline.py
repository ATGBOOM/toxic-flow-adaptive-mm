import pandas as pd
from pathlib import Path
from loader import load_trades

RAW_ROOT = Path('data/raw')
PROCESSED_ROOT = Path('data/processed')


def load_data(folder, output_path):
    data_frames = []
    for file in sorted(folder.iterdir()):
        if file.suffix == ".csv":
            df = load_trades(file)
            data_frames.append(df)
    combined = pd.concat(data_frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)
    return combined


def process_all():
    for asset_dir in sorted(RAW_ROOT.iterdir()):
        if not asset_dir.is_dir():
            continue
        for week_dir in sorted(asset_dir.iterdir()):
            if not week_dir.is_dir():
                continue
            output_path = PROCESSED_ROOT / asset_dir.name / f"{week_dir.name}.parquet"
            print(f"Processing {asset_dir.name}/{week_dir.name} -> {output_path}")
            load_data(week_dir, output_path)


parq = process_all()

# df = pd.read_parquet("data/processed/BTCUSDT/week3.parquet")
# print(len(df))           # should be hundreds of thousands of rows for a week of BTC
# print(df["timestamp"].min(), df["timestamp"].max())  # should bracket your target week
# print(df["price"].describe())  # prices should look sensible
# print(df["sign"].value_counts())  # roughly balanced