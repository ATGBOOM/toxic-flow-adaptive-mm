# loader.py

from pathlib import Path

import pandas as pd


def load_trades(file_path: str | Path) -> pd.DataFrame:
    """Load a single gzipped CSV trade file and return a cleaned DataFrame.

    Args:
        file_path: Path to a .csv.gz trade file containing timestamp, side,
            size, and price columns.

    Returns:
        DataFrame with columns [timestamp, price, qty, sign] where timestamp
        is a datetime, qty is trade size, and sign is +1 (buy) or -1 (sell).
    """
    df = pd.read_csv(
        file_path,
        usecols=["timestamp", "side", "size", "price"],
        compression="gzip"  # files are .csv.gz
    )

    # timestamp is Unix seconds as float — convert to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # standardise direction to +1/-1 (Buy = +1, Sell = -1)
    df["sign"] = df["side"].map({"Buy": 1, "Sell": -1})
    df.drop(columns=["side"], inplace=True)

    # rename size -> qty to keep downstream code consistent
    df.rename(columns={"size": "qty"}, inplace=True)

    # gap detection
    gaps = df["timestamp"].diff()
    large_gaps = gaps[gaps > pd.Timedelta(seconds=60)]
    if not large_gaps.empty:
        print(f"  Warning: {len(large_gaps)} gaps. Largest: {large_gaps.max()}")

    return df[["timestamp", "price", "qty", "sign"]]
