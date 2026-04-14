# loader.py

import pandas as pd

def load_trades(file_path):
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