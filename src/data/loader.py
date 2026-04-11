
import pandas as pd

COLUMN_NAMES = ["trade_id", "price", "qty", "quote_qty", "timestamp", "is_buyer_maker", "is_best_match"]

  
def load_trades(file_path):
    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, 
                     usecols=["price", "qty", "timestamp", "is_buyer_maker"])
    
    first_ts = df["timestamp"].iloc[0]
    unit = "us" if first_ts > 1e15 else "ms"
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit)
    df["sign"] = df["is_buyer_maker"].map({True: -1, False: 1})
    df.drop(columns=["is_buyer_maker"], inplace=True)
    gaps = df["timestamp"].diff()
    large_gaps = gaps[gaps > pd.Timedelta(seconds=60)]
    if not large_gaps.empty:
        print(f"Warning: {len(large_gaps)} gaps detected. Largest: {large_gaps.max()}")

    
    return df


