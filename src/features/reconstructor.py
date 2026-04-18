def apply_update(bids: dict, asks: dict, message: dict):
    is_snapshot = message["type"] == "snapshot"
    
    if is_snapshot:
        bids.clear()
        asks.clear()
    
    for price, size in message["data"]["b"]:
        price, size = float(price), float(size)
        if size == 0:
            bids.pop(price, None)  # pop with default avoids KeyError
        else:
            bids[price] = size
    
    for price, size in message["data"]["a"]:
        price, size = float(price), float(size)
        if size == 0:
            asks.pop(price, None)
        else:
            asks[price] = size

def compute_book_features(bids: dict, asks: dict) -> dict:
    if not bids or not asks:
      return None
    sorted_bids = sorted(bids.keys(), reverse=True)
    sorted_asks = sorted(asks.keys())
    best_bid = sorted_bids[0]
    best_ask = sorted_asks[0]
    bid_size = bids[best_bid]
    ask_size = asks[best_ask]
    
    spread = best_ask - best_bid
    microprice = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
    midprice = (best_bid + best_ask) / 2
    
    features = {
        "spread": spread,
        "microprice": microprice,
        "midprice": midprice,
    }
    
    for n in [1, 5, 10, 25]:
        bid_vol = sum(bids[p] for p in sorted_bids[:n])
        ask_vol = sum(asks[p] for p in sorted_asks[:n])
        features[f"depth_imbalance_{n}"] = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    bid_vol_5 = sum(bids[p] for p in sorted_bids[:5])
    bid_vol_25 = sum(bids[p] for p in sorted_bids[:25])
    ask_vol_5 = sum(asks[p] for p in sorted_asks[:5])
    ask_vol_25 = sum(asks[p] for p in sorted_asks[:25])
    
    features["bid_pressure"] = bid_vol_5 / bid_vol_25
    features["ask_pressure"] = ask_vol_5 / ask_vol_25
    features["pressure_imbalance"] = features["bid_pressure"] - features["ask_pressure"]
    
    return features

  
import json
import os
import pandas as pd
import zipfile
import time

def reconstruct_and_extract_from_state(ob_zip_path, trades_df, bids, asks):
    results = []
    trade_times = trades_df["timestamp"].astype("int64") // 10**6
    trade_times = trade_times.values
    trade_idx = 0
    n_trades = len(trade_times)
    
    with zipfile.ZipFile(ob_zip_path, "r") as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as f:
            for line in f:
                msg = json.loads(line)
                book_ts = msg["ts"]
                
                while trade_idx < n_trades and trade_times[trade_idx] < book_ts:
                    features = compute_book_features(bids, asks)
                    if features is not None:
                        features["timestamp"] = trade_times[trade_idx]
                        results.append(features)
                    trade_idx += 1
                
                apply_update(bids, asks, msg)
    
    while trade_idx < n_trades:
        features = compute_book_features(bids, asks)
        if features is not None:
            features["timestamp"] = trade_times[trade_idx]
            results.append(features)
        trade_idx += 1
    
    return pd.DataFrame(results)

trades_df = pd.read_parquet("data/processed/BTCUSDT/week1.parquet")

def process_week(asset: str, week: str, week_dates: list, 
                 ob_dir: str, trades_path: str, output_dir: str):
    trades_df = pd.read_parquet(trades_path)
    trade_ts_ms = trades_df["timestamp"].astype("int64") // 10**6
    
    os.makedirs(output_dir, exist_ok=True)
    bids = {}
    asks = {}
    
    for date_str in week_dates:
        zip_name = f"{date_str}_{asset}_ob500.data.zip"
        zip_path = os.path.join(ob_dir, zip_name)
        
        if not os.path.exists(zip_path):
            print(f"  MISSING: {zip_name}, skipping")
            continue
        
        day_start = pd.Timestamp(date_str).value // 10**6
        day_end = pd.Timestamp(pd.Timestamp(date_str) + pd.Timedelta(days=1)).value // 10**6
        day_mask = (trade_ts_ms >= day_start) & (trade_ts_ms < day_end)
        day_trades = trades_df[day_mask].copy()
        
        t0 = time.time()
        day_features = reconstruct_and_extract_from_state(
            zip_path, day_trades, bids, asks
        )
        t1 = time.time()
        
        out_path = os.path.join(output_dir, f"{asset}_{date_str}_book_features.parquet")
        day_features.to_parquet(out_path, index=False)
        print(f"  {date_str}: {len(day_features)} rows in {t1-t0:.0f}s -> {out_path}")

week1_dates = ["2024-09-09", "2024-09-10", "2024-09-11", "2024-09-12",
               "2024-09-13", "2024-09-14", "2024-09-15"]
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WEEKS = {
    "week1": ["2024-09-09", "2024-09-10", "2024-09-11", "2024-09-12",
              "2024-09-13", "2024-09-14", "2024-09-15"],
    "week2": ["2024-10-28", "2024-10-29", "2024-10-30", "2024-10-31",
              "2024-11-01", "2024-11-02", "2024-11-03"],
    "week3": ["2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27",
              "2025-02-28", "2025-03-01", "2025-03-02"],
}

if __name__ == "__main__":

    for asset in ASSETS:
        for week_name, dates in WEEKS.items():
            trades_path = f"data/processed/{asset}/{week_name}.parquet"
            if not os.path.exists(trades_path):
                print(f"\nSKIPPING {asset} {week_name} — no trade parquet")
                continue
            ob_dir = f"data/raw/orderbook/{asset}"
            print(f"\n=== {asset} {week_name} ===")
            process_week(asset, week_name, dates, ob_dir, trades_path,
                        "data/processed/features")