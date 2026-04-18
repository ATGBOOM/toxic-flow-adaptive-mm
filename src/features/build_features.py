import pandas as pd
import numpy as np
import os
import time
from sys import path
path.insert(0, "src/models")
from vpin import compute_vpin


# import from your existing files
# from reconstructor import (
#     apply_update, compute_book_features, 
#     reconstruct_and_extract_from_state, process_week
# )

def add_trade_features(trades_df):
    """Add trade-derived features to a trades dataframe."""
    ts = trades_df["ts_ms"].values
    qty = trades_df["qty"].values
    signs = trades_df["sign"].values
    prices = trades_df["price"].values
    
    # trade intensity
    for window_s in [1, 5, 10]:
        window_ms = window_s * 1000
        lookback_idx = np.searchsorted(ts, ts - window_ms)
        trades_df[f"trade_intensity_{window_s}s"] = np.arange(len(ts)) - lookback_idx
    
    # volume acceleration
    cum_qty = np.cumsum(qty)
    idx_5s = np.searchsorted(ts, ts - 5000)
    idx_30s = np.searchsorted(ts, ts - 30000)
    vol_5s = cum_qty - cum_qty[idx_5s]
    vol_30s = cum_qty - cum_qty[idx_30s]
    expected_5s = vol_30s * (5 / 30)
    with np.errstate(invalid='ignore'):
        trades_df["volume_acceleration"] = np.where(expected_5s > 0, vol_5s / expected_5s, 1.0)
    
    # signed volume imbalance
    cum_signed_qty = np.cumsum(signs * qty)
    idx_10s = np.searchsorted(ts, ts - 10000)
    trades_df["signed_vol_imbalance_10s"] = cum_signed_qty - cum_signed_qty[idx_10s]
    
    return trades_df
# add this to build_features.py or run in notebook first to test

def add_vpin_feature(trades_df, asset, week_name):
    """Join VPIN values onto trade-level feature matrix using forward fill."""
    
    # recompute VPIN using the same code from Session 6
    trades_raw = pd.read_parquet(f"data/processed/{asset}/{week_name}.parquet")
    daily_vol = trades_raw["qty"].sum() / 7
    bucket_size = daily_vol / 50
    
    vpin_df = compute_vpin(trades_raw, bucket_size, n_buckets=50)
    
    # convert vpin timestamps to ms to match feature matrix
    vpin_df["ts_ms"] = vpin_df["timestamp"].astype("int64") // 10**6
    vpin_df = vpin_df.sort_values("ts_ms")
    
    # for each trade, find the most recent VPIN value
    trade_ts = trades_df["ts_ms"].values
    vpin_ts = vpin_df["ts_ms"].values
    vpin_vals = vpin_df["vpin"].values
    
    # searchsorted finds where each trade would insert into vpin timestamps
    # subtract 1 to get the most recent vpin BEFORE this trade
    idx = np.searchsorted(vpin_ts, trade_ts, side="right") - 1
    idx = np.clip(idx, 0, len(vpin_vals) - 1)
    
    trades_df["vpin"] = vpin_vals[idx]
    
    # trades before the first VPIN value get NaN
    trades_df.loc[trades_df["ts_ms"] < vpin_ts[0], "vpin"] = np.nan
    
    return trades_df

def add_toxicity_label(trades_df, threshold_bps=8, horizon_s=10):
    """Add toxic label based on forward price movement."""
    ts = trades_df["ts_ms"].values
    prices = trades_df["price"].values
    signs = trades_df["sign"].values
    
    horizon_ms = horizon_s * 1000
    future_idx = np.searchsorted(ts, ts + horizon_ms)
    future_idx = np.clip(future_idx, 0, len(prices) - 1)
    fwd_bps = (prices[future_idx] - prices) / prices * 10000
    
    trades_df["fwd_10s_bps"] = fwd_bps
    trades_df["toxic"] = ((signs == 1) & (fwd_bps > threshold_bps)) | \
                          ((signs == -1) & (fwd_bps < -threshold_bps))
    
    return trades_df


def build_full_features(asset, week_name, week_dates, ob_dir, trades_path, output_dir):
    """Build complete feature matrix for one asset-week."""
    print(f"\n=== {asset} {week_name} ===")
    
    # step 1: run book reconstruction if not already done
    trades_df = pd.read_parquet(trades_path)
    trades_df["ts_ms"] = trades_df["timestamp"].astype("int64") // 10**6
    
    # step 2: load and concatenate book features for this week
    book_frames = []
    trade_frames = []
    
    for date_str in week_dates:
        book_path = os.path.join(output_dir, f"{asset}_{date_str}_book_features.parquet")
        if not os.path.exists(book_path):
            print(f"  MISSING book features: {book_path}, skipping day")
            continue
        
        book_day = pd.read_parquet(book_path)
        
        # filter trades to this day
        day_start = pd.Timestamp(date_str).value // 10**6
        day_end = pd.Timestamp(pd.Timestamp(date_str) + pd.Timedelta(days=1)).value // 10**6
        day_trades = trades_df[(trades_df["ts_ms"] >= day_start) & 
                               (trades_df["ts_ms"] < day_end)].copy()
        
        # align: keep only trades after first book snapshot
        if len(book_day) > 0 and len(day_trades) > 0:
            day_trades = day_trades[day_trades["ts_ms"] >= book_day["timestamp"].iloc[0]].copy()
        
        if len(day_trades) != len(book_day):
            print(f"  WARNING {date_str}: trades={len(day_trades)} vs book={len(book_day)}, skipping")
            continue
        
        # paste book features onto trades
        for col in book_day.columns:
            if col != "timestamp":
                day_trades[col] = book_day[col].values
        
        trade_frames.append(day_trades)
        print(f"  {date_str}: {len(day_trades)} rows merged")
    
    if not trade_frames:
        print("  No data — skipping")
        return
    
    combined = pd.concat(trade_frames, ignore_index=True)
    
    # step 3: add trade-derived features (on full week for correct rolling windows)
    combined = add_trade_features(combined)
    
    # step 4: add toxicity labels
    combined = add_toxicity_label(combined)

    # step 4.5: add VPIN
    combined = add_vpin_feature(combined, asset, week_name)
    
    # step 5: save
    out_path = os.path.join(output_dir, f"{asset}_{week_name}_full_features.parquet")
    combined.to_parquet(out_path, index=False)
    
    toxic_rate = combined["toxic"].mean()
    print(f"  DONE: {len(combined)} rows, toxic rate={toxic_rate:.4f}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("running the correct")
    ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    WEEKS = {
        "week1": ["2024-09-09", "2024-09-10", "2024-09-11", "2024-09-12",
                  "2024-09-13", "2024-09-14", "2024-09-15"],
        "week2": ["2024-10-28", "2024-10-29", "2024-10-30", "2024-10-31",
                  "2024-11-01", "2024-11-02", "2024-11-03"],
        "week3": ["2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27",
                  "2025-02-28", "2025-03-01", "2025-03-02"],
    }
    
    for asset in ASSETS:
        for week_name, dates in WEEKS.items():
            trades_path = f"data/processed/{asset}/{week_name}.parquet"
            if not os.path.exists(trades_path):
                print(f"\nSKIPPING {asset} {week_name} — no trade parquet")
                continue
            build_full_features(
                asset, week_name, dates,
                f"data/raw/orderbook/{asset}",
                trades_path,
                "data/processed/features"
            )

