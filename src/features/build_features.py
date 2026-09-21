import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from vpin import compute_vpin


def add_trade_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Add causal trailing-window trade features to a trades DataFrame.

    Features are computed immediately after observing the current trade. Each
    trailing window is inclusive at both ends, [timestamp - window, timestamp],
    so the current trade and trades exactly on the lower boundary are included.
    No future trade is used.

    Args:
        trades_df: DataFrame with columns [ts_ms, qty, sign, price] where
            ts_ms is milliseconds since epoch, qty is trade size, and sign
            is +1 (buy) or -1 (sell). Must be sorted by ts_ms ascending.

    Returns:
        The same DataFrame with additional columns: trade_intensity_1s,
        trade_intensity_5s, trade_intensity_10s, volume_acceleration, and
        signed_vol_imbalance_10s.
    """
    ts = trades_df["ts_ms"].values
    qty = trades_df["qty"].values
    signs = trades_df["sign"].values

    # trade intensity
    for window_s in [1, 5, 10]:
        window_ms = window_s * 1000
        lookback_idx = np.searchsorted(ts, ts - window_ms, side="left")
        trades_df[f"trade_intensity_{window_s}s"] = (
            np.arange(len(ts)) - lookback_idx + 1
        )

    # volume acceleration
    qty_prefix = np.concatenate(([0.0], np.cumsum(qty)))
    row_ends = np.arange(1, len(ts) + 1)
    idx_5s = np.searchsorted(ts, ts - 5000, side="left")
    idx_30s = np.searchsorted(ts, ts - 30000, side="left")
    vol_5s = qty_prefix[row_ends] - qty_prefix[idx_5s]
    vol_30s = qty_prefix[row_ends] - qty_prefix[idx_30s]
    expected_5s = vol_30s * (5 / 30)
    with np.errstate(invalid='ignore'):
        trades_df["volume_acceleration"] = np.where(expected_5s > 0, vol_5s / expected_5s, 1.0)

    # signed volume imbalance
    signed_prefix = np.concatenate(([0.0], np.cumsum(signs * qty)))
    idx_10s = np.searchsorted(ts, ts - 10000, side="left")
    trades_df["signed_vol_imbalance_10s"] = (
        signed_prefix[row_ends] - signed_prefix[idx_10s]
    )

    return trades_df


def add_vpin_feature(trades_df: pd.DataFrame, asset: str, week_name: str) -> pd.DataFrame:
    """Join VPIN values onto a trade-level feature matrix using forward fill.

    Args:
        trades_df: DataFrame with a ts_ms column (milliseconds since epoch).
        asset: Asset symbol, e.g. 'BTCUSDT'.
        week_name: Week label, e.g. 'week1'.

    Returns:
        The same DataFrame with a new vpin column. Trades before the first
        VPIN bucket is complete receive NaN.
    """
    data_root = Path(__file__).parent.parent.parent / 'data' / 'processed'
    trades_raw = pd.read_parquet(data_root / asset / f'{week_name}.parquet')
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


def add_toxicity_label(
    trades_df: pd.DataFrame, threshold_bps: float = 8, horizon_s: int = 10
) -> pd.DataFrame:
    """Add a forward-looking toxicity label based on price movement after each trade.

    Args:
        trades_df: DataFrame with columns [ts_ms, price, sign].
        threshold_bps: Forward price move (in basis points) that defines a
            toxic trade. Defaults to 8 bps.
        horizon_s: Lookahead horizon in seconds. Defaults to 10.

    Returns:
        The same DataFrame with new columns fwd_10s_bps (forward return in
        bps) and toxic (nullable boolean). Rows without the full requested
        forward horizon are invalid and receive missing values in both columns.
    """
    ts = trades_df["ts_ms"].values
    prices = trades_df["price"].values
    signs = trades_df["sign"].values

    if len(trades_df) == 0:
        trades_df["fwd_10s_bps"] = pd.Series(dtype=float)
        trades_df["toxic"] = pd.Series(dtype="boolean")
        return trades_df

    horizon_ms = horizon_s * 1000
    target_ts = ts + horizon_ms
    future_idx = np.searchsorted(ts, target_ts, side="left")
    valid = (target_ts <= ts[-1]) & (future_idx < len(prices))

    fwd_bps = np.full(len(prices), np.nan, dtype=float)
    fwd_bps[valid] = (
        (prices[future_idx[valid]] - prices[valid]) / prices[valid] * 10000
    )

    toxic = pd.Series(pd.NA, index=trades_df.index, dtype="boolean")
    toxic_valid = (
        ((signs[valid] == 1) & (fwd_bps[valid] > threshold_bps))
        | ((signs[valid] == -1) & (fwd_bps[valid] < -threshold_bps))
    )
    toxic.iloc[np.flatnonzero(valid)] = toxic_valid

    trades_df["fwd_10s_bps"] = fwd_bps
    trades_df["toxic"] = toxic

    return trades_df


def build_full_features(
    asset: str,
    week_name: str,
    week_dates: list[str],
    ob_dir: str,
    trades_path: str,
    output_dir: str,
) -> None:
    """Build and save the complete feature matrix for one asset-week.

    Merges book features (pre-computed from order book reconstruction) with
    trade-derived features and toxicity labels, then writes a parquet file.

    Args:
        asset: Asset symbol, e.g. 'BTCUSDT'.
        week_name: Week label, e.g. 'week1'.
        week_dates: List of date strings ('YYYY-MM-DD') in the week.
        ob_dir: Directory containing per-day book feature parquets.
        trades_path: Path to the raw trade parquet for this asset-week.
        output_dir: Directory where the output feature parquet is written.

    Returns:
        None. Writes <asset>_<week_name>_full_features.parquet to output_dir.
    """
    print(f"\n=== {asset} {week_name} ===")

    # step 1: 
    trades_df = pd.read_parquet(trades_path)
    trades_df["ts_ms"] = trades_df["timestamp"].astype("int64") // 10**6

    # step 2: load and concatenate book features for this week
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
