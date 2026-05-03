import json
import os
import time
import zipfile
from pathlib import Path

import pandas as pd


def apply_update(bids: dict, asks: dict, message: dict) -> None:
    """Apply a single order-book snapshot or delta message to the live book.

    Args:
        bids: Mutable dict mapping price (float) to size (float) for the
            bid side. Modified in-place.
        asks: Mutable dict mapping price (float) to size (float) for the
            ask side. Modified in-place.
        message: Parsed JSON message with keys 'type' ('snapshot' or 'delta')
            and 'data' containing 'b' (bid updates) and 'a' (ask updates),
            each a list of [price, size] pairs.

    Returns:
        None
    """
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


def compute_book_features(bids: dict, asks: dict) -> dict | None:
    """Compute microstructure features from the current order book state.

    Args:
        bids: Dict mapping price (float) to size (float) for bid levels.
        asks: Dict mapping price (float) to size (float) for ask levels.

    Returns:
        Dict of feature values (spread, microprice, midprice, depth_imbalance_N
        for N in [1,5,10,25], bid_pressure, ask_pressure, pressure_imbalance),
        or None if either side of the book is empty.
    """
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


def reconstruct_and_extract_from_state(
    ob_zip_path: str | Path,
    trades_df: pd.DataFrame,
    bids: dict,
    asks: dict,
) -> pd.DataFrame:
    """Replay order-book messages and extract book features at each trade timestamp.

    Iterates through a zipped NDJSON order-book file in time order. For each
    trade whose timestamp precedes the next book message, the current book
    state is snapshotted and a feature row is appended.

    Args:
        ob_zip_path: Path to the .zip file containing a single NDJSON order-
            book stream (one JSON object per line).
        trades_df: DataFrame of trades for the day with a timestamp column
            (nanosecond epoch integers convertible via astype int64 // 1e6).
        bids: Mutable bid-side book state dict, shared across calls to allow
            the book to persist between days.
        asks: Mutable ask-side book state dict, shared across calls.

    Returns:
        DataFrame with one row per trade and columns for each book feature
        plus timestamp (millisecond epoch int).
    """
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


def process_week(
    asset: str,
    week: str,
    week_dates: list[str],
    ob_dir: str,
    trades_path: str,
    output_dir: str,
) -> None:
    """Process all days in a week, writing per-day book-feature parquets.

    Maintains a persistent bid/ask book across days so that the end-of-day
    state carries over to the next day's opening messages.

    Args:
        asset: Asset symbol, e.g. 'BTCUSDT'.
        week: Week label, e.g. 'week1'.
        week_dates: Ordered list of date strings ('YYYY-MM-DD') in the week.
        ob_dir: Directory containing per-day order-book zip files named
            '<date>_<asset>_ob500.data.zip'.
        trades_path: Path to the weekly trade parquet for this asset.
        output_dir: Directory where per-day book feature parquets are written.

    Returns:
        None
    """
    trades_df = pd.read_parquet(trades_path)
    trade_ts_ms = trades_df["timestamp"].astype("int64") // 10**6

    os.makedirs(output_dir, exist_ok=True)
    bids: dict = {}
    asks: dict = {}

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
    _data_root = Path(__file__).parent.parent.parent / 'data'

    for asset in ASSETS:
        for week_name, dates in WEEKS.items():
            trades_path = str(_data_root / 'processed' / asset / f'{week_name}.parquet')
            if not os.path.exists(trades_path):
                print(f"\nSKIPPING {asset} {week_name} — no trade parquet")
                continue
            ob_dir = str(_data_root / 'raw' / 'orderbook' / asset)
            print(f"\n=== {asset} {week_name} ===")
            process_week(asset, week_name, dates, ob_dir, trades_path,
                        str(_data_root / 'processed' / 'features'))
