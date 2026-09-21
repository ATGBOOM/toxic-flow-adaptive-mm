from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
WEEKS = ['WEEK1', 'WEEK2', 'WEEK3']
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'BTCUSDT' / 'week1.parquet'


def build_volume_bucket(df: pd.DataFrame, bucket_size: float) -> list[dict]:
    """Split trades into fixed-volume buckets, splitting trades that span boundaries.

    Args:
        df: DataFrame with columns [timestamp, qty, sign] where sign is +1 or -1.
        bucket_size: Target total volume (sum of qty) per bucket.

    Returns:
        List of dicts, each with keys v_buy (float), v_sell (float), and
        timestamp (the timestamp of the trade that caused the bucket to flush).
    """
    buckets = []
    vol = {"v_buy": 0.0, "v_sell": 0.0, "timestamp": None}

    # extract to numpy arrays — much faster than iterrows
    qtys = df['qty'].to_numpy()
    signs = df['sign'].to_numpy()
    timestamps = df['timestamp'].to_numpy()

    for i in range(len(qtys)):
        remaining_qty = qtys[i]
        ts = timestamps[i]
        sign = signs[i]

        while remaining_qty > 0:
            capacity = bucket_size - (vol['v_buy'] + vol['v_sell'])

            if remaining_qty < capacity:
                if sign == 1:
                    vol['v_buy'] += remaining_qty
                else:
                    vol['v_sell'] += remaining_qty
                vol['timestamp'] = ts
                remaining_qty = 0
            else:
                # A trade that exactly fills the remaining capacity closes the
                # bucket now; do not wait for the next trade to flush it.
                if sign == 1:
                    vol['v_buy'] += capacity
                else:
                    vol['v_sell'] += capacity
                vol['timestamp'] = ts
                buckets.append(vol.copy())
                vol = {'v_buy': 0.0, 'v_sell': 0.0, 'timestamp': None}
                remaining_qty -= capacity

    return buckets


def compute_rolling_vpins(
    volume_buckets: list[dict], n_buckets: int, bucket_size: float
) -> pd.DataFrame:
    """Compute rolling VPIN over a sliding window of completed volume buckets.

    Args:
        volume_buckets: List of bucket dicts from build_volume_bucket, each
            containing v_buy, v_sell, and timestamp.
        n_buckets: Number of buckets in the rolling window.
        bucket_size: Volume per bucket; used as the denominator normaliser.

    Returns:
        DataFrame with columns [timestamp, vpin]. One row per complete window,
        timestamped at the last bucket in that window.
    """
    if len(volume_buckets) < n_buckets:
        return pd.DataFrame(columns=["timestamp", "vpin"])

    imbalances = np.array([abs(b["v_buy"] - b["v_sell"]) for b in volume_buckets])
    timestamps = np.array([b["timestamp"] for b in volume_buckets])

    # Sliding-window sum via cumsum: O(n) instead of O(n * n_buckets) list slicing.
    cs = np.concatenate([[0.0], np.cumsum(imbalances)])
    window_sums = cs[n_buckets:] - cs[:-n_buckets]
    vpin_values = window_sums / (n_buckets * bucket_size)

    # Each VPIN value is timestamped at the last bucket in its window.
    return pd.DataFrame({
        "timestamp": timestamps[n_buckets - 1:],
        "vpin": vpin_values,
    })


def compute_vpin(df: pd.DataFrame, bucket_size: float, n_buckets: int = 50) -> pd.DataFrame:
    """Compute rolling VPIN (Volume-Synchronised Probability of Informed Trading).

    Implements the Easley et al. (2012) VPIN estimator: trades are aggregated
    into equal-volume buckets and the rolling average absolute buy-sell
    imbalance is computed as a fraction of total bucket volume.

    Args:
        df: DataFrame with columns [timestamp, price, qty, sign] where sign
            is +1 for buys and -1 for sells.
        bucket_size: Total volume (sum of qty) per bucket. Typically set to
            1/50 of daily volume following Easley et al.
        n_buckets: Number of buckets in the rolling VPIN window. Defaults to 50.

    Returns:
        DataFrame with columns [timestamp, vpin]. Timestamps correspond to the
        last trade in each rolling window's final bucket.
    """
    volume_buckets = build_volume_bucket(df, bucket_size)

    # formula is 1/n * sum of (V_B - V_S)/V
    rolling_vpins = compute_rolling_vpins(volume_buckets, n_buckets, bucket_size)

    return rolling_vpins

