"""
generate_mock_data.py — interview-demo fixture generator.

Not part of the pipeline. Writes small, synthetic raw trade and order-book
files in the exact format the real pipeline expects, so the actual entry
points (pipeline.py, reconstructor.py, build_features.py, data_loader.py,
save_predictions.py, bootstrap_eval.py) can all be run live, unmodified,
without needing the real Bybit download.

Scope: all three assets (BTCUSDT, ETHUSDT, SOLUSDT), one representative day
per week (inside each WEEK_RANGES entry in src/data/pipeline.py so
date_to_week() buckets it correctly), 30 minutes of activity per day — nine
asset-week fixtures total. Missing-day handling in the real pipeline means
every other day within a week is simply skipped, gracefully, with no code
changes needed; this covers the one day per week that pipeline.py,
save_predictions.py, and bootstrap_eval.py all actually need present.

Caveat: with one day per asset-week, block_bootstrap_improvement() sees
exactly one daily block (n_blocks=1) per asset for week3. Resampling one
block with replacement always reproduces the same block, so the bootstrap
CI will collapse to a point (zero width) rather than show real spread —
that's an expected consequence of a single-day fixture, not a bug in the
bootstrap code, which genuinely needs multiple days to produce a
meaningful interval.

Usage: python scripts/generate_mock_data.py
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
WINDOW_S = 1800   # 30 minutes of synthetic activity per day
N_TRADES = 800
N_LEVELS = 30     # book levels per side
SEED_BASE = 42

# One representative date inside each real WEEK_RANGES entry
# (src/data/pipeline.py), and a plausible base price per asset.
FIXTURES = [
    ("BTCUSDT", "week1", "2024-09-09", 60_000),
    ("BTCUSDT", "week2", "2024-10-28", 70_000),
    ("BTCUSDT", "week3", "2025-02-24", 95_000),
    ("ETHUSDT", "week1", "2024-09-09", 2_400),
    ("ETHUSDT", "week2", "2024-10-28", 2_600),
    ("ETHUSDT", "week3", "2025-02-24", 2_700),
    ("SOLUSDT", "week1", "2024-09-09", 135),
    ("SOLUSDT", "week2", "2024-10-28", 165),
    ("SOLUSDT", "week3", "2025-02-24", 180),
]


def build_price_path(rng: np.random.Generator, base_price: float) -> np.ndarray:
    """One value per second over the window: a random walk with a few
    injected shock bursts so the 8bps/10s toxicity label actually fires."""
    steps = rng.normal(0, 0.0002, size=WINDOW_S)  # ~2bps/s baseline vol
    price = base_price * np.exp(np.cumsum(steps))

    shocks = [(200, 220, +0.006), (500, 525, -0.007),
              (900, 915, +0.008), (1300, 1330, -0.006)]
    for start, end, total_move in shocks:
        ramp = np.linspace(0, total_move, end - start)
        price[start:end] *= np.exp(ramp)
        price[end:] *= np.exp(total_move)
    return price


def write_trades(
    rng: np.random.Generator, price_path: np.ndarray, asset: str, date: str,
) -> pd.DataFrame:
    day_start = pd.Timestamp(date, tz=None)
    gaps = rng.exponential(WINDOW_S / N_TRADES, size=N_TRADES)
    offsets = np.cumsum(gaps)
    offsets = offsets[offsets < WINDOW_S]

    idx = offsets.astype(int)
    base_price = price_path[idx]
    noise = rng.normal(0, 0.0001, size=len(idx))
    trade_price = np.round(base_price * (1 + noise), 4)

    # size: mostly small, occasional larger trade (exercises VPIN
    # bucket-splitting)
    size = rng.lognormal(mean=np.log(0.03), sigma=0.8, size=len(idx))
    size = np.where(rng.random(len(idx)) < 0.05, size * 5, size)
    size = np.round(size, 4)

    side = np.where(rng.random(len(idx)) < 0.5, "Buy", "Sell")

    ts_unix_s = (day_start + pd.to_timedelta(offsets[:len(idx)], unit="s"))
    ts_unix_s = ts_unix_s.view("int64") / 1e9  # Unix seconds, float

    df = pd.DataFrame({
        "timestamp": ts_unix_s,
        "side": side,
        "size": size,
        "price": trade_price,
    }).sort_values("timestamp").reset_index(drop=True)

    out_dir = ROOT / "data" / "raw" / "trades" / asset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{asset}{date}.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")
    print(f"  wrote {out_path} ({len(df)} trades)")
    return df


def write_orderbook(
    rng: np.random.Generator, price_path: np.ndarray, asset: str, date: str,
) -> None:
    """Emits snapshot/delta messages while tracking local bid/ask state, the
    same way apply_update() would, so deltas explicitly retire any level
    that drifts out of the visible window instead of leaving it dangling —
    a dangling stale level is exactly what would make the replayed book
    cross (best bid >= best ask), which isn't a real pipeline bug, just an
    unrealistic fixture, and would be a confusing thing to hit live."""
    day_start = pd.Timestamp(date, tz=None)
    tick = max(round(price_path[0] * 0.00005, 4), 0.0001)  # ~0.5bps tick

    lines = []
    local_bids: dict = {}
    local_asks: dict = {}

    def new_size() -> str:
        return str(round(float(rng.lognormal(mean=np.log(1.0), sigma=0.6)), 4))

    def snapshot_levels(center: float, side_sign: int, book: dict) -> list:
        book.clear()
        out = []
        for i in range(N_LEVELS):
            p = round(center + side_sign * (i + 1) * tick, 4)
            sz = new_size()
            book[p] = float(sz)
            out.append([str(p), sz])
        return out

    t = 0
    while t < WINDOW_S:
        ts_ms = int((day_start + pd.Timedelta(seconds=t)).value // 10**6)
        center = float(price_path[min(t, WINDOW_S - 1)])
        band = N_LEVELS * tick

        if t % 300 == 0:
            msg = {
                "type": "snapshot",
                "ts": ts_ms,
                "data": {
                    "b": snapshot_levels(center - tick, -1, local_bids),
                    "a": snapshot_levels(center + tick, +1, local_asks),
                },
            }
        else:
            updates_b, updates_a = [], []

            # retire any tracked level that has drifted outside the window
            # around the current center — this is what a real delta stream
            # does implicitly as resting orders get cancelled/filled away
            # from the touch; without it, stale far levels never leave the
            # book and can cross a newly-inserted near-touch level.
            for p in [p for p in local_bids if abs(p - center) > band]:
                updates_b.append([str(p), "0"])
                del local_bids[p]
            for p in [p for p in local_asks if abs(p - center) > band]:
                updates_a.append([str(p), "0"])
                del local_asks[p]

            # never let a delete draw empty a side outright — a real book
            # never goes one-sided, and compute_book_features() returns
            # None (dropping the trade) if either side is empty
            for _ in range(rng.integers(1, 3)):
                p = round(center - tick * rng.integers(1, N_LEVELS), 4)
                if rng.random() < 0.1 and len(local_bids) > 5:
                    updates_b.append([str(p), "0"])
                    local_bids.pop(p, None)
                else:
                    sz = new_size()
                    local_bids[p] = float(sz)
                    updates_b.append([str(p), sz])
            for _ in range(rng.integers(1, 3)):
                p = round(center + tick * rng.integers(1, N_LEVELS), 4)
                if rng.random() < 0.1 and len(local_asks) > 5:
                    updates_a.append([str(p), "0"])
                    local_asks.pop(p, None)
                else:
                    sz = new_size()
                    local_asks[p] = float(sz)
                    updates_a.append([str(p), sz])

            # hard invariant, enforced directly rather than tuned via
            # distance bands: the book must never cross. A stale extremal
            # quote left over from before a price move is exactly what
            # would cross a fresh one on the other side — retire whichever
            # extremal level is furthest from the *current* center (the
            # stale one), refreshing a side if that would empty it.
            while local_bids and local_asks and max(local_bids) >= min(local_asks):
                best_bid, best_ask = max(local_bids), min(local_asks)
                if abs(best_bid - center) >= abs(best_ask - center):
                    updates_b.append([str(best_bid), "0"])
                    del local_bids[best_bid]
                else:
                    updates_a.append([str(best_ask), "0"])
                    del local_asks[best_ask]
            if not local_bids:
                p = round(center - tick, 4)
                sz = new_size()
                local_bids[p] = float(sz)
                updates_b.append([str(p), sz])
            if not local_asks:
                p = round(center + tick, 4)
                sz = new_size()
                local_asks[p] = float(sz)
                updates_a.append([str(p), sz])

            msg = {"type": "delta", "ts": ts_ms,
                   "data": {"b": updates_b, "a": updates_a}}

        lines.append(json.dumps(msg))
        t += 1  # one message every 1s

    out_dir = ROOT / "data" / "raw" / "orderbook" / asset
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{date}_{asset}_ob500.data.zip"
    inner_name = f"{date}_{asset}_ob500.data"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, "\n".join(lines))
    print(f"  wrote {zip_path} ({len(lines)} book messages)")


def generate_fixture(asset: str, week: str, date: str, base_price: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    print(f"\n{asset} {week} ({date})")
    price_path = build_price_path(rng, base_price)
    write_trades(rng, price_path, asset, date)
    write_orderbook(rng, price_path, asset, date)


def main() -> None:
    print(f"Generating mock fixtures: {len(FIXTURES)} asset-weeks, "
          f"{WINDOW_S}s window each\n")
    for i, (asset, week, date, base_price) in enumerate(FIXTURES):
        generate_fixture(asset, week, date, base_price, seed=SEED_BASE + i)

    print("\nDone. Next: run the real pipeline entry points in order —")
    print("  cd src/data && python pipeline.py && cd ../..")
    print("  python src/features/reconstructor.py")
    print("  python src/features/build_features.py")
    print("  python src/evaluations/save_predictions.py")
    print("  python src/evaluations/bootstrap_eval.py")


if __name__ == "__main__":
    main()
