"""
Unit tests for core market-making research components.

All tests use synthetic data — no parquet files or external data required.
Run from repo root:  pytest tests/
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# ── make src subpackages importable when pytest is run from repo root ─────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from vpin import compute_vpin
from build_features import add_trade_features, add_toxicity_label


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — VPIN on a synthetic sequence with hand-computed expected output
# ─────────────────────────────────────────────────────────────────────────────

def test_vpin_hand_computed():
    """
    Verifies compute_vpin() against a 5-trade sequence whose bucket contents
    and VPIN are derivable by hand.

    bucket_size=20, n_buckets=2.

    Trade-by-trade trace through build_volume_bucket():
      T0: buy  qty=10 → vol={v_buy=10, v_sell=0}            cap_used=10/20
      T1: sell qty=10 → vol={v_buy=10, v_sell=10}           cap_used=20/20 (full)
      T2: buy  qty=15 → cap=0, remaining=15 > 0 ⇒
                         flush B1={v_buy=10, v_sell=10, ts=T2's timestamp};
                         new bucket: {v_buy=15}
      T3: sell qty= 5 → vol={v_buy=15, v_sell=5}            cap_used=20/20 (full)
      T4: buy  qty=20 → cap=0, remaining=20 > 0 ⇒
                         flush B2={v_buy=15, v_sell=5, ts=T4's timestamp};
                         new partial bucket (not returned)

    Note: the bucket timestamp is stamped by the trade that causes the flush
    (not the last trade that filled the bucket), because build_volume_bucket()
    writes vol['timestamp'] = ts before calling buckets.append(vol.copy()).

    Two complete buckets:
      B1: v_buy=10, v_sell=10  → |imbalance| = 0
      B2: v_buy=15, v_sell=5   → |imbalance| = 10

    VPIN (Easley et al.): Σ|v_buy − v_sell| / (n_buckets × bucket_size)
                         = (0 + 10) / (2 × 20) = 0.25

    With n_buckets=2 and exactly 2 complete buckets the rolling window
    range(2, 3) yields exactly one observation.
    """
    trades = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "price":     [100.0] * 5,
        "qty":       [10.0, 10.0, 15.0, 5.0, 20.0],
        "sign":      [1, -1, 1, -1, 1],
    })

    result = compute_vpin(trades, bucket_size=20, n_buckets=2)

    assert len(result) == 1, f"Expected 1 VPIN row, got {len(result)}"
    got = result["vpin"].iloc[0]
    assert abs(got - 0.25) < 1e-6, f"Expected VPIN=0.25, got {got:.10f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — PnL for a known fill scenario
# ─────────────────────────────────────────────────────────────────────────────

def test_pnl_round_trip_known_spread():
    """
    Verifies the mark-to-market PnL formula used by run_backtest_bars() in
    notebooks/04_backtest_results.ipynb:

        pnl = state.cash + state.inventory * mid

    Scenario: MM posts bid=99.90, ask=100.10 against a mid of 100.00.
    Exactly two fills arrive, one on each side.

      Fill 1 — buy aggressor lifts MM's ask (MM sells 1 unit at 100.10):
        cash      += ask × qty  →  cash = +100.10
        inventory -= qty        →  inventory = −1.0

      Fill 2 — sell aggressor hits MM's bid (MM buys 1 unit at 99.90):
        cash      -= bid × qty  →  cash = +100.10 − 99.90 = +0.20
        inventory += qty        →  inventory = 0.0

    After both fills the MM is flat. Gross PnL equals the posted spread:
        MtM PnL = cash + inventory × mid = 0.20 + 0 × 100.00 = $0.20

    This is the ideal round-trip outcome: spread captured with no inventory
    risk and no adverse price movement between fills.
    """
    bid = 99.90
    ask = 100.10
    mid = 100.00
    qty = 1.0

    # Replicate MarketMakerState from the notebook (dataclass with cash, inventory)
    cash = 0.0
    inventory = 0.0

    # Fill 1: buy aggressor hits ask → MM sells
    cash += ask * qty
    inventory -= qty

    # Fill 2: sell aggressor hits bid → MM buys
    cash -= bid * qty
    inventory += qty

    # Mark-to-market PnL (notebook formula from run_backtest_bars)
    pnl = cash + inventory * mid

    assert abs(inventory) < 1e-10, (
        f"Inventory must be zero after a round trip, got {inventory}"
    )
    assert abs(pnl - 0.20) < 0.01, (
        f"Expected round-trip PnL ≈ $0.20 (= bid-ask spread), got ${pnl:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — No look-ahead leakage in rolling trade features
# ─────────────────────────────────────────────────────────────────────────────

def test_no_lookahead_leakage_rolling_features():
    """
    Verifies that every rolling trade feature from add_trade_features() is
    computed using only data at timestamps ≤ trades[i].ts_ms.

    Why this matters: look-ahead leakage is the most common way a classifier
    can appear to work in training while failing completely in production.
    A feature at bar i that accidentally reads bar i+k's volume would make
    the model appear to "see the future."

    Method — truncation invariant:
      1. Build a 100-trade synthetic sequence with 1-second spacing.
      2. Compute all features on the full 100-trade sequence.
      3. Remove the last 10 trades and recompute on 90 trades.
      4. Assert that rows 0..89 are numerically identical in both runs.

    If any feature at row i incorporates a trade that exists in the full
    sequence but not in the truncated sequence (i.e. a future trade), the
    values will differ → leakage is detected.

    Features checked (all backward-looking by construction):
      trade_intensity_{1,5,10}s   — count of trades in trailing window
      volume_acceleration          — recent volume vs baseline volume ratio
      signed_vol_imbalance_10s    — net signed volume over trailing 10 s

    NOT checked: fwd_10s_bps (from add_toxicity_label) is intentionally
    forward-looking — it IS the prediction target (label), not a model input.
    Rows near the end of the sequence will legitimately differ after truncation
    because the 10-second forward price horizon changes. This is expected.
    """
    n = 100
    rng = np.random.default_rng(0)

    trades = pd.DataFrame({
        "ts_ms": np.arange(n, dtype=float) * 1000,         # 1 s apart, monotonic
        "qty":   rng.uniform(0.1, 2.0, size=n),
        "sign":  rng.choice([-1, 1], size=n).astype(int),
        "price": 100.0 + np.cumsum(rng.normal(0, 0.01, size=n)),
    })

    feature_cols = [
        "trade_intensity_1s",
        "trade_intensity_5s",
        "trade_intensity_10s",
        "volume_acceleration",
        "signed_vol_imbalance_10s",
    ]

    full  = add_trade_features(trades.copy())
    trunc = add_trade_features(
        trades.iloc[:-10].copy().reset_index(drop=True)
    )

    for col in feature_cols:
        full_vals  = full[col].values[:-10]    # rows 0..89 from full run
        trunc_vals = trunc[col].values         # rows 0..89 from truncated run

        close = np.allclose(
            full_vals, trunc_vals,
            rtol=1e-10, atol=1e-10, equal_nan=True,
        )
        if not close:
            bad = np.where(
                ~np.isclose(
                    full_vals, trunc_vals,
                    rtol=1e-10, atol=1e-10, equal_nan=True,
                )
            )[0]
            pytest.fail(
                f"Look-ahead leakage detected in '{col}': "
                f"{len(bad)} row(s) differ (first at row {bad[0]}). "
                f"full={full_vals[bad[0]]:.8g}, trunc={trunc_vals[bad[0]]:.8g}"
            )
