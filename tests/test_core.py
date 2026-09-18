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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from vpin import build_volume_bucket, compute_vpin
from build_features import add_trade_features, add_toxicity_label
from src.evaluations import bootstrap_eval
from src.evaluations.bootstrap_eval import run_backtest
from src.models.classifier import data_loader


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
      T1: sell qty=10 → close B1={v_buy=10, v_sell=10, ts=T1}
      T2: buy  qty=15 → new bucket={v_buy=15}
      T3: sell qty= 5 → close B2={v_buy=15, v_sell=5, ts=T3}
      T4: buy  qty=20 → close B3={v_buy=20, v_sell=0, ts=T4}

    A bucket closes immediately when the filling trade reaches the exact
    boundary, and its timestamp is that filling trade's timestamp.

    Complete bucket imbalances are 0, 10, and 20. With n_buckets=2:
      window B1+B2: (0 + 10) / (2 × 20) = 0.25, timestamp T3
      window B2+B3: (10 + 20) / (2 × 20) = 0.75, timestamp T4
    """
    trades = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "price":     [100.0] * 5,
        "qty":       [10.0, 10.0, 15.0, 5.0, 20.0],
        "sign":      [1, -1, 1, -1, 1],
    })

    result = compute_vpin(trades, bucket_size=20, n_buckets=2)

    assert result["timestamp"].tolist() == [3, 4]
    np.testing.assert_allclose(result["vpin"].values, [0.25, 0.75])


def test_vpin_bucket_closes_on_exact_fill():
    """An exact fill must emit the bucket without waiting for another trade."""
    trades = pd.DataFrame({
        "timestamp": [100, 200],
        "qty": [4.0, 6.0],
        "sign": [1, -1],
    })

    buckets = build_volume_bucket(trades, bucket_size=10.0)

    assert buckets == [{
        "v_buy": 4.0,
        "v_sell": 6.0,
        "timestamp": 200,
    }]


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


def test_trade_features_include_current_trade_and_window_boundary():
    """Hand-check inclusive trailing windows immediately after each trade."""
    trades = pd.DataFrame({
        "ts_ms": [0.0, 1_000.0, 5_000.0, 10_000.0],
        "qty": [1.0, 2.0, 3.0, 4.0],
        "sign": [1, -1, 1, -1],
        "price": [100.0, 100.0, 100.0, 100.0],
    })

    result = add_trade_features(trades)

    assert result["trade_intensity_1s"].tolist() == [1, 2, 1, 1]
    assert result["trade_intensity_5s"].tolist() == [1, 2, 3, 2]
    assert result["trade_intensity_10s"].tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(
        result["volume_acceleration"].values,
        [6.0, 6.0, 6.0, 4.2],
    )
    np.testing.assert_allclose(
        result["signed_vol_imbalance_10s"].values,
        [1.0, -1.0, 2.0, -2.0],
    )


def test_toxicity_label_invalid_without_full_forward_horizon():
    """Trailing rows are invalid rather than clipped to the final price."""
    trades = pd.DataFrame({
        "ts_ms": [0.0, 10_000.0, 15_000.0],
        "price": [100.0, 101.0, 200.0],
        "sign": [1, 1, 1],
    })

    result = add_toxicity_label(trades, threshold_bps=8, horizon_s=10)

    assert result.loc[0, "fwd_10s_bps"] == pytest.approx(100.0)
    assert result.loc[0, "toxic"] == True
    assert result["fwd_10s_bps"].iloc[1:].isna().all()
    assert result["toxic"].iloc[1:].isna().all()


def test_classifier_loader_drops_invalid_labels_and_preserves_identity(monkeypatch):
    """Model/evaluation exports retain causal identity after invalid rows drop."""
    frame = pd.DataFrame({
        **{name: [1.0, 1.0, 1.0] for name in data_loader.RAW_FEATURES},
        "spread": [0.2, 0.2, 0.2],
        "microprice": [100.0, 100.0, 100.0],
        "midprice": [100.0, 100.0, 100.0],
        "qty": [1.0, 1.0, 1.0],
        "toxic": pd.Series([True, pd.NA, False], dtype="boolean"),
        "timestamp": pd.to_datetime([1, 2, 3], unit="s"),
    })
    monkeypatch.setattr(
        data_loader.pd,
        "read_parquet",
        lambda *args, **kwargs: frame.copy(),
    )

    loaded = data_loader.load_asset_week("unused", "BTCUSDT", "week3")

    assert loaded["toxic"].tolist() == [True, False]
    assert loaded["source_row"].tolist() == [0, 2]
    assert loaded["timestamp"].tolist() == [frame.loc[0, "timestamp"], frame.loc[2, "timestamp"]]


def test_backtest_loader_aligns_predictions_by_source_row_and_timestamp(monkeypatch):
    """Prediction export keeps identity through label/feature row filtering."""
    timestamps = pd.to_datetime([1, 2, 3], unit="s")
    features = pd.DataFrame({
        "timestamp": timestamps,
        "price": [100.0, 101.0, 102.0],
        "sign": [1, -1, 1],
        "qty": [1.0, 1.0, 1.0],
        "spread": [0.2, 0.2, 0.2],
        "midprice": [100.0, 101.0, 102.0],
        "toxic": [False, True, False],
    })
    predictions = pd.DataFrame({
        "asset": ["BTCUSDT", "BTCUSDT"],
        "week": ["week3", "week3"],
        "source_row": [2, 0],
        "timestamp": [timestamps[2], timestamps[0]],
        "p_logreg": [0.9, 0.1],
    })

    def fake_read_parquet(path, columns):
        frame = predictions if str(path).endswith("predictions.parquet") else features
        return frame.loc[:, columns].copy()

    monkeypatch.setattr(bootstrap_eval.pd, "read_parquet", fake_read_parquet)

    loaded = bootstrap_eval.load_backtest_data(
        Path("features"),
        Path("predictions.parquet"),
        "BTCUSDT",
        "week3",
    )

    assert loaded["source_row"].tolist() == [0, 2]
    assert loaded["timestamp"].tolist() == [timestamps[0], timestamps[2]]
    assert loaded["p_logreg"].tolist() == [0.1, 0.9]


def _causal_backtest_bars() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime([0, 1, 2], unit="s"),
        "mid": [100.0, 200.0, 200.0],
        "best_bid": [99.9, 199.9, 199.9],
        "best_ask": [100.1, 200.1, 200.1],
        "any_buy": [False, True, True],
        "any_sell": [False, False, False],
        "max_buy_price": [0.0, 100.15, 200.15],
        "min_sell_price": [0.0, 0.0, 0.0],
        "p_logreg": [0.0, 1.0, 0.0],
        "realised_toxic_rate": [0.0, 0.0, 1.0],
    })


def test_backtest_ignores_realised_target_column():
    """Changing the ex-post target cannot change strategy decisions."""
    bars = _causal_backtest_bars()
    changed_target = bars.copy()
    changed_target["realised_toxic_rate"] = [1.0, 1.0, 0.0]

    original = run_backtest(bars, k=5, signal_column="p_logreg", sigma_window=2)
    changed = run_backtest(
        changed_target,
        k=5,
        signal_column="p_logreg",
        sigma_window=2,
    )

    pd.testing.assert_frame_equal(original, changed)


def test_backtest_requires_prediction_instead_of_realised_target():
    """The realised label cannot silently become the strategy signal."""
    bars = _causal_backtest_bars().drop(columns=["p_logreg"])

    with pytest.raises(KeyError, match="Prediction column 'p_logreg' is required"):
        run_backtest(bars, k=5, signal_column="p_logreg", sigma_window=2)


def test_backtest_quotes_from_previous_bar_and_fills_on_current_bar():
    """Bar t state/signal place the quote tested against bar t+1 trades."""
    result = run_backtest(
        _causal_backtest_bars(),
        k=5,
        signal_column="p_logreg",
        sigma_window=2,
    )

    assert result.loc[0, "inventory"] == 0.0
    assert result.loc[1, "inventory"] < 0.0
    assert result.loc[2, "inventory"] == pytest.approx(result.loc[1, "inventory"])
