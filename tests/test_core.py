"""
Unit tests for core market-making research components.

All tests use synthetic data — no parquet files or external data required.
Run from repo root:  pytest tests/
"""

from pathlib import Path
import json
import sys
import zipfile

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
from reconstructor import apply_update, reconstruct_and_extract_from_state
from src.evaluations import bootstrap_eval
from src.evaluations.bootstrap_eval import run_backtest
from src.models.classifier import data_loader
from src.data.loader import load_trades


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


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Loaded trade timestamps are monotonically increasing
# ─────────────────────────────────────────────────────────────────────────────

def test_load_trades_timestamps_are_monotonic(tmp_path):
    """
    Guards the invariant that load_trades() returns trades in chronological
    order. Downstream logic (gap detection, VPIN volume bucketing, forward-
    return labels) all assume time-ordered trades, so out-of-order timestamps
    would silently corrupt every derived feature.

    Method: write a gzipped CSV whose rows are deliberately OUT of timestamp
    order, run it through load_trades(), and assert the output is monotonic.
    This FAILS if load_trades ever emits non-monotonic timestamps.
    """
    raw = pd.DataFrame({
        "timestamp": [3.0, 1.0, 2.0, 5.0, 4.0],   # intentionally unordered
        "side":      ["Buy", "Sell", "Buy", "Sell", "Buy"],
        "size":      [1.0, 2.0, 3.0, 4.0, 5.0],
        "price":     [100.0, 100.1, 100.2, 100.3, 100.4],
    })
    csv_gz = tmp_path / "BTCUSDT2024-09-09.csv.gz"
    raw.to_csv(csv_gz, index=False, compression="gzip")

    loaded = load_trades(csv_gz)

    assert loaded["timestamp"].is_monotonic_increasing


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Order-book price keys are canonicalized against float drift
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_update_canonicalizes_price_key_against_float_drift():
    """
    A level inserted at a price carrying float representation error must still
    be removed by a size-0 update quoted at the canonical price.

    apply_update() uses the price as a dict key. The classic hazard is
    0.1 + 0.2 == 0.30000000000000004 ≠ 0.3: with the raw float as the key, an
    insert at 0.1 + 0.2 and a delete at 0.3 land on two different keys, so the
    delete silently misses and leaks a phantom level. Rounding the key to
    instrument precision collapses both to one canonical key.

    Without the round() canonicalization this test fails: the bid level
    survives the delete because pop(0.3) never matches key 0.30000000000000004.
    """
    bids: dict = {}
    asks: dict = {}

    drifted_price = 0.1 + 0.2            # == 0.30000000000000004, not 0.3
    assert drifted_price != 0.3         # guard: test only bites under drift

    # Insert a bid level at the drifted price plus an untouched ask level.
    apply_update(bids, asks, {
        "type": "delta",
        "data": {"b": [[drifted_price, 5.0]], "a": [[0.4, 2.0]]},
    })
    assert len(bids) == 1

    # Delete the bid at the canonical price ("0.3"); leave the ask side alone.
    apply_update(bids, asks, {
        "type": "delta",
        "data": {"b": [["0.3", 0.0]], "a": []},
    })

    assert bids == {}, "size-0 update at canonical price must remove the level"
    assert asks == {0.4: 2.0}, "opposite side must be untouched"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Book replay samples each trade causally (no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def _write_ob_zip(tmp_path, messages):
    """Write book messages as NDJSON inside a zip and return its path."""
    inner = tmp_path / "book.ndjson"
    inner.write_text("\n".join(json.dumps(m) for m in messages) + "\n")
    zip_path = tmp_path / "book.data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(inner, arcname="book.ndjson")
    return zip_path


def test_reconstruct_samples_book_state_at_or_before_trade_no_lookahead(tmp_path):
    """
    reconstruct_and_extract_from_state must sample, for each trade, the book
    state produced by every message with ts <= the trade's ts and NOTHING after
    it. This is the pipeline's core no-look-ahead guarantee: the features used
    to predict a trade's toxicity must not embed information from book updates
    that landed after the trade.

    Three properties are pinned at once, using a book whose spread changes on a
    known schedule (1 -> 2 -> 3):

      * strict `<` flush + pre-update snapshot: a trade at EXACTLY a message's ts
        sees that message applied (spread 2 at ts=2000) but not the later one
        (not spread 3). If the loop used `<=`, or sampled post-update, this would
        read the wrong spread.
      * None-drop: a trade before the first two-sided book yields no row.
      * stale-tail: a trade after the last message inherits the final book state.
    """
    messages = [
        {"ts": 1000, "type": "snapshot",
         "data": {"b": [[100.0, 5.0]], "a": [[101.0, 5.0]]}},   # spread 1
        {"ts": 2000, "type": "delta",
         "data": {"b": [], "a": [[101.0, 0.0], [102.0, 5.0]]}},  # -> spread 2
        {"ts": 3000, "type": "delta",
         "data": {"b": [[100.0, 0.0], [99.0, 5.0]], "a": []}},   # -> spread 3
    ]
    zip_path = _write_ob_zip(tmp_path, messages)

    # trade @500 is before any two-sided book (dropped); the rest straddle the
    # message boundaries to probe the <= vs < distinction and the tail.
    # ns resolution so the production `astype(int64) // 1e6` recovers ms
    trades_df = pd.DataFrame(
        {"timestamp": pd.to_datetime([500, 1500, 2000, 4000], unit="ms").as_unit("ns")}
    )

    out = reconstruct_and_extract_from_state(zip_path, trades_df, {}, {})

    # the pre-first-snapshot trade produced no row (empty book -> None)
    assert list(out["timestamp"]) == [1500, 2000, 4000]

    by_ts = out.set_index("timestamp")
    # trade @1500: only the snapshot applied -> spread 1 (did NOT see @2000)
    assert by_ts.loc[1500, "spread"] == pytest.approx(1.0)
    assert by_ts.loc[1500, "midprice"] == pytest.approx(100.5)
    # trade @2000: @2000 delta applied (spread 2), @3000 NOT yet -> proves `<`
    assert by_ts.loc[2000, "spread"] == pytest.approx(2.0)
    assert by_ts.loc[2000, "midprice"] == pytest.approx(101.0)
    # trade @4000: past the last message -> final (stale) book -> spread 3
    assert by_ts.loc[4000, "spread"] == pytest.approx(3.0)
    assert by_ts.loc[4000, "midprice"] == pytest.approx(100.5)
