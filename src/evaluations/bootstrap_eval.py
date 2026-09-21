"""
bootstrap_eval.py — Session 15: Statistical Hardening

Computes block-bootstrap confidence intervals on prediction-driven MtM
improvement (adaptive k=5 vs baseline k=0) for each asset × week combination.

This module requires timestamped out-of-sample predictions and causally applies
bar t state/signal to quotes evaluated against bar t+1 trades. Historical PnL
tables produced by the former realised-label path have not been rerun.

Bootstrap unit: daily blocks.
Rationale: consecutive 1-second bars have strongly correlated MtM paths
(cumulative PnL). Resampling individual bars destroys the path structure.
Daily blocks preserve within-day autocorrelation while resampling at
the longest exchangeable unit.

Limitation: ~7 days per week gives ~7 bootstrap blocks — genuinely
underpowered. CIs will be wide. This is the honest result.

Performance: uses vectorised backtest (precompute all bar-level scalars,
tight Python loop only for path-dependent inventory/cash tracking).
~0.5-1s per backtest run on 600k bars vs ~30s with iterrows().

File layout follows presentation order, not alphabetical/historical order:
  1. Load & align predictions      (load_backtest_data)
  2. Build bars                    (build_bars)
  3. Backtest engine — the algorithm (run_backtest)
  4. Bootstrap: resample + CI      (_run_both_strategies, _mtm_improvement,
                                     block_bootstrap_improvement,
                                     compute_bootstrap_ci)
  5. Regime table & reporting      (toxic_fill_rate, avg_spread_ratio,
                                     build_regime_table)
  6. Main
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import NamedTuple
import warnings
warnings.filterwarnings("ignore")


# ── 1. Load & align predictions ─────────────────────────────────────────────

BACKTEST_COLS = [
    "timestamp",
    "price",
    "sign",
    "qty",
    "spread",
    "midprice",
    "toxic",
]


def load_backtest_data(
    data_dir: Path,
    predictions_path: Path,
    asset: str,
    week: str,
    prediction_column: str = "p_logreg",
) -> pd.DataFrame:
    """
    Load features and align timestamped out-of-sample predictions by row identity.

    Args:
        data_dir: Path to processed features directory.
        predictions_path: Parquet export containing asset, week, source_row,
            timestamp, and the named prediction column.
        asset: Asset name, e.g. 'BTCUSDT'.
        week: Week label, e.g. 'week2'.
        prediction_column: Prediction column to use as the strategy signal.

    Returns:
        Tick-level DataFrame ready for build_bars().
    """
    path = data_dir / f"{asset}_{week}_full_features.parquet"
    df = pd.read_parquet(path, columns=BACKTEST_COLS)
    df["source_row"] = np.arange(len(df), dtype=np.int64)
    df = df.dropna(subset=BACKTEST_COLS).copy()

    prediction_cols = [
        "asset",
        "week",
        "source_row",
        "timestamp",
        prediction_column,
    ]
    predictions = pd.read_parquet(predictions_path, columns=prediction_cols)
    predictions = predictions[
        (predictions["asset"] == asset) & (predictions["week"] == week)
    ].dropna(subset=[prediction_column])

    df = df.merge(
        predictions[["source_row", "timestamp", prediction_column]],
        on="source_row",
        how="inner",
        suffixes=("", "_prediction"),
        validate="one_to_one",
    )
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        feature_ts = pd.to_datetime(df["timestamp"], unit="s")
    else:
        feature_ts = pd.to_datetime(df["timestamp"])
    if pd.api.types.is_numeric_dtype(df["timestamp_prediction"]):
        prediction_ts = pd.to_datetime(df["timestamp_prediction"], unit="s")
    else:
        prediction_ts = pd.to_datetime(df["timestamp_prediction"])

    if not np.array_equal(
        feature_ts.astype("int64").to_numpy(),
        prediction_ts.astype("int64").to_numpy(),
    ):
        raise ValueError("Prediction timestamps do not match feature-row timestamps")

    df["timestamp"] = feature_ts
    df = (
        df.drop(columns=["timestamp_prediction"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df["best_bid"]  = df["midprice"] - df["spread"] / 2
    df["best_ask"]  = df["midprice"] + df["spread"] / 2
    df["mid"]       = df["midprice"]
    return df


# ── 2. Build bars ────────────────────────────────────────────────────────────

def build_bars(
    df: pd.DataFrame,
    signal_column: str = "p_logreg",
) -> pd.DataFrame:
    """
    Aggregate tick-level trades into 1-second bars.

    Args:
        df: Tick-level DataFrame with columns: timestamp, sign, qty,
            price, best_bid, best_ask, mid, and the named prediction column.
            An optional toxic column is retained only as realised ex-post data.
        signal_column: Prediction column to aggregate for later quote decisions.

    Returns:
        Bar-level DataFrame.
    """
    if signal_column not in df.columns:
        raise KeyError(f"Missing prediction column '{signal_column}'")

    df = df.copy()
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("datetime")

    buys  = df[df["sign"] == 1]
    sells = df[df["sign"] == -1]

    bar_columns = {
        "mid":            df["mid"].resample("1s").last(),
        "best_bid":       df["best_bid"].resample("1s").last(),
        "best_ask":       df["best_ask"].resample("1s").last(),
        "any_buy":        buys["qty"].resample("1s").count() > 0,
        "any_sell":       sells["qty"].resample("1s").count() > 0,
        "max_buy_price":  buys["price"].resample("1s").max(),
        "min_sell_price": sells["price"].resample("1s").min(),
        "buy_volume":     buys["qty"].resample("1s").sum(),
        "sell_volume":    sells["qty"].resample("1s").sum(),
        signal_column:    df[signal_column].resample("1s").last(),
        "buy_trades":     buys["qty"].resample("1s").count(),
        "sell_trades":    sells["qty"].resample("1s").count(),
    }
    if "toxic" in df.columns:
        bar_columns["realised_toxic_rate"] = df["toxic"].resample("1s").mean()
    bars = pd.DataFrame(bar_columns)

    bars["any_buy"]          = bars["any_buy"].fillna(False)
    bars["any_sell"]         = bars["any_sell"].fillna(False)
    bars["max_buy_price"]    = bars["max_buy_price"].fillna(0)
    bars["min_sell_price"]   = bars["min_sell_price"].fillna(0)
    bars["buy_volume"]       = bars["buy_volume"].fillna(0)
    bars["sell_volume"]      = bars["sell_volume"].fillna(0)
    bars["buy_trades"]       = bars["buy_trades"].fillna(0)
    bars["sell_trades"]      = bars["sell_trades"].fillna(0)
    if "realised_toxic_rate" in bars.columns:
        bars["realised_toxic_rate"] = bars["realised_toxic_rate"].fillna(0)

    bars = (
        bars.dropna(subset=["mid", signal_column])
        .reset_index()
        .rename(columns={"datetime": "timestamp"})
    )
    return bars


# ── 3. Backtest engine — the core algorithm ─────────────────────────────────

def run_backtest(
    bars: pd.DataFrame,
    k: float,
    signal_column: str = "p_logreg",
    alpha: float = 0.5,
    N: float = 20.0,
    M: float = 10.0,
    sigma_window: int = 300,
) -> pd.DataFrame:
    """
    Vectorised market-making backtest for a named prediction signal.

    All bar-level scalars (sigma, fill_size, gamma, spreads) are precomputed
    as numpy arrays. The inner loop only tracks cash and inventory, which are
    path-dependent and cannot be vectorised without approximation.

    Causal ordering: state and signal observed on bar t place quotes for bar
    t+1; only trades in bar t+1 can fill those quotes. The first bar establishes
    state and cannot fill.

    Args:
        bars: Bar-level DataFrame with columns: mid, best_bid, best_ask,
              any_buy, any_sell, max_buy_price, min_sell_price,
              the named prediction column, and timestamp. A realised target may
              be present for ex-post diagnostics but is never read here.
        k: Prediction spread multiplier. k=0 is the static baseline.
        signal_column: Name of the out-of-sample prediction column used to
            widen quotes, e.g. p_logreg.
        alpha: Adverse selection tolerance for fill size computation.
        N: Inventory limit as multiple of fill_size.
        M: Parameter for gamma derivation (target inventory in fill_size units
           before skew makes quotes uncompetitive).
        sigma_window: Rolling window (bars) for log-return volatility estimate.

    Returns:
        DataFrame with columns: timestamp, mtm_pnl, inventory.
    """
    n = len(bars)
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "mtm_pnl", "inventory"])
    if signal_column not in bars.columns:
        raise KeyError(
            f"Prediction column '{signal_column}' is required; "
            "realised toxicity labels are not valid strategy signals."
        )

    # Extract numpy arrays — eliminates per-row pandas overhead
    mid_arr      = bars["mid"].values
    best_bid_arr = bars["best_bid"].values
    best_ask_arr = bars["best_ask"].values
    any_sell_arr = bars["any_sell"].values
    any_buy_arr  = bars["any_buy"].values
    min_sell_arr = bars["min_sell_price"].values
    max_buy_arr  = bars["max_buy_price"].values
    signal_arr   = bars[signal_column].to_numpy(dtype=float)
    ts_arr       = bars["timestamp"].values
    if not np.isfinite(signal_arr).all():
        raise ValueError(f"Prediction column '{signal_column}' contains missing values")

    # Precompute rolling volatility (single pandas call, not per-row)
    log_ret = np.empty(n)
    log_ret[0] = 0.0
    log_ret[1:] = np.diff(np.log(mid_arr))
    sigma_arr = (
        pd.Series(log_ret)
        .rolling(sigma_window, min_periods=1)
        .std()
        .fillna(1e-6)
        .values
    )
    sigma_arr = np.where(sigma_arr > 0, sigma_arr, 1e-6)

    # Precompute all bar-level quantities that don't depend on inventory
    market_spread   = best_ask_arr - best_bid_arr
    adaptive_spread = market_spread * (1.0 + k * signal_arr)
    fill_size_arr   = alpha * market_spread / (sigma_arr * mid_arr)
    max_inv_arr     = N * fill_size_arr
    gamma_arr       = (market_spread / 2.0) / (M * fill_size_arr * sigma_arr * mid_arr)
    half_spread_arr = adaptive_spread / 2.0

    # Tight loop — only inventory/cash are path-dependent
    cash      = 0.0
    inventory = 0.0
    mtm_arr   = np.zeros(n)
    inv_arr   = np.zeros(n)

    for i in range(1, n):
        state_i = i - 1
        fs = fill_size_arr[state_i]
        quote_mid = mid_arr[state_i]
        current_mid = mid_arr[i]
        hs = half_spread_arr[state_i]

        # Quote from information known at the end of the previous bar.
        skew = (
            -inventory
            * gamma_arr[state_i]
            * sigma_arr[state_i]
            * quote_mid
        )
        res = quote_mid + skew
        mm_bid = res - hs
        mm_ask = res + hs

        # Evaluate the standing quote only against later/current-bar trades.
        if (
            any_sell_arr[i]
            and mm_bid >= min_sell_arr[i]
            and abs(inventory) < max_inv_arr[state_i]
        ):
            cash      -= mm_bid * fs
            inventory += fs

        if (
            any_buy_arr[i]
            and mm_ask <= max_buy_arr[i]
            and abs(inventory) < max_inv_arr[state_i]
        ):
            cash      += mm_ask * fs   # fixed: was mm_bid in original notebook
            inventory -= fs

        mtm_arr[i] = cash + inventory * current_mid
        inv_arr[i] = inventory

    return pd.DataFrame({"timestamp": ts_arr, "mtm_pnl": mtm_arr, "inventory": inv_arr})


# ── 4. Bootstrap: resample + confidence interval ────────────────────────────

class BootstrapResult(NamedTuple):
    asset: str
    week: str
    observed_improvement: float
    ci_lower: float
    ci_upper: float
    n_blocks: int
    n_bootstrap: int
    mean_bootstrap: float
    std_bootstrap: float


def _run_both_strategies(
    bars: pd.DataFrame,
    k_baseline: float,
    k_adaptive: float,
    signal_column: str = "p_logreg",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both strategies once and return their full result DataFrames.

    Shared by _mtm_improvement (which only needs the final scalar
    difference, e.g. inside the 200x bootstrap loop where each resample is
    genuinely new data) and by callers that need the full per-bar output —
    compute_bootstrap_ci's observed value and build_regime_table both used
    to call run_backtest() a second and third time on the same unresampled
    bars just to get this; now they reuse one computation instead.
    """
    base = run_backtest(bars, k=k_baseline, signal_column=signal_column)
    adap = run_backtest(bars, k=k_adaptive, signal_column=signal_column)
    return base, adap


def _mtm_improvement(
    bars: pd.DataFrame,
    k_baseline: float,
    k_adaptive: float,
    signal_column: str = "p_logreg",
) -> float:
    """Run both strategies on bars and return final MtM improvement."""
    base, adap = _run_both_strategies(bars, k_baseline, k_adaptive, signal_column)
    return float(adap["mtm_pnl"].iloc[-1] - base["mtm_pnl"].iloc[-1])


def block_bootstrap_improvement(
    bars: pd.DataFrame,
    k_baseline: float = 0,
    k_adaptive: float = 5,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
    signal_column: str = "p_logreg",
) -> tuple[np.ndarray, int]:
    """
    Block-bootstrap the MtM improvement statistic.

    Bootstrap unit is calendar day. For each sample:
      1. Resample days with replacement (same number of days as original).
      2. Concatenate the daily bar sequences.
      3. Reset index (required for sigma_arr positional indexing).
      4. Run both strategies, record improvement.

    Why daily blocks: MtM is a cumulative path — consecutive bars are strongly
    autocorrelated within a day. Daily blocks preserve this structure while
    treating days as approximately exchangeable.

    Args:
        bars: Full bar-level DataFrame for one asset-week.
        k_baseline: k for static strategy.
        k_adaptive: k for adaptive strategy.
        n_bootstrap: Number of bootstrap samples.
        rng: numpy Generator for reproducibility.
        signal_column: Named prediction column used by both strategies.

    Returns:
        (improvements array of shape (n_bootstrap,), n_blocks)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    bars = bars.copy()
    bars["_date"] = pd.to_datetime(bars["timestamp"]).dt.date
    days = bars["_date"].unique()
    n_blocks = len(days)
    day_groups = {d: bars[bars["_date"] == d].drop(columns=["_date"]) for d in days}

    improvements = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled = rng.choice(days, size=n_blocks, replace=True)
        resampled = pd.concat(
            [day_groups[d] for d in sampled], ignore_index=True
        )
        improvements[b] = _mtm_improvement(
            resampled,
            k_baseline,
            k_adaptive,
            signal_column,
        )

    return improvements, n_blocks


def compute_bootstrap_ci(
    bars: pd.DataFrame,
    res_base: pd.DataFrame,
    res_adap: pd.DataFrame,
    asset: str,
    week: str,
    k_baseline: float = 0,
    k_adaptive: float = 5,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    signal_column: str = "p_logreg",
) -> BootstrapResult:
    """
    Compute observed MtM improvement and percentile bootstrap CI.

    Uses percentile bootstrap (not BCa). Appropriate here because the
    statistic is a scalar difference of cumulative sums — no ratio,
    no skew correction needed.

    Args:
        bars: Bar-level DataFrame for one asset-week (used for the
            resampled bootstrap runs — each resample is genuinely new data
            and must be backtested fresh).
        res_base: Already-computed run_backtest() output for k_baseline on
            the unresampled bars, from _run_both_strategies() — reused here
            instead of rerun, since the caller already needed it.
        res_adap: Already-computed run_backtest() output for k_adaptive on
            the unresampled bars, same reasoning.
        asset: Asset label for reporting.
        week: Week label for reporting.
        k_baseline: k for static strategy.
        k_adaptive: k for adaptive strategy.
        n_bootstrap: Number of bootstrap resamples.
        alpha: CI level (0.05 → 95% CI).
        rng: numpy Generator for reproducibility.
        signal_column: Named prediction column used by both strategies.

    Returns:
        BootstrapResult namedtuple.
    """
    observed = float(res_adap["mtm_pnl"].iloc[-1] - res_base["mtm_pnl"].iloc[-1])
    boot_dist, n_blocks = block_bootstrap_improvement(
        bars,
        k_baseline,
        k_adaptive,
        n_bootstrap,
        rng,
        signal_column,
    )
    ci_lo = float(np.percentile(boot_dist, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_dist, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        asset=asset,
        week=week,
        observed_improvement=float(observed),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_blocks=n_blocks,
        n_bootstrap=n_bootstrap,
        mean_bootstrap=float(boot_dist.mean()),
        std_bootstrap=float(boot_dist.std()),
    )


# ── 5. Regime table & reporting ─────────────────────────────────────────────

def toxic_fill_rate(bars: pd.DataFrame, results: pd.DataFrame) -> float:
    """
    Fraction of fills on bars with any toxic activity.

    A fill occurred when inventory changed between consecutive bars.

    Args:
        bars: Bar-level DataFrame (same length as results, aligned by position).
        results: Backtest output with inventory column.

    Returns:
        Fraction of fills on toxic bars (0–1), or nan if no fills.
    """
    filled = results["inventory"].diff().abs().values > 1e-10
    if "realised_toxic_rate" not in bars.columns:
        return float("nan")
    toxic = bars["realised_toxic_rate"].values > 0
    if filled.sum() == 0:
        return float("nan")
    return float(toxic[filled].mean())


def avg_spread_ratio(
    bars: pd.DataFrame,
    k: float,
    signal_column: str = "p_logreg",
) -> float:
    """
    Mean ratio of adaptive spread to market spread across all bars.

    Args:
        bars: Bar-level DataFrame with a prediction signal, best_bid, best_ask.
        k: Prediction multiplier.
        signal_column: Named prediction column used to widen spreads.

    Returns:
        Mean (adaptive_spread / market_spread).
    """
    ms = bars["best_ask"].values - bars["best_bid"].values
    as_ = ms * (1.0 + k * bars[signal_column].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(ms > 0, as_ / ms, np.nan)
    return float(np.nanmean(ratios))


def build_regime_table(
    all_bars: dict[tuple[str, str], pd.DataFrame],
    backtest_results: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]],
    bootstrap_results: list[BootstrapResult],
    k_adaptive: float = 5,
    out_of_sample_weeks: list[str] | None = None,
    signal_column: str = "p_logreg",
) -> pd.DataFrame:
    """
    Build per-asset × per-week results table for the README.

    Args:
        all_bars: Dict mapping (asset, week) → bar DataFrame.
        backtest_results: Dict mapping (asset, week) → (res_base, res_adap),
            the already-computed unresampled backtest runs from
            _run_both_strategies() — reused here instead of rerunning
            run_backtest() a second/third time on the same bars.
        bootstrap_results: List of BootstrapResult for out-of-sample weeks.
        k_adaptive: k for adaptive strategy, used only by avg_spread_ratio
            (k_baseline isn't needed here — it's already baked into
            backtest_results).
        out_of_sample_weeks: Weeks to include (default: week2, week3).
        signal_column: Named prediction column used by the adaptive strategy.

    Returns:
        DataFrame with columns: Asset, Week, Baseline MtM, Adaptive MtM,
        Improvement, 95% CI, Toxic Fill Rate (Base/Adaptive), Avg Spread Ratio.
    """
    if out_of_sample_weeks is None:
        out_of_sample_weeks = ["week2", "week3"]

    ci_lookup = {(r.asset, r.week): r for r in bootstrap_results}
    rows = []

    for (asset, week), bars in sorted(all_bars.items()):
        if week not in out_of_sample_weeks:
            continue

        res_base, res_adap = backtest_results[(asset, week)]
        mtm_base    = res_base["mtm_pnl"].iloc[-1]
        mtm_adap    = res_adap["mtm_pnl"].iloc[-1]
        improvement = mtm_adap - mtm_base

        ci    = ci_lookup.get((asset, week))
        ci_lo = ci.ci_lower if ci else float("nan")
        ci_hi = ci.ci_upper if ci else float("nan")

        tfr_base = toxic_fill_rate(bars, res_base)
        tfr_adap = toxic_fill_rate(bars, res_adap)
        sr = avg_spread_ratio(bars, k_adaptive, signal_column)

        rows.append({
            "Asset":                       asset.replace("USDT", ""),
            "Week":                        week,
            "Baseline MtM ($)":            round(mtm_base, 0),
            "Adaptive MtM ($)":            round(mtm_adap, 0),
            "Improvement ($)":             round(improvement, 0),
            "95% CI":                      f"[{ci_lo:+.0f}, {ci_hi:+.0f}]" if ci else "n/a",
            "Toxic Fill Rate — Base":      f"{tfr_base:.1%}" if not np.isnan(tfr_base) else "n/a",
            "Toxic Fill Rate — Adaptive":  f"{tfr_adap:.1%}" if not np.isnan(tfr_adap) else "n/a",
            "Avg Spread Ratio":            f"{sr:.3f}",
        })

    return pd.DataFrame(rows)


# ── 6. Main ──────────────────────────────────────────────────────────────────

def main(
    data_dir: str = "data/processed/features",
    predictions_path: str = "results/predictions/split2_week3_predictions.parquet",
    signal_column: str = "p_logreg",
    n_bootstrap: int = 200,
    k_baseline: float = 0,
    k_adaptive: float = 5,
    out_of_sample_weeks: list[str] | None = None,
    seed: int = 42,
) -> None:
    """
    Run Session 15 statistical hardening: bootstrap CIs + regime table.

    Args:
        data_dir: Path to processed features parquet files.
        predictions_path: Path to timestamped out-of-sample prediction export.
        signal_column: Prediction column used by the adaptive strategy.
        n_bootstrap: Number of bootstrap resamples (200 is sufficient given
                     ~7 daily blocks per week; going higher barely moves
                     the percentile estimates).
        k_baseline: k for static strategy.
        k_adaptive: k for adaptive strategy.
        out_of_sample_weeks: Weeks to evaluate. Defaults to week3 because the
                             bundled export workflow produces split-2 week-3
                             predictions only.
        seed: Random seed for reproducibility.

    This command creates new prediction-driven outputs. It does not repair or
    validate the historical PnL tables, which have not been rerun.
    """
    if out_of_sample_weeks is None:
        out_of_sample_weeks = ["week3"]

    assets    = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    data_path = Path(data_dir)
    prediction_path = Path(predictions_path)
    rng       = np.random.default_rng(seed)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    all_data: dict[tuple[str, str], pd.DataFrame] = {}
    for asset in assets:
        for week in out_of_sample_weeks:
            df = load_backtest_data(
                data_path,
                prediction_path,
                asset,
                week,
                signal_column,
            )
            all_data[(asset, week)] = df
            print(f"  {asset} {week}: {len(df):,} ticks")

    # ── Build bars ────────────────────────────────────────────────────────────
    print("\nBuilding 1-second bars...")
    all_bars: dict[tuple[str, str], pd.DataFrame] = {}
    for (asset, week), df in all_data.items():
        bars = build_bars(df, signal_column=signal_column)
        all_bars[(asset, week)] = bars
        print(f"  {asset} {week}: {len(bars):,} bars")

    # ── Backtest (once per asset-week, shared by bootstrap + regime table) ─────
    print("\nRunning baseline + adaptive backtest on unresampled bars...")
    backtest_results: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for asset in assets:
        for week in out_of_sample_weeks:
            bars = all_bars[(asset, week)]
            backtest_results[(asset, week)] = _run_both_strategies(
                bars, k_baseline, k_adaptive, signal_column
            )

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    print(f"\nRunning block bootstrap (n={n_bootstrap}, seed={seed})...")
    print("Bootstrap unit: calendar day.")
    print("~7 blocks/week → wide CIs expected. This is the honest result.\n")

    bootstrap_results: list[BootstrapResult] = []
    for asset in assets:
        for week in out_of_sample_weeks:
            bars = all_bars[(asset, week)]
            res_base, res_adap = backtest_results[(asset, week)]
            print(f"  {asset} {week}...", end=" ", flush=True)
            result = compute_bootstrap_ci(
                bars=bars,
                res_base=res_base,
                res_adap=res_adap,
                asset=asset,
                week=week,
                k_baseline=k_baseline,
                k_adaptive=k_adaptive,
                n_bootstrap=n_bootstrap,
                rng=rng,
                signal_column=signal_column,
            )
            bootstrap_results.append(result)
            print(
                f"observed ${result.observed_improvement:+,.0f} | "
                f"95% CI [{result.ci_lower:+,.0f}, {result.ci_upper:+,.0f}] | "
                f"{result.n_blocks} blocks"
            )

    # ── Regime table ──────────────────────────────────────────────────────────
    print("\nBuilding per-regime breakdown table...")
    table = build_regime_table(
        all_bars=all_bars,
        backtest_results=backtest_results,
        bootstrap_results=bootstrap_results,
        k_adaptive=k_adaptive,
        out_of_sample_weeks=out_of_sample_weeks,
        signal_column=signal_column,
    )

    print("\n" + "=" * 90)
    print("PER-REGIME BREAKDOWN TABLE (out-of-sample only)")
    print("=" * 90)
    print(table.to_string(index=False))

    # ── README narrative ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("README NARRATIVE — paste into limitations / results section")
    print("=" * 90)
    for r in bootstrap_results:
        asset_short = r.asset.replace("USDT", "")
        ci_width    = r.ci_upper - r.ci_lower
        sign_str    = "positive" if r.ci_lower > 0 else (
                      "negative" if r.ci_upper < 0 else "spans zero")
        print(
            f"\n{asset_short} {r.week}: adaptive strategy shows "
            f"${r.observed_improvement:+,.0f} MtM improvement "
            f"(block bootstrap 95% CI: [{r.ci_lower:+,.0f}, {r.ci_upper:+,.0f}], "
            f"n={r.n_blocks} daily blocks, B={r.n_bootstrap}). "
            f"CI {sign_str}. "
            f"CI width ${ci_width:,.0f} reflects ~{r.n_blocks} exchangeable blocks."
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "session15_regime_table.csv", index=False)
    pd.DataFrame([r._asdict() for r in bootstrap_results]).to_csv(
        out_dir / "session15_bootstrap_results.csv", index=False
    )
    print(f"\nSaved → results/session15_regime_table.csv")
    print(f"Saved → results/session15_bootstrap_results.csv")


if __name__ == "__main__":
    main()
