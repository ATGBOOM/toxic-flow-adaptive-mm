# Technical Audit Findings

This document separates what the repository currently demonstrates from what its
README claims. Use it as the correction checklist before presenting the project.

## Executive Assessment

The strongest defensible part of the project is the research framing and the
walk-forward classifier comparison. The weakest part is the historical
market-making evaluation: the recorded backtest used the realized
forward-looking toxicity label, not classifier predictions. Its PnL results are
therefore an oracle-signal experiment, not evidence that the classifier improves
market-making PnL. The reusable path has since been causally hardened, but those
historical results have not been rerun.

The repository also contains useful negative results. VPIN has little ranking
power at the chosen standard parameters, while a simple linear model transfers
better than CatBoost into the selected stress week. These should be framed as
evidence from a small, deliberately selected sample, not universal conclusions
about crypto markets.

## Claim Status

| Claim | Status | Reason |
|---|---|---|
| Standard-parameter VPIN has AUC near 0.5 | Supported by recorded outputs, but needs a dependence-aware CI | AUC alone does not establish "indistinguishable from random" |
| Short-window VPIN partially recovers signal | Supported by the recorded parameter grid | The mechanism offered is plausible but not causally tested |
| Logistic regression beats VPIN on week 3 AP | Supported by recorded predictions | AP uncertainty calculation is IID and understates episode dependence |
| Logistic regression generalizes better than CatBoost | Supported on the selected week 3 | Week 3 was also used for model selection, so it is validation evidence |
| Trade intensity is the dominant GBT feature | Not established by the current SHAP table | The displayed values are signed means from only five observations |
| Classifier-driven adaptive MM improves all six OOS cells | Not supported | Recorded results used the realized `toxic` label, not `p_logreg` |
| ETH week 3 improvement is statistically distinguishable | Not reliable as currently computed | Four daily blocks, invalid block concatenation, and oracle signal |

## Critical Findings

### 1. Historical backtest signal was forward-looking

The historical implementation of `build_bars()` computed:

```python
"toxic_rate": df["toxic"].resample("1s").mean()
```

The `toxic` label is defined using the price about ten seconds in the future.
`run_backtest()` then uses that same bar's `toxic_rate` to set its spread. This
gives the strategy information that would not be available when quotes are
placed.

The notebook text says the previous bar is used, but the code does not shift the
signal. Even a one-bar shift would still leak because the previous bar's label
depends on prices up to roughly nine seconds after the current bar begins.

Implemented during the interview-preparation audit:

1. Save timestamped out-of-sample `p_logreg` values.
2. Aggregate only predictions available before quote placement.
3. Quote from the previous observed book state.
4. Test fills only against later trades.
5. Added synthetic tests for target-signal separation and quote-before-fill
   ordering.

Still outstanding: recompute every PnL result and confidence interval. Until
that happens, the recorded table remains an **oracle toxicity upper-bound
experiment**.

### 2. Historical quote and fill events were not causally ordered

The historical bar path took the last book state in a one-second bar and checked
that quote against aggressive trades in the same bar. The reusable path now
uses state and signal from bar `t` for quotes evaluated against bar `t+1`.

A defensible bar-level design uses state known at the end of bar `t` to place
quotes for bar `t+1`. An event-driven design is preferable.

### 3. The SHAP importance table is computed incorrectly

The notebook prints:

```python
shap_values[:5].mean(axis=0)
```

The README values match that five-row signed mean. Global SHAP importance should
normally be:

```python
np.abs(shap_values).mean(axis=0)
```

over the full explanation sample. Signed means can cancel, and five observations
cannot support a global ranking.

### 4. Bootstrap assumptions do not match the data

The AP bootstrap samples individual trades independently after selecting one
fixed 50,000-row subsample. Toxic labels overlap over ten-second horizons and
are highly clustered, so rows are not independent.

The PnL bootstrap concatenates randomly selected days while carrying inventory
between them. This creates artificial close-to-open transitions, including
backward timestamp jumps and volatility shocks at block boundaries.

Required correction:

- For AP, use paired temporal blocks and bootstrap the AP difference between
  models.
- For PnL, reset state per day and bootstrap paired daily strategy differences,
  or use a carefully defined stationary/circular block bootstrap.
- With only four to seven days, report intervals as exploratory regardless.

## High-Priority Methodological Findings

### Toxicity label

- The label measures ex-post adverse price movement, not informed trader identity.
- It uses trade price rather than midprice, so bid-ask bounce can enter the target.
- The final horizon is clipped to the last row, creating invalid trailing labels.
- A search across a data gap can use the next available day as the "10-second"
  future price.
- The 8 bps threshold was selected from one BTC sample and then applied to all
  assets. That is defensible as a common economic threshold, but not as a common
  percentile threshold.
- Overlapping ten-second labels create clusters by construction. The 99.2%
  clustering statistic is not evidence of a Hawkes process without an
  intensity-preserving null model.

Preferred wording: "The target is an adverse-move proxy for toxicity."

### Features

- `microprice_minus_mid` is in dollars and is not cross-asset normalized.
  Use `(microprice - midprice) / midprice * 10_000`.
- Excluding `sign` does not fully enforce directional symmetry because
  `signed_vol_imbalance_10s` remains directional.
- Trade features now use explicit immediately-after-current-trade semantics:
  trailing windows include the current trade and lower boundary, with no future
  trades.
- Pooling assets weights metrics by row count. Report macro-averaged and
  per-asset metrics as well as pooled metrics.

### VPIN

- The implementation uses observed aggressor side. That is an order-flow VPIN
  variant; it is not identical to the bulk-volume-classification estimator in
  the original literature.
- The canonical implementation previously delayed exact-fill buckets. It now
  closes them immediately, timestamps them with the filling trade, and has
  hand-computed regression coverage.
- The robustness script contains a separate, different implementation and
  includes a partial final bucket normalized as if it were full.
- `add_vpin_feature()` always divides weekly volume by seven, including
  truncated weeks. The robustness script uses a four-day ETH average instead.
- Hard-coded daily volumes should be derived from input data and saved with a
  data manifest.

### Classifier evaluation

- Week 3 is used to choose logistic regression over CatBoost and to report final
  performance. It is therefore a validation period, not an untouched final test.
- Randomly sampling 500,000 trades is reproducible but contains many neighboring,
  highly similar rows. Effective sample size is much smaller than row count.
- AP should always be interpreted relative to prevalence. VPIN AP of 0.177 on
  week 3 is approximately the no-skill prevalence baseline.
- Comparing non-overlapping individual CIs is not a paired significance test.
- Brier comparison against the test-set base rate is a useful diagnostic but
  uses hindsight knowledge of that test prevalence.
- The stated 0.38 "breakeven precision" assumes fixed 5 bps profit and 8 bps
  loss. Those values are not the actual fill-conditioned economics of the
  backtest and omit fees, rebates, queueing, and the distribution of losses.

## Backtest Model Limitations

- Fill size is not dimensionally anchored to a dollar risk budget. The expression
  `spread / (sigma * mid)` is dimensionless but is treated as base-asset units.
- The estimated `kappa` is trades per bar, whereas A-S `kappa` is the sensitivity
  of fill intensity to quote distance. The resulting strategy is
  A-S-inspired, not a calibrated Avellaneda-Stoikov solution.
- Given the chosen formulas, `gamma` simplifies algebraically to
  `1 / (2 * M * alpha)`, so it is constant rather than dynamically estimated.
- Queue position, available fill quantity, latency, fees, maker rebates,
  funding, liquidation, and cross-venue hedging are omitted.
- The inventory check can overshoot its limit by one fill.
- Net inventory change misses two-sided fills that cancel within the same bar,
  so `toxic_fill_rate()` does not count all fills.
- "A passive market maker loses money, as expected" is too strong. Profitability
  depends on spread capture, adverse selection, inventory management, and costs.

## Engineering and Reproducibility

- Raw data, processed data, saved predictions, models, and result CSVs are not
  committed, so recorded results cannot be independently rerun from this clone.
- The README lists files and directories that do not exist.
- `src/data/pipeline.py` uses a script-local import (`from loader import ...`)
  rather than a package import.
- There is no executable backtest module under `src/backtest`; key historical
  logic remains in notebooks.
- The original PnL arithmetic test remains, and new causal tests invoke the
  reusable backtest directly.
- The focused synthetic suite runs without external data and covers VPIN
  exact-fill behavior, rolling-window boundaries, invalid label horizons,
  prediction identity/alignment, and causal backtest signal/fill ordering.

## Recommended Presentation Position

Present the project as a **methodological transfer study with a negative VPIN
result and an exploratory classifier result**.

Do not present the historical PnL table as classifier performance. The reusable
code is now causally ordered and prediction-driven, but the table was not rerun;
label it as an oracle upper bound until new artifacts exist.
