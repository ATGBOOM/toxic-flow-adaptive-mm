# Code Walkthrough

This guide follows one observation from raw Bybit data to a model score and then
to the experimental market-making simulation.

## 1. End-to-End Data Flow

```text
Bybit trades -> cleaned weekly trade parquet
             -> reconstructed order book at trade times
             -> trade and book features
             -> forward adverse-move label
             -> VPIN
             -> pooled walk-forward classifier
             -> model probabilities

Feature parquet -> one-second bars -> quote simulation -> MtM PnL
```

The historical backtest path was not connected to classifier predictions and
consumed the realized label. The hardened reusable path now exports timestamped
out-of-sample probabilities, aligns them by asset/week/source row, and consumes
the named prediction while retaining the realized label only for ex-post
diagnostics. The historical PnL was not rerun after this change.

## 2. Trade Loading

### `src/data/loader.py`

`load_trades(file_path)`:

1. Reads `timestamp`, `side`, `size`, and `price` from a gzipped CSV.
2. Converts Unix seconds to pandas datetimes.
3. maps aggressor `Buy` to `+1` and `Sell` to `-1`.
4. Renames `size` to `qty`.
5. Reports gaps over 60 seconds.

Interpretation of `sign`: it is the aggressor's side, not the market maker's
side. A buy aggressor consumes asks; the market maker would sell.

Defense point: gap reporting is diagnostic only. It does not repair, reject, or
annotate affected observations.

### `src/data/pipeline.py`

`WEEK_RANGES` defines the three hand-selected regimes. `date_to_week()` performs
a lexicographic date lookup, which works because dates use ISO `YYYY-MM-DD`.

`process_all()`:

- discovers asset directories;
- loads each daily file;
- groups frames by selected week;
- concatenates and writes one parquet per asset-week.

The output contract is:

```text
timestamp: datetime64
price: float
qty: float
sign: +1 or -1
```

Decision to defend: Parquet makes expensive pipeline stages independently
restartable. Limitation: there is no manifest containing source URLs, hashes,
row counts, timezone, or missing intervals.

## 3. Order Book Reconstruction

### `apply_update(bids, asks, message)`

The book is stored as two mutable dictionaries:

```text
price -> resting quantity
```

A snapshot clears both sides. A delta with size zero deletes a level; otherwise
it replaces that level's quantity.

Why dictionaries: updates are cheap by exact price. Cost is paid later when all
prices are sorted to compute features.

### `compute_book_features(bids, asks)`

Best prices:

```text
best_bid = max(bid prices)
best_ask = min(ask prices)
mid = (best_bid + best_ask) / 2
spread = best_ask - best_bid
```

Microprice:

```text
microprice =
    (best_bid * ask_size + best_ask * bid_size)
    / (bid_size + ask_size)
```

It weights toward the side likely to move next. More bid size pushes the
microprice toward the ask; more ask size pushes it toward the bid.

Depth imbalance at `N` levels:

```text
(bid_volume_N - ask_volume_N) / (bid_volume_N + ask_volume_N)
```

It ranges from `-1` to `+1`. Positive means more visible bid depth.

Pressure features compare top-five depth with top-25 depth separately on each
side. `pressure_imbalance` is the difference in those concentration ratios.

### `reconstruct_and_extract_from_state(...)`

The function replays book messages in timestamp order. Before applying the next
book message, it emits the current book state for trades with an earlier
timestamp. Mutable dictionaries are passed between days to preserve state.

The intended causal rule is: use the latest book state known before a trade.

Important edge cases:

- Equal book/trade timestamps are resolved in favor of applying the book update.
- Trades after the final message receive a stale final state.
- Missing days leave the prior state alive until another snapshot resets it.
- Float prices can be less robust than integer ticks.

### `process_week(...)`

This selects each day's trades, replays that day's zipped NDJSON book file, and
writes one book-feature parquet per day. Missing order-book files are skipped.

## 4. Trade Features and Labels

### `add_trade_features(trades_df)`

`np.searchsorted` finds the first index inside each trailing window. This avoids
an expensive pandas rolling operation per row.

Trade intensity:

```text
current_index - first_index_at_or_after(t - window) + 1
```

Features are computed immediately after observing the current trade. The
trailing interval is inclusive, `[t - window, t]`: the current trade and a trade
exactly on the lower boundary are counted, while no future trade is used.

Volume acceleration:

```text
volume_last_5s / (volume_last_30s * 5/30)
```

Above one means recent volume is faster than its 30-second local rate.

Signed volume imbalance:

```text
sum(sign * qty) over the trailing 10 seconds
```

Positive means net aggressive buying; negative means net aggressive selling.

### `add_toxicity_label(...)`

For each trade, the code finds the first trade at or after `t + 10 seconds` and
computes:

```text
forward_bps = (future_price - current_price) / current_price * 10,000
```

The label is true when:

```text
buy aggressor  and forward_bps > +8
sell aggressor and forward_bps < -8
```

Interpretation: the aggressor traded immediately before a sufficiently large
same-direction move. This is a proxy for adverse selection, not proof that the
trader possessed private information. Rows for which the data does not extend
through the full requested horizon are invalid (`NA`) and are dropped by model
and evaluation loaders rather than clipped to the final price.

### `add_vpin_feature(...)`

The function computes bucket-level VPIN and assigns each trade the latest
completed VPIN value. Trades in the warm-up period are set to missing and later
dropped by the model loader.

### `build_full_features(...)`

This function:

1. loads the weekly trades;
2. loads available daily book features;
3. aligns rows by position after a count check;
4. adds trade features;
5. creates the label;
6. adds VPIN;
7. writes one full feature parquet.

Alignment is positional, not a timestamp join. The equal-row-count check protects
against obvious mismatch but not a same-length ordering error.

## 5. VPIN

### Idea

Calendar-time bars contain radically different amounts of information when
activity changes. VPIN instead advances after a fixed amount of traded volume.

For bucket `i`:

```text
imbalance_i = abs(buy_volume_i - sell_volume_i)
```

Over the latest `n` equal-volume buckets of size `V`:

```text
VPIN = sum(imbalance_i) / (n * V)
```

The output lies between zero and one. High values indicate one-sided aggressive
flow, not necessarily informed trading.

### `build_volume_bucket(df, bucket_size)`

Each trade is allocated into the current bucket. A large trade can be split over
multiple buckets. Partial final buckets are discarded in the canonical module.

Known bug: a trade that exactly fills a bucket does not flush it until the next
trade. This delays the bucket timestamp.

### `compute_rolling_vpins(...)`

For every complete rolling window, it sums absolute bucket imbalances and
normalizes by total window volume.

### `src/evaluations/vpin_robustness.py`

This is a second implementation used for the 4x4 parameter grid. It is faster
because rolling sums use cumulative sums. It should be consolidated with the
canonical implementation before a final defense.

Defense of the negative result: standard long-window VPIN can average opposing
flow together. What is demonstrated is low predictive ranking under the tested
definition and sample. The proposed two-sided-flow explanation is a hypothesis.

## 6. Model Data Preparation

### `load_asset_week(...)`

The loader creates:

```text
spread_bps = spread / midprice * 10,000
microprice_minus_mid = microprice - midprice
qty_normalised = qty / rolling_mean_1000(qty)
```

It then removes VPIN warm-up rows and all remaining missing values.

`microprice_minus_mid` should be normalized before claiming cross-asset scale
invariance.

### `get_feature_columns(...)`

The 16 inputs contain:

- four depth imbalances;
- bid, ask, and relative pressure;
- three trade intensity windows;
- volume acceleration;
- signed volume imbalance;
- VPIN;
- relative spread;
- microprice displacement;
- normalized quantity.

### `subsample_stratified(...)`

The function samples toxic and non-toxic rows separately to preserve the
original class ratio. It does not rebalance classes.

### `prepare_split(...)`

Training weeks are pooled across assets and randomly subsampled. Test weeks are
loaded in full. The temporal separation between weeks is correct; the random
operation occurs only inside the training period.

## 7. Classifiers

### VPIN baseline

`VPINBaseline` returns raw VPIN as a score. It is suitable for ranking metrics
such as AUC and AP, but it is not a calibrated toxicity probability.

### Logistic regression

Inputs are standardized using training means and standard deviations:

```text
z_j = (x_j - training_mean_j) / training_std_j
```

The model estimates:

```text
P(toxic | x) = sigmoid(beta_0 + sum(beta_j * z_j))
```

Why it can transfer well: a linear, regularized boundary has lower variance than
a deep interaction model. Why coefficient signs require care: correlated
intensity windows make each coefficient conditional on the others.

The sklearn default applies L2 regularization even though the design document
once says "no regularisation initially."

### CatBoost

The model uses 500 depth-six trees, learning rate `0.05`, and L2 leaf penalty
`3`. It can learn nonlinear thresholds and interactions without standardization.

No time-based hyperparameter tuning loop is committed. The stated parameters
should be described as fixed experimental choices, not fully tuned optima.

### Metrics

- ROC AUC: probability that a random positive ranks above a random negative.
- AP: area-like summary of precision across recall; no-skill AP equals prevalence.
- Brier: mean squared probability error; evaluates calibration and resolution.
- Precision: fraction of alerts that are toxic.
- Recall: fraction of toxic observations detected.

In imbalanced data, AP is more operationally informative than accuracy.

## 8. Prediction Evaluation Notebook

The notebook studies reliability, Brier decomposition, thresholds, SHAP, and AP
uncertainty. Its stored outputs are research artifacts, not a fully reproducible
pipeline because predictions and models are not committed.

Correct interpretation of the week-3 result:

- Logistic regression ranks toxic observations better than VPIN.
- Its probabilities understate the selected week's prevalence.
- Its useful high-precision region has low recall.
- Week 3 is a selected stress period and also functions as model validation.

## 9. Market-Making Simulation

### Bar construction

Trades are grouped into one-second bars. Each bar stores book state, aggressor
presence and extreme price, volume, and realized toxic-label rate.

### Quote rule

The simulation uses:

```text
adaptive_spread = market_spread * (1 + k * signal)
skew = -inventory * gamma * sigma * mid
reservation_price = mid + skew
bid = reservation_price - adaptive_spread/2
ask = reservation_price + adaptive_spread/2
```

Long inventory makes skew negative, lowering both quotes to encourage selling.
Short inventory makes skew positive, raising both quotes to encourage buying.

### Fill rule

- A sell aggressor fills the bid when its price reaches or passes the bid.
- A buy aggressor fills the ask when its price reaches or passes the ask.
- One fixed-size fill is allowed per side per bar.

Cash and inventory:

```text
MM buys:  cash -= bid * qty; inventory += qty
MM sells: cash += ask * qty; inventory -= qty
MtM = cash + inventory * current_mid
```

### What the reusable path currently tests

The reusable path requires a named prediction such as `p_logreg`. Asset, week,
timestamp, and source-row identity align predictions to feature rows. State and
signal from bar `t` place quotes that can only fill on bar `t+1`. A realised
label may be retained for ex-post diagnostics but is not read by the strategy.

The historical notebook/README PnL values came from the former realised-label
path and have not been rerun, so they remain oracle-signal results.

## 10. Tests

`test_vpin_hand_computed()` and `test_vpin_bucket_closes_on_exact_fill()` check
that exact fills close immediately, use the filling trade timestamp, and produce
the hand-computed rolling values.

`test_pnl_round_trip_known_spread()` checks accounting arithmetic in isolation.
It does not call the real backtest engine.

`test_no_lookahead_leakage_rolling_features()` uses a truncation invariant:
removing future rows must not change earlier feature values. This is a good
general leakage test for backward-looking features.

Additional boundary tests hand-compute inclusive rolling windows, reject clipped
terminal labels, verify invalid-label filtering and identity preservation, prove
that changing the realised target cannot alter decisions, and enforce
quote-before-fill ordering.

Remaining missing tests include:

- order-book timestamp ordering;
- data gaps inside otherwise valid label horizons;
- prediction-to-bar alignment;
- inventory limits and two-sided fills;
- deterministic end-to-end synthetic integration.

## 11. Remaining Repository Files

`src/models/classifier/investigate.py` runs three exploratory CatBoost checks:
balanced versus unbalanced classes, 500,000 versus 2 million training rows, and
with versus without an asset identifier. Each comparison uses one random seed,
so conclusions should be called indicative rather than stable learning curves.

`src/evaluations/save_predictions.py` retrains the pooled week-1-plus-week-2
models, serializes them, and writes week-3 predictions with asset, week,
timestamp, and source-row identity for causal backtest alignment.

`src/evaluations/bootstrap_eval.py` contains the current reusable backtest,
daily-block bootstrap, and table generation. The strategy path is causally
ordered and prediction-driven, but the historical metrics have not been rerun
and the bootstrap limitations still apply.

`src/evaluations/evaluate.py` is empty.

`notebooks/01_eda.ipynb` contains BTC-focused return and OFI exploration.

`notebooks/view_xy_thresholds.ipynb` contains the original label threshold and
feature exploration, primarily on one BTC day.

`notebooks/03_classifier_evaluation.ipynb` contains calibration, Brier,
precision-recall, SHAP, and bootstrap analysis.

`notebooks/04_backtest_results.ipynb` is an older finite-horizon A-S experiment.
It revealed that the chosen `kappa` estimate did not control quotes as intended.

`notebooks/05_backtest_results_with_infinte_time_horizon.ipynb` is the later
market-spread and inventory-skew experiment from which the reusable bootstrap
module was derived. Its filename contains the typo `infinte`.

## 12. Study Order

Use this order and do not move on until you can answer the checkpoint aloud.

1. **Data and book reconstruction**

   Checkpoint: Given one snapshot, two deltas, and one trade timestamp, manually
   state the book used for that trade and compute spread, mid, microprice, and
   depth imbalance.

2. **Label and features**

   Checkpoint: Given a five-trade timeline, manually compute one intensity,
   signed-volume, forward-return, and toxicity-label row. Explain exactly which
   values are available in real time.

3. **VPIN**

   Checkpoint: Split trades across two fixed-volume buckets and calculate one
   rolling VPIN value by hand. Explain why it detects one-sided flow.

4. **Classifiers and metrics**

   Checkpoint: Explain why AP depends on prevalence, why scaling is needed for
   logistic regression, and why CatBoost can fail under regime shift.

5. **Market making**

   Checkpoint: Starting from cash zero and inventory zero, process one bid fill
   and one ask fill, calculate MtM, and identify when the signal and quote must
   be known to avoid look-ahead.

6. **Research defense**

   Checkpoint: State the strongest supported conclusion, the largest invalid
   claim, and the experiment that would resolve it in under 60 seconds.
