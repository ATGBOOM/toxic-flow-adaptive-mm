# Architecture

## System Diagram

```
Raw Bybit tick data
  BTCUSDT/ETHUSDT/SOLUSDT
  data/raw/trades/{asset}/*.csv.gz          (trade tape, gzipped CSV)
  data/raw/orderbook/{asset}/*.data.zip     (L2 order book, 500-level snapshots)
          │
          ▼
┌─────────────────────────────┐
│  src/data/loader.py         │  load_trades()
│  src/data/pipeline.py       │  process_all()
│                             │
│  - parse Unix timestamps    │
│  - map Buy/Sell → ±1        │
│  - detect and log gaps      │
│  - segment into regime weeks│
└─────────────┬───────────────┘
              │
              ▼
  data/processed/{asset}/{week}.parquet
  (cleaned trade tape: timestamp, price, qty, sign)
              │
              ├─────────────────────────────────────────────┐
              ▼                                             ▼
┌─────────────────────────────┐             ┌──────────────────────────┐
│  src/features/              │             │  src/models/vpin.py      │
│    reconstructor.py         │             │                          │
│                             │             │  compute_vpin()          │
│  reconstruct_and_extract_   │             │  build_volume_bucket()   │
│    from_state()             │             │  compute_rolling_vpins() │
│  process_week()             │             │                          │
│                             │             │  - volume-clock buckets  │
│  - replay L2 book messages  │             │  - 50-bucket rolling     │
│  - sample book state at     │             │    imbalance             │
│    each trade arrival       │             └──────────┬───────────────┘
│    (Option B)               │                        │
│  - spread, microprice, mid  │                        │
│  - depth_imbalance_{1,5,    │                        │
│    10,25}, pressure_        │                        │
│    imbalance                │                        │
└─────────────┬───────────────┘                        │
              │                                         │
              ▼                                         │
  data/processed/features/                             │
    {asset}_{date}_book_features.parquet               │
              │                                         │
              ▼                                         │
┌─────────────────────────────┐                        │
│  src/features/              │◄───────────────────────┘
│    build_features.py        │
│                             │
│  build_full_features()      │
│  add_trade_features()       │
│  add_toxicity_label()       │
│  add_vpin_feature()         │
│                             │
│  - merge book + trade data  │
│  - trade intensity (1/5/10s)│
│  - volume acceleration      │
│  - signed vol imbalance     │
│  - forward price label      │
│    (8 bps over 10s horizon) │
│  - join VPIN via forward    │
│    fill                     │
└─────────────┬───────────────┘
              │
              ▼
  data/processed/features/
    {asset}_{week}_full_features.parquet
    (16 features + toxic label, ~1M rows per asset-week)
              │
              ▼
┌─────────────────────────────┐
│  src/models/classifier/     │
│    data_loader.py           │
│    classifier.py            │
│                             │
│  ToxicityClassifier         │
│  VPINBaseline               │
│  prepare_split()            │
│  subsample_stratified()     │
│                             │
│  - walk-forward splits      │
│    (w1→w2, w1+w2→w3)        │
│  - 500k stratified subsample│
│  - StandardScaler for LR    │
│  - logistic regression      │
│  - CatBoost (500 trees,     │
│    depth 6)                 │
│  - VPIN threshold baseline  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  src/evaluations/           │
│    save_predictions.py      │
│                             │
│  - serialize logreg, scaler,│
│    GBT via joblib           │
│  - write per-asset week 3   │
│    predictions parquet      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  notebooks/                 │
│    03_classifier_evaluation │
│    04_backtest_results      │
│    05_backtest_results_with_│
│      infinite_time_horizon  │
│                             │
│  - AP / AUC / Brier scores  │
│  - bootstrap 95% CI         │
│  - Avellaneda-Stoikov MM    │
│    with continuous spread   │
│    widening ∝ p_toxic       │
│  - Sharpe, MtM PnL          │
└─────────────────────────────┘
```

---

## Design Decisions

- **Parquet for intermediate storage.** Each pipeline stage (cleaning, book feature extraction, full feature assembly) writes parquet to `data/processed/`. The serialisation cost is low relative to the L2 replay time (seconds per day of book data), and it lets any downstream stage be re-run independently without reprocessing everything upstream. The alternative — holding everything in memory across the pipeline — was ruled out early because L2 replay across three assets and three weeks exceeds available RAM.

- **Walk-forward validation instead of k-fold.** The three regime weeks (consolidation Sep 2024, breakout Oct/Nov 2024, stress Feb/Mar 2025) are structurally different: toxic rates range from 5–7% in normal regimes to 14–23% in the stress week. K-fold mixes future information into training folds and averages across regimes that have fundamentally different label distributions. Walk-forward splits — train on week 1, test on week 2; train on weeks 1+2, test on week 3 — respect temporal ordering and measure genuine out-of-distribution generalisation, which is what actually matters for a live system.

- **Logistic regression as the production model, not CatBoost.** CatBoost outperforms logistic regression on in-distribution test sets (weeks 1→2), but the relationship reverses on the stress week: logistic regression achieves AP 0.294 versus CatBoost's AP 0.244 (bootstrap 95% CI: LR [0.243, 0.256] vs VPIN [0.171, 0.183]). CatBoost learns regime-specific non-linearities that do not transfer when the market structure shifts. Logistic regression's linear decision boundary, combined with standardised inputs, generalises more robustly under distribution shift. Given that a market-making system needs to remain functional precisely during stress regimes — when spreads matter most — OOD performance is the correct selection criterion.

- **Volume clock (VPIN) instead of calendar time.** Information arrives with trades, not with the clock. Sampling at fixed time intervals overweights quiet periods and underweights bursts of informed activity. The VPIN framework (Easley et al. 2012) accumulates volume into fixed-size buckets — here, 1/50 of daily volume — and measures directional imbalance per bucket. This produces a signal that is inherently calibrated to activity level. The key empirical finding is that VPIN alone achieves AUC 0.506 (effectively random) across all test periods: it fails hardest in the stress regime because simultaneous aggressive selling and opportunistic buying produces two-sided high volume, suppressing net imbalance even as volatility spikes. VPIN detects directional informed flow but cannot distinguish it from volatility-driven two-sided flow.

- **Option B: book state at trade arrival, not fixed-interval snapshots.** The feature matrix is constructed by replaying the L2 order book message stream and sampling spread, microprice, and depth imbalance at the moment each trade arrives, rather than at fixed time intervals. This preserves the causal structure — the book state the market maker observed before deciding to quote. Fixed-interval snapshots alias trades between sample points and introduce look-ahead at the microsecond level when trades arrive after the sample timestamp but are attributed to the preceding interval. Option B also avoids the sample frequency selection problem: trade arrivals are the natural event clock.

---

## What a Production System Would Add

The most fundamental gap is queue position. This codebase operates on the public trade tape, which records only completed trades; it has no information about where the market maker's resting orders sit in the queue relative to other participants. Capturing true queue position requires co-location with nanosecond-resolution timestamps and direct feed access, which changes both the latency budget and the data infrastructure entirely. Order-to-trade ratios would also need to be incorporated: a trader submitting hundreds of orders per execution is engaged in layering or quote stuffing, which looks very different from an informed trader who sends a single aggressive order — the trade tape alone cannot distinguish these, but a full message feed can. Named counterparty data would unlock the most powerful signal: the Cartea and Sánchez-Betancourt (2025) framework for adverse selection explicitly conditions on counterparty identity, decomposing flow into informed, uninformed, and strategic components. Crypto perpetual markets do not expose counterparty identity, so that decomposition is structurally unavailable here; a production venue or prime-brokerage relationship with a centralised clearing counterparty could provide this. Cross-venue aggregation would further sharpen the signal — informed traders often establish positions across spot, perpetuals, and options simultaneously, and a signal that sees only the perpetual leg is observing a projection of a multi-venue strategy. Finally, the backtest in this project uses a simplified spread-widening model proportional to `p_toxic`; a production implementation would replace this with real transaction cost modelling that accounts for rebate schedules, maker-taker asymmetries, and the inventory risk term in the Avellaneda-Stoikov objective, making the spread-widening tradeoff economically precise rather than proportional.
