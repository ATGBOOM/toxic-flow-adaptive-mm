# Toxicity Classifier — Design Document (Session 8)

## Objective

Build a classifier that predicts whether an incoming trade is toxic (adverse price move > 8 bps within 10 seconds), using limit order book and trade features. The classifier must demonstrably outperform a VPIN-only baseline and generalise across market regimes.

---

## Data

### Source
Feature parquets from Session 7: one file per asset-week, ~10-24M rows each, 23 columns.

### Assets (pooled into single model)
- BTCUSDT, ETHUSDT, SOLUSDT

### Regime Weeks
| Week | Dates | Regime | Toxic Rate (BTC) |
|------|-------|--------|-----------------|
| 1 | Sep 9-15 2024 | Consolidation | 5.2% |
| 2 | Oct 28 - Nov 3 2024 | Breakout | 4.2% |
| 3 | Feb 24 - Mar 2 2025 | Stress/correction | 14.1% |

### Train/Test Splits
- **Split 1:** Train on week 1 (all assets) → Test on week 2 and week 3 separately
- **Split 2:** Train on weeks 1+2 (all assets) → Test on week 3

Comparing Split 1 vs Split 2 on week 3 answers: does seeing a breakout regime in training help predict a crash?

### Subsampling
- Training: 500k rows, stratified random (preserves toxic/non-toxic ratio)
- Testing: full dataset (no subsampling)
- Rationale: 500k provides ~25k toxic examples at 5% base rate — sufficient for any classifier. Full test set gives precise evaluation metrics.

---

## Feature Set (16 features)

### Engineered at load time (not in original parquets)
| Feature | Derivation | Rationale |
|---------|-----------|-----------|
| `spread_bps` | spread / midprice * 10000 | Normalises spread across assets and price levels for cross-asset pooling |
| `microprice_minus_mid` | microprice - midprice | Captures book asymmetry without encoding raw price level |
| `qty_normalised` | qty / rolling mean of recent qty | "Is this trade unusually large?" — comparable across assets unlike raw qty |

### From parquet directly
| Feature | What it captures |
|---------|-----------------|
| `depth_imbalance_1` | Book asymmetry at best bid/ask |
| `depth_imbalance_5` | Book asymmetry at top 5 levels |
| `depth_imbalance_10` | Book asymmetry at top 10 levels |
| `depth_imbalance_25` | Book asymmetry at full visible book |
| `bid_pressure` | Total bid-side depth |
| `ask_pressure` | Total ask-side depth |
| `pressure_imbalance` | Bid vs ask total depth ratio |
| `trade_intensity_1s` | Number of trades in last 1 second |
| `trade_intensity_5s` | Number of trades in last 5 seconds |
| `trade_intensity_10s` | Number of trades in last 10 seconds |
| `volume_acceleration` | Change in trade intensity |
| `signed_vol_imbalance_10s` | Net buy vs sell volume in last 10 seconds |
| `vpin` | Volume-synchronised probability of informed trading |

### Excluded columns and why
| Column | Reason for exclusion |
|--------|---------------------|
| `timestamp`, `ts_ms` | Calendar time shouldn't drive predictions — we want microstructure signals |
| `price` | Raw price level is meaningless across assets and time periods |
| `qty` | Replaced by `qty_normalised` — raw qty conflates asset identity with trade size |
| `sign` | Excluded to enforce direction symmetry — toxicity should be symmetric in theory |
| `spread` | Replaced by `spread_bps` for cross-asset comparability |
| `microprice`, `midprice` | Replaced by their difference — avoids encoding raw price level |
| `fwd_10s_bps` | **LEAKAGE** — this is the forward-looking return used to compute the label |
| `toxic` | This is the label, not a feature |

### Asset indicator
- Run both with and without an `asset_id` feature (0/1/2 for BTC/ETH/SOL)
- If model performs well without: features capture universal microstructure patterns (stronger result)
- If adding asset_id helps significantly: toxicity is partially asset-specific (still a valid finding)

---

## Models

### 1. VPIN-only baseline
- Threshold classifier: predict toxic if VPIN > threshold
- Sweep thresholds to generate full precision-recall curve
- Purpose: establish the bar the ML models must beat
- This is the baseline from Easley et al. (2012), our simplest toxicity measure

### 2. Logistic regression
- Simplest ML baseline, interpretable coefficients
- Standardise features (zero mean, unit variance) before fitting
- No regularisation initially; add L2 if overfitting observed
- Purpose: check if a linear decision boundary suffices
- Following Cartea & Sánchez-Betancourt (2023) benchmark methodology

### 3. Gradient boosted trees (CatBoost)
- Main classifier — handles nonlinear feature interactions, robust to scale
- Walk-forward: hyperparameters tuned only within training window
- Key hyperparameters: learning rate, depth, iterations, l2_leaf_reg
- Purpose: best achievable performance with standard ML on these features

---

## Evaluation Plan (Session 9)

### Metrics
- **Precision-recall curves** with economic interpretation at key thresholds
- **Calibration:** reliability diagrams — are 70% predictions actually toxic 70% of the time?
- **Brier score:** overall probability calibration quality
- **SHAP values:** feature importance connected to microstructure theory
- **Statistical significance:** paired tests comparing classifier vs VPIN baseline

### Economic framing
At each operating threshold, translate to:
- "Detects X% of toxic trades (recall) while generating Y% false positives"
- "A market maker using this signal avoids Z% of adverse fills at cost of missing W% of profitable trades"

### Regime comparison
- Compare classifier performance on week 2 (breakout) vs week 3 (stress)
- If Split 2 outperforms Split 1 on week 3 → seeing more regimes in training helps
- If both splits fail on week 3 → stress regime is genuinely out-of-distribution

---

## Known Concerns

1. **Class imbalance:** ~5% toxic in training (weeks 1-2). Models may default to "predict non-toxic." Mitigation: evaluate on precision-recall, not accuracy.
2. **VPIN NaN warmup:** ~15% of rows have NaN VPIN. Strategy: drop these rows — they're the first ~1.5M trades before the volume clock fills, not informative.
3. **Distribution shift:** Week 3 toxic rate (14-23%) is 3-4x higher than training (4-7%). The classifier has never seen this base rate. This is the central challenge.
4. **Missing week 3 data:** ETH missing Feb 28, Mar 1-2 orderbook; SOL missing Feb 24. Not critical — enough data remains.
5. **qty normalisation:** Rolling window size for "recent average qty" needs to be chosen. Will use 1000-trade rolling mean — long enough to be stable, short enough to adapt.