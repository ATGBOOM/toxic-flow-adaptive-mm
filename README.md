# Toxic Order Flow Detection & Adaptive Market-Making

## Purpose
Implementing VPIN-based toxic order flow detection on crypto LOB data, extended with a 
Bayesian classifier and adaptive market-making strategy. Framed as a methodological test 
of whether equity microstructure patterns transfer to crypto venues. Not a trading bot.

## Papers
- Easley, López de Prado, O'Hara (2012) — VPIN
- Avellaneda & Stoikov (2008) — optimal market making
- Cartea & Sánchez-Betancourt (2025) — analytical toxic flow adjustment
- Cartea & Sánchez-Betancourt (2023) — PULSE feature set and evaluation framework
- Bieganowski (2026) — crypto microstructure feature engineering

## Data

### Trade Data
- **Source:** Bybit public data portal, USDT perpetual contracts
- **Assets:** BTCUSDT, ETHUSDT, SOLUSDT
- **Rationale for perps over spot:** Price discovery in crypto occurs primarily on the 
  perpetual futures market. Informed traders concentrate here due to capital efficiency 
  (leverage) and liquidity. Perp prices track spot within a few basis points via the 
  funding rate mechanism.

### Order Book Data
- **Source:** Bybit public data portal, USDT perpetual contracts (same venue as trades)
- **Format:** JSON lines, incremental delta updates with periodic snapshots, 500 levels deep
- **Reconstruction required:** Files contain ~2 snapshots and ~860k deltas per day. 
  Full order book reconstruction (apply deltas sequentially onto initial snapshot) is 
  needed to recover book state at any point in time. Implemented in Session 7.

### Regime Weeks
Three deliberately chosen weeks across distinct market conditions:

| Week | Dates | Regime | BTC Price |
|------|-------|--------|-----------|
| 1 | Sep 9-15 2024 | Low volatility consolidation | ~$55k |
| 2 | Oct 28 - Nov 3 2024 | Directional breakout | above $70k |
| 3 | Feb 24 - Mar 2 2025 | Stress/correction | from $100k+ highs |

**Why three regimes:** To test whether toxicity patterns and classifier performance are 
regime-dependent, rather than fitting to one market condition.

**Sampling bias caveat:** These weeks were selected for regime variation, not randomly. 
Findings should not be generalised as representative of typical market conditions.

## Pipeline

### Trade Data
Raw CSV.gz → typed DataFrame (Unix timestamp in seconds, direction as +1/-1 sign, 
gap detection) → parquet. One file per asset per week.

### Order Book Data
Raw JSON lines → order book reconstruction (snapshot initialisation + sequential delta 
application) → feature extraction at each trade timestamp → parquet. Book state is 
carried across midnight boundaries for continuity. Processing time: ~100s per asset-day.

### Feature Matrix
Book features + trade features + VPIN + toxicity label merged into a single parquet 
per asset-week. One row per trade, ~20 feature columns plus label.

## EDA Findings (Session 5)

### 1. Sampling frequency dominates distributional statistics
Tick-level return statistics are dominated by microstructure artefacts. Kurtosis drops 
from ~10,000 at tick level to 5-10 at 1-minute bars — the correct range for analysis. 
This directly motivates the volume clock in VPIN: time-based sampling produces statistics 
that reflect trade arrival rates more than genuine price dynamics.

### 2. OFI signal decays at dramatically different rates across regimes

| | Raw OFI 10s | Raw OFI 1min | Decay ratio |
|--|--|--|--|
| Week 1 (consolidation) | 0.054 | 0.001 | 54x |
| Week 2 (breakout) | 0.045 | 0.006 | 8x |
| Week 3 (stress) | 0.017 | 0.015 | 1x |

Week 3 shows persistent OFI — the signal survives temporal aggregation, consistent with 
sustained directional selling pressure. Week 1 shows fast decay, consistent with noise 
trading that rapidly reverses.

### 3. Normalisation sharpens signal at short frequencies, hurts at long
Volume-normalised OFI outperforms raw OFI at 10-second bars. At 1-minute bars 
normalisation adds noise. All correlations are small (max 0.085) — OFI alone is a weak 
predictor, motivating the richer feature set in Phase 3.

## VPIN Results (Session 6)

### Implementation
VPIN implemented from scratch following Easley et al. (2012). Bucket size = 1/50 of 
daily volume (~2376 BTC for BTC week 1), producing ~300 buckets per week. Trade 
splitting implemented correctly for trades straddling bucket boundaries.

### Cross-Regime Results (fixed bucket size = 2376 BTC)

| Asset | Week 1 (consolidation) | Week 2 (breakout) | Week 3 (correction) |
|-------|----------------------|-------------------|---------------------|
| BTCUSDT | 0.162 | 0.171 | 0.149 |
| ETHUSDT | 0.175 | 0.194 | 0.186 |
| SOLUSDT | 0.164 | 0.166 | 0.155 |

### Key Finding — VPIN Fails in Volatility Regimes
Week 3 (the most volatile regime) has the lowest mean VPIN for BTC and SOL. In a 
correction, aggressive sellers and opportunistic buyers are simultaneously active — 
high volume on both sides produces low net imbalance. VPIN detects directional informed 
trading well but fails when both sides react simultaneously. This is a genuine limitation 
of the metric, not a data artefact.

### Validation
Correlation between VPIN and absolute 30-minute forward return (BTC week 2): **0.145**. 
This is the baseline benchmark the classifier must exceed.

### VPIN Limitations
1. Bucket size sensitivity — choice of 1/50 daily volume is principled but arbitrary
2. Cross-asset comparison unreliable due to different volume scales
3. Fails to detect toxicity in two-sided volatile regimes
4. Lagging signal by construction — summarises past n buckets

## Feature Engineering & Toxicity Labelling (Session 7)

### Order Book Reconstruction
Bybit ob500 files contain ~2 snapshots and ~863k delta updates per day per asset. 
Reconstruction streams through the file maintaining a running book state (two 
dictionaries: price → size for bids and asks). For each delta, levels with size > 0 
are updated; levels with size = 0 are deleted. Book state carries across midnight 
boundaries between consecutive days.

Features are extracted at each trade timestamp (Option B sampling — "what did the book 
look like when this trade arrived?"), producing one feature row per trade. This ensures 
perfect alignment between features and the prediction target.

### Book-Derived Features
- **Spread:** best ask − best bid. Median = $0.10 (minimum tick) for BTC, max = $31.70 
  during momentary dislocations. BTC sits at the tightest possible spread most of the time.
- **Microprice:** size-weighted midpoint = (best_bid × ask_size + best_ask × bid_size) / 
  (bid_size + ask_size). Better estimate of fair value than simple midpoint.
- **Depth imbalance at levels 1, 5, 10, 25:** 
  (bid_volume_top_N − ask_volume_top_N) / (bid_volume_top_N + ask_volume_top_N). 
  Ranges from −1 (all volume on ask side) to +1 (all volume on bid side).
- **Bid/ask pressure:** volume concentration in top 5 levels relative to top 25. 
  High pressure = volume concentrated near best price, ready to absorb incoming trades.
- **Pressure imbalance:** bid_pressure − ask_pressure.

### Trade-Derived Features
- **Trade intensity (1s, 5s, 10s windows):** rolling count of trades using searchsorted 
  on timestamp arrays. No loops — O(n log n) via binary search.
- **Volume acceleration:** volume in last 5s / (volume in last 30s × 5/30). Values > 1.0 
  mean trading is speeding up. Computed via cumulative sums + searchsorted.
- **Signed volume imbalance (10s):** sum of sign × qty in last 10 seconds. Measures net 
  directional pressure. Computed via cumulative sums of signed volume.
- **VPIN:** forward-filled from Session 6 bucket-level computation onto trade timestamps 
  using searchsorted. ~1.6M trades per week in warmup period before first VPIN value.

### Toxicity Definition

**Label:** A trade is toxic if the price moves adversely by more than 8 bps within 10 
seconds in the direction of the trade. Buy is toxic if price rises > 8 bps; sell is 
toxic if price falls > 8 bps.

**Parameter selection (data-driven):**
- **Horizon (Y = 10 seconds):** Long enough for informed trades to show impact, short 
  enough to measure the trade's effect rather than background drift. Consistent with 
  Cartea & Sánchez-Betancourt (2023).
- **Threshold (X = 8 bps):** Chosen from empirical distribution of absolute 10-second 
  forward returns on BTC Sep 9:

| Percentile | Absolute 10s forward return (bps) |
|------------|-----------------------------------|
| Median | 2.42 |
| 75th | 4.79 |
| 90th | 8.09 |
| 95th | 11.11 |
| 99th | 30.50 |

8 bps ≈ 90th percentile of absolute moves. Directional filtering (requiring the move 
to match trade sign) reduces the label rate from ~10% to ~5.5%, as expected.

**Tradeoff reasoning:** Threshold too small → flags normal market noise, hurting precision. 
Threshold too large → misses real toxic trades, hurting recall. 90th percentile balances 
selectivity with sufficient positive examples for classifier training.

### Toxic Rate by Regime — Key Finding

| Asset | Week 1 (consolidation) | Week 2 (breakout) | Week 3 (stress) |
|-------|----------------------|-------------------|-----------------|
| BTCUSDT | 5.2% | 4.2% | **14.1%** |
| ETHUSDT | 7.0% | 5.7% | **16.8%** |
| SOLUSDT | 6.7% | 6.9% | **23.4%** |

**Finding 1 — Stress regimes have 3-4x higher toxic rates.** During corrections, informed 
traders (or faster-reacting traders) dominate flow. This is consistent with microstructure 
theory: adverse selection intensifies during high-uncertainty periods.

**Finding 2 — Less liquid assets are more vulnerable.** SOL at 23.4% toxic rate during 
stress vs BTC at 14.1%. Less uninformed flow to dilute the informed signal.

**Finding 3 — VPIN contradicts toxic rate in stress regimes.** VPIN was lowest in week 3 
(Session 6) but toxic rate was highest. VPIN measures net imbalance; toxicity measures 
adverse price impact. In a crash, both sides trade aggressively (low VPIN) but the 
informed side consistently wins (high toxicity). This is the strongest argument for a 
richer classifier over VPIN alone.

### Toxic Trade Clustering
99.2% of toxic trades are within 1 second of another toxic trade. Median gap = 0.0s. 
Toxic trades arrive in rapid bursts — consistent with informed agents executing 
aggressively over short windows. Implication: the classifier is really predicting 
"toxic episodes" rather than individual toxic trades.

### Feature-Toxicity Signal Strength (BTC Sep 9)

| Feature | Toxic mean | Non-toxic mean | Ratio |
|---------|-----------|---------------|-------|
| Signed vol imbalance 10s | 150.0 | 21.6 | **6.9x** |
| Depth imbalance L1 | 0.045 | 0.012 | **3.7x** |
| Trade intensity 1s | 1044 | 316 | **3.3x** |
| Spread | 0.593 | 0.246 | **2.4x** |
| Volume acceleration | 1.98 | 1.69 | 1.2x |
| Pressure imbalance | 0.000 | 0.006 | ~0x |

**Strongest signals:** Signed volume imbalance, trade intensity, and depth imbalance — 
all trade-flow features rather than deep book structure. Spread is useful (market makers 
already widen during toxic episodes). Volume acceleration and pressure imbalance show 
minimal signal.

### Missing Data
- ETH week 3: missing Feb 28, Mar 1-2 orderbook files (4 of 7 days available)
- SOL week 3: missing Feb 24 orderbook file (6 of 7 days available)
- Not critical for classifier training — sufficient data across other asset-weeks.

## Classifier Results (Session 8)

### Design Decisions
See `docs/CLASSIFIER_DESIGN.md` for full rationale on each decision.

**Feature engineering at load time (3 new features):**
- `spread_bps` = spread / midprice × 10000 (cross-asset comparable)
- `microprice_minus_mid` = microprice − midprice (book asymmetry without raw price level)
- `qty_normalised` = qty / rolling 1000-trade mean (relative trade size, cross-asset comparable)

**Excluded columns:**
- `sign` — excluded to enforce direction symmetry; toxicity should be symmetric in theory
- `price`, `midprice`, `microprice` — raw price levels meaningless across assets/time
- `fwd_10s_bps` — forward-looking return used to compute label; including it is pure leakage
- `qty` — replaced by normalised version; raw qty conflates asset identity with trade size

**Final feature set (16 features):** spread_bps, microprice_minus_mid, qty_normalised, 
depth_imbalance_1/5/10/25, bid_pressure, ask_pressure, pressure_imbalance, 
trade_intensity_1s/5s/10s, volume_acceleration, signed_vol_imbalance_10s, vpin.

**Training splits:**
- Split 1: Train week 1 → test weeks 2 and 3 separately
- Split 2: Train weeks 1+2 → test week 3
- Subsampling: 500k stratified random from training weeks, full test sets

**Models:** VPIN-only baseline (threshold classifier), logistic regression, CatBoost (gradient boosted trees)

### Results

#### Split 1: Train week 1 → Test week 2 (similar regime)

| Model | AP | AUC | Brier |
|-------|------|------|-------|
| VPIN baseline | 0.055 | 0.506 | 0.065 |
| Logistic regression | 0.070 | 0.593 | 0.052 |
| CatBoost (balanced) | 0.095 | 0.673 | 0.144 |

#### Split 1: Train week 1 → Test week 3 (stress regime, out-of-distribution)

| Model | AP | AUC | Brier |
|-------|------|------|-------|
| VPIN baseline | 0.177 | 0.507 | 0.148 |
| Logistic regression | 0.281 | 0.650 | 0.155 |
| CatBoost (balanced) | 0.242 | 0.616 | 0.176 |

#### Split 2: Train weeks 1+2 → Test week 3

| Model | AP | AUC | Brier |
|-------|------|------|-------|
| VPIN baseline | 0.177 | 0.507 | 0.148 |
| Logistic regression | 0.294 | 0.664 | 0.155 |
| CatBoost (balanced) | 0.244 | 0.600 | 0.266 |

### Key Findings

**Finding 1 — VPIN is no better than random as a standalone classifier.** AUC of 0.506 
across all tests (0.5 = random). AP matches base rate exactly. VPIN cannot rank which 
trades are toxic. This quantitatively confirms the Session 6 observation that VPIN fails 
in two-sided volatile regimes.

**Finding 2 — Logistic regression outperforms CatBoost on out-of-distribution data.** 
On week 2 (similar regime to training), CatBoost wins: AP 0.095 vs 0.070. On week 3 
(stress, OOD), logistic regression wins: AP 0.281 vs 0.242. CatBoost overfits to 
training-regime patterns that don't transfer. Classic bias-variance tradeoff in action.

**Finding 3 — CatBoost is badly miscalibrated.** Brier scores of 0.144-0.266 compared 
to logreg's 0.052-0.155. The `auto_class_weights='Balanced'` parameter inflates 
predicted probabilities. The model's rankings may be reasonable but its probability 
estimates are not trustworthy.

**Finding 4 — Adding breakout data to training marginally helps.** Split 2 logreg 
(trained on weeks 1+2) gets AP 0.294 vs Split 1's 0.281 on week 3. CatBoost barely 
changes (0.244 vs 0.242). More regime diversity helps the simple model slightly.

**Finding 5 — Trade intensity dominates feature importance.** Logistic regression 
coefficients (standardised):
- trade_intensity_10s: +0.48 (strongest by far)
- trade_intensity_5s: −0.36 (together with 10s, captures burst shape)
- volume_acceleration: +0.18
- ask/bid_pressure: −0.13/−0.10 (thin books → more toxic)
- vpin: +0.04 (barely matters once other features are present)

### Open Questions (to investigate)
1. Is CatBoost underperformance due to balanced class weights inflating probabilities?
2. Would 2M training rows help CatBoost learn better patterns?
3. Does adding an asset indicator feature improve cross-asset pooling?
4. What do precision-recall tradeoffs look like at operationally useful thresholds?
5. Feature leakage check: confirmed clean — all features are backward-looking.

## Status
- [x] Phase 0: Prerequisites (Sessions 1-3)
- [x] Phase 1: Data pipeline (Session 4)
- [x] Phase 2: EDA (Session 5)
- [x] Phase 3: VPIN implementation (Session 6)
- [x] Phase 4: Feature engineering + LOB reconstruction (Session 7)
- [x] Phase 5a: Toxicity classifier — initial models (Session 8) ← current
- [ ] Phase 5b: Toxicity classifier — investigation & evaluation (Session 8 continued)
- [ ] Phase 5c: Rigorous evaluation (Session 9)
- [ ] Phase 6: Adaptive market-making (Sessions 10-11)
- [ ] Phase 7: Writeup and packaging (Sessions 12-13)