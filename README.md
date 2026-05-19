# Toxic Order Flow Detection & Adaptive Market-Making

**Research question:** Do equity microstructure tools — specifically VPIN-based toxic flow detection — transfer to anonymous, high-frequency cryptocurrency perpetual futures markets?

**Key finding:** VPIN fails as a standalone toxicity detector at standard equity-market parameters (AUC 0.506, indistinguishable from random at V=1/50, n=50), with partial recovery at shorter rolling windows (AUC up to 0.615 at n=20) consistent with the structural mechanism: two-sided aggressive trading during corrections suppresses net order imbalance even when adverse selection is at its peak. A logistic classifier on trade intensity features achieves AP 0.294 vs VPIN baseline 0.177 on the stress regime (bootstrap 95% CI: [0.287, 0.304] vs [0.171, 0.183]). An adaptive Avellaneda-Stoikov market maker using the classifier signal shows directionally consistent MtM improvement across all six out-of-sample asset-week combinations, with the only statistically distinguishable result in the highest-toxicity cell (ETH stress: 95% CI [+$78, +$178]).

---

## Overview

This project implements and critically evaluates a pipeline of equity market microstructure tools on cryptocurrency perpetual futures data from Bybit, across three assets (BTCUSDT, ETHUSDT, SOLUSDT) and three deliberately chosen market regimes. The framing is methodological — testing whether patterns documented in equity markets transfer to anonymous, high-frequency crypto venues — not a trading system.

The pipeline covers: VPIN implementation from scratch following Easley et al. (2012), a supervised toxicity classifier with rigorous walk-forward evaluation, and an adaptive market-making strategy based on Avellaneda & Stoikov (2008) that uses the classifier signal to adjust quotes dynamically. Each component is evaluated against an analytical or statistical baseline, and failure modes are documented as carefully as successes.

## Papers

- Easley, López de Prado, O'Hara (2012) — VPIN: volume-synchronised probability of informed trading
- Avellaneda & Stoikov (2008) — optimal market making in a limit order book
- Cartea & Sánchez-Betancourt (2025, arXiv:2503.18005) — analytical dealing strategy for toxic flow
- Cartea & Sánchez-Betancourt (2023, arXiv:2312.05827) — PULSE: online Bayesian toxicity detection
- Bieganowski (2026, arXiv:2602.00776) — explainable patterns in cryptocurrency microstructure
- Muhle-Karbe, Ouazzani Chahdi, Rosenbaum & Szymanski (arXiv:2601.23172) — Hawkes process decomposition of informed flow

## Repository Structure

```
toxic-flow-adaptive-mm/
├── README.md
├── ARCHITECTURE.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_vpin_analysis.ipynb
│   ├── 02_vpin_robustness.ipynb
│   ├── 03_classifier_evaluation.ipynb
│   └── 04_backtest_results.ipynb
├── src/
│   ├── data/                  # Data loading and pipeline
│   ├── features/              # Feature engineering and LOB reconstruction
│   ├── models/                # VPIN, classifier, market maker
│   ├── backtest/              # Backtesting engine
│   └── evaluations/           # Statistical evaluation tools
├── configs/
├── tests/
└── results/
```

## Data

### Source and Assets
Bybit public data portal, USDT perpetual contracts. Assets: BTCUSDT, ETHUSDT, SOLUSDT.

Perpetual futures rather than spot: price discovery in crypto occurs primarily on the perpetual futures market. Informed traders concentrate here due to capital efficiency (leverage) and liquidity. Perp prices track spot within a few basis points via the funding rate mechanism.

### Order Book Data
Bybit ob500 files: incremental delta updates with periodic snapshots, 500 levels deep, approximately 2 snapshots and 863k deltas per day per asset. Full reconstruction required — deltas are applied sequentially onto the initial snapshot, with book state carried across midnight boundaries. Features are extracted at each trade timestamp, ensuring alignment between book state and the prediction target.

### Regime Weeks

Three weeks selected for regime variation rather than randomly. Findings should not be generalised as representative of typical market conditions.

| Week | Dates | Regime | BTC Price |
|------|-------|--------|-----------|
| 1 | Sep 9–15 2024 | Low volatility consolidation | ~$55k |
| 2 | Oct 28 – Nov 3 2024 | Directional breakout (post US election) | above $70k |
| 3 | Feb 24 – Mar 2 2025 | Stress / correction | from $100k+ highs |

**Missing data:** ETH week 3 has orderbook files for 4 of 7 days (Feb 28, Mar 1–2 unavailable). SOL week 3 is missing Feb 24. Results for truncated weeks are flagged throughout.

---

## EDA Findings

### Sampling frequency dominates distributional statistics
Tick-level return statistics are dominated by microstructure artefacts. Kurtosis drops from ~10,000 at tick level to 5–10 at 1-minute bars. This directly motivates the volume clock in VPIN: time-based sampling produces statistics that reflect trade arrival rates more than genuine price dynamics.

### OFI signal decays at dramatically different rates across regimes

| | Raw OFI 10s | Raw OFI 1min | Decay ratio |
|--|--|--|--|
| Week 1 (consolidation) | 0.054 | 0.001 | 54x |
| Week 2 (breakout) | 0.045 | 0.006 | 8x |
| Week 3 (stress) | 0.017 | 0.015 | 1x |

Week 3 shows persistent OFI — the signal survives temporal aggregation, consistent with sustained directional selling pressure. Week 1 shows fast decay consistent with noise trading. All correlations are small (max 0.085), motivating the richer feature set in the classifier.

---

## VPIN

### Implementation
VPIN implemented from scratch following Easley et al. (2012). Bucket size = 1/50 of daily volume (~2,376 BTC for BTC week 1), producing ~300 buckets per week. Trade splitting is implemented correctly for trades straddling bucket boundaries.

### Cross-Regime Results

| Asset | Week 1 (consolidation) | Week 2 (breakout) | Week 3 (stress) |
|-------|----------------------|-------------------|-----------------|
| BTCUSDT | 0.162 | 0.171 | 0.149 |
| ETHUSDT | 0.175 | 0.194 | 0.186 |
| SOLUSDT | 0.164 | 0.166 | 0.155 |

Mean VPIN is lowest in the stress regime for BTC and SOL — the opposite of what a toxicity metric should show. AUC against the toxic label at standard parameters (V=1/50, n=50): 0.506 across all test periods, indistinguishable from random. Correlation between VPIN and absolute 30-minute forward return on BTC week 2: 0.145 — the baseline the classifier must exceed.

### Why VPIN Fails in Stress Regimes

During a correction, informed sellers and opportunistic buyers are simultaneously active. High volume on both sides produces low net order imbalance even when adverse selection is at its peak. VPIN measures directional imbalance — it detects informed trading that is one-sided. In a crash, it is blind to the two-sided informed activity that characterises the regime.

This is not a parameter artefact. VPIN was re-run across a grid of bucket sizes V ∈ {1/100, 1/50, 1/25, 1/10} of average daily volume and rolling windows n ∈ {20, 50, 100, 250} (144 evaluations total):

| Regime | AUC range (valid cells) | Mean AUC |
|--------|------------------------|----------|
| Consolidation | [0.465, 0.570] | 0.519 |
| Breakout | [0.427, 0.598] | 0.499 |
| Stress | [0.470, 0.614] | 0.553 |

At standard parameters (V=1/50, n=50) the failure is consistent across all assets and regimes. At small rolling windows (n=20) VPIN partially recovers signal in the stress regime — BTC stress reaches AUC 0.615 at (V=1/50, n=20), SOL stress reaches 0.614 at (V=1/10, n=20). The mechanism is that a shorter window is less likely to average together the opposing flow that cancels the signal. The classifier's advantage over VPIN holds at the parameter settings used in the equity microstructure literature, which is the relevant comparison.

†ETH week3 bucket sizes are derived from a 4-day average daily volume and are larger than a full-week figure would produce. Results for this cell should be interpreted with this caveat.

---

## Feature Engineering & Toxicity Labelling

### Toxicity Definition
A trade is toxic if the price moves adversely by more than 8 bps within 10 seconds in the direction of the trade. Buy is toxic if price rises >8 bps; sell is toxic if price falls >8 bps.

**Parameter justification:** 10 seconds is long enough for informed trades to show impact, short enough to measure the trade's effect rather than background drift — consistent with Cartea & Sánchez-Betancourt (2023). 8 bps ≈ 90th percentile of absolute 10-second forward returns on BTC (empirical distribution below). Directional filtering reduces the label rate from ~10% to ~5.5%.

| Percentile | Absolute 10s forward return (bps) |
|------------|-----------------------------------|
| Median | 2.42 |
| 75th | 4.79 |
| 90th | 8.09 |
| 95th | 11.11 |
| 99th | 30.50 |

### Toxic Rate by Regime

| Asset | Week 1 (consolidation) | Week 2 (breakout) | Week 3 (stress) |
|-------|----------------------|-------------------|-----------------|
| BTCUSDT | 5.2% | 4.2% | **14.1%** |
| ETHUSDT | 7.0% | 5.7% | **16.8%** |
| SOLUSDT | 6.7% | 6.9% | **23.4%** |

Stress regimes show 3–4x higher toxic rates. Less liquid assets are more vulnerable — SOL at 23.4% vs BTC at 14.1% during the correction. VPIN was lowest in week 3 but toxic rate was highest: the strongest argument for a richer classifier.

### Toxic Trade Clustering
99.2% of toxic trades arrive within 1 second of another toxic trade (median inter-arrival gap = 0.0s). This clustering is consistent with self-exciting point process dynamics: each toxic event increases the short-term arrival rate of subsequent toxic events, producing the burst structure observed in the data. This is the Hawkes process formalism for trade arrival in microstructure (Muhle-Karbe et al., arXiv:2601.23172, for the core/reaction flow decomposition).

An important identification caveat: self-excitation (a single informed agent's order splitting creating sequential toxic trades) and genuine co-ordinated informed activity (multiple independent agents reacting to the same signal) are observationally equivalent in the public Bybit feed, which provides no participant identifiers. The clustering observation is robust; its interpretation as multi-agent informed activity is not.

Implication for the classifier: it is predicting toxic episodes rather than individual toxic trades.

### Features

**Book-derived:** spread, microprice, depth imbalance at levels 1/5/10/25, bid/ask pressure, pressure imbalance.

**Trade-derived:** trade intensity at 1s/5s/10s windows (rolling count via searchsorted, O(n log n)), volume acceleration, signed volume imbalance over 10s, VPIN (forward-filled from bucket computation).

**Feature-toxicity signal strength (BTC consolidation week):**

| Feature | Toxic mean | Non-toxic mean | Ratio |
|---------|-----------|---------------|-------|
| Signed vol imbalance 10s | 150.0 | 21.6 | **6.9x** |
| Depth imbalance L1 | 0.045 | 0.012 | **3.7x** |
| Trade intensity 1s | 1044 | 316 | **3.3x** |
| Spread | 0.593 | 0.246 | **2.4x** |
| Volume acceleration | 1.98 | 1.69 | 1.2x |
| Pressure imbalance | 0.000 | 0.006 | ~0x |

Trade-flow features dominate. Deep book structure (pressure imbalance) shows minimal signal.

---

## Classifier

### Design

Three new features engineered at load time for cross-asset comparability: `spread_bps` (spread / midprice × 10000), `microprice_minus_mid` (book asymmetry without raw price level), `qty_normalised` (qty / rolling 1000-trade mean).

Excluded: `sign` (direction symmetry enforced), raw price levels (meaningless cross-asset), `fwd_10s_bps` (leakage — used to compute the label), raw `qty` (replaced by normalised version).

**Final feature set (16 features):** spread_bps, microprice_minus_mid, qty_normalised, depth_imbalance_1/5/10/25, bid_pressure, ask_pressure, pressure_imbalance, trade_intensity_1s/5s/10s, volume_acceleration, signed_vol_imbalance_10s, vpin.

**Walk-forward validation only.** Random splits are invalid for time series — they leak future microstructure state into training. Two splits: train week 1 → test weeks 2 and 3; train weeks 1+2 → test week 3. Subsampling: 500k stratified random from training weeks, full test sets.

### Results

#### Train week 1 → Test week 2 (similar regime)

| Model | AP | AUC | Brier |
|-------|------|------|-------|
| VPIN baseline | 0.055 | 0.506 | 0.065 |
| Logistic regression | 0.070 | 0.593 | 0.052 |
| CatBoost | 0.095 | 0.673 | 0.144 |

#### Train week 1 → Test week 3 (stress, out-of-distribution)

| Model | AP | AUC | Brier |
|-------|------|------|-------|
| VPIN baseline | 0.177 | 0.507 | 0.148 |
| Logistic regression | 0.281 | 0.650 | 0.155 |
| CatBoost | 0.242 | 0.616 | 0.176 |

#### Train weeks 1+2 → Test week 3

| Model | AP | AUC | Brier |
|-------|------|------|-------|
| VPIN baseline | 0.177 | 0.507 | 0.148 |
| Logistic regression | 0.294 | 0.664 | 0.155 |
| CatBoost | 0.244 | 0.600 | 0.266 |

### Key Findings

**VPIN is no better than random as a standalone classifier.** AUC 0.506 across all tests. SHAP analysis confirms this: VPIN mean SHAP = 0.008, near-zero once trade intensity and book features are included. The metric that fails as a standalone predictor is also redundant within a richer feature set.

**Logistic regression outperforms CatBoost on out-of-distribution data.** On week 2 (similar regime to training), CatBoost wins: AP 0.095 vs 0.070. On week 3 (stress, OOD), logistic regression wins: AP 0.281 vs 0.242. CatBoost overfits to calm-regime patterns that do not transfer — classic bias-variance tradeoff. More training data makes this worse: at 2M rows, CatBoost AP on OOD week 3 drops to 0.181 vs 0.248 at 500k, as richer exposure to calm-regime patterns makes the model more confidently wrong under stress.

**Trade intensity is the dominant predictor.** Logistic regression standardised coefficients: trade_intensity_10s (+0.48), trade_intensity_5s (−0.36), volume_acceleration (+0.18). SHAP values (GBT): trade_intensity_10s (0.319), trade_intensity_5s (0.181), volume_acceleration (0.055). Informed traders are detectable by their urgency — the temporal pattern of execution — not their footprint in the order book. This is consistent with microstructure intuitions about fleeting orders and order-to-trade ratios.

**Both models are underconfident.** Predicted probabilities are systematically lower than true toxic rates, with miscalibration worsening at higher predicted probabilities — precisely where accurate estimates matter most for spread adjustment. The classifier is best calibrated in the low-toxicity regime where intervention is not needed, and least calibrated in the high-toxicity regime where accurate probability estimates matter most.

**Neither model beats the trivial predictor on Brier score.** Both score above the baseline Brier of 0.1454 (always predicting the base rate), reflecting near-zero resolution: the model rarely varies its output, so bin-level actual rates barely deviate from the overall base rate.

### Precision-Recall at Operationally Relevant Thresholds (test week 3)

Market maker breakeven precision = S/(S+L) = 5/(5+8) = **0.38**.

**Logistic Regression**

| Threshold | Precision | Recall | FPR | Intervention Rate |
|-----------|-----------|--------|-----|-------------------|
| 0.10 | 0.383 | 0.159 | 0.055 | 7.3% |
| 0.15 | 0.403 | 0.066 | 0.021 | 2.9% |
| 0.20 | 0.416 | 0.036 | 0.011 | 1.5% |

Logistic regression clears the breakeven precision bar at all thresholds but recall is below 16% even at T=0.10. The classifier identifies a statistically detectable subset of toxic flow but leaves the market maker exposed to over 84% of adverse fills.

**GBT**

| Threshold | Precision | Recall | FPR | Intervention Rate |
|-----------|-----------|--------|-----|-------------------|
| 0.10 | 0.245 | 0.543 | 0.359 | 39.1% |
| 0.20 | 0.300 | 0.082 | 0.041 | 4.9% |

GBT never clears the breakeven precision bar. At T=0.10, recall is 0.543 but FPR is 0.359 — flagging 39% of all trades as potentially toxic. Operationally unworkable.

**Recommended use:** continuous spread widening proportional to p_toxic rather than binary quote pulling. The classifier does not clear breakeven for a binary decision but has consistent directional value as a continuous signal.

### Bootstrap Confidence Intervals on AP (week 3, 1000 iterations)

| Model | Mean AP | 95% CI |
|-------|---------|--------|
| Logistic regression | 0.295 | [0.287, 0.304] |
| GBT | 0.249 | [0.243, 0.256] |

Confidence intervals do not overlap. The logistic regression outperformance on the stress regime is statistically robust, not a sampling artefact.

---

## Adaptive Market-Making

### Strategy

**Baseline:** Avellaneda-Stoikov (2008) infinite-horizon market maker with inventory skew. At high fill frequency on BTC perps (κ ≈ 19 trades/bar), the A-S spread formula collapses to one minimum tick ($0.10) and becomes independent of γ. Inventory control operates entirely through reservation price skew.

**Adaptive extension:** spread widens proportionally to per-bar classifier toxicity rate:

```
adaptive_spread = market_spread × (1 + k × toxic_rate)
```

k=5 calibrated on BTC week 1 (in-sample). Applied unchanged to weeks 2 and 3. Fill size derived from a risk budget:

```
fill_size = α × market_spread / (σ × mid)    [α = 0.5]
```

**Why Cartea & Sánchez-Betancourt (2025) does not transfer:** their closed-form price adjustment assumes the broker can identify informed and uninformed clients by name and stream bespoke quotes to each. Crypto order flow is anonymous by construction — there is no mechanism to separate client streams, rendering the signal extraction and optimal discount derivation inoperable. The anonymous venue structure is a hard constraint, not a data limitation.

### Out-of-Sample Results

k calibrated on week 1 only. Weeks 2 and 3 are strictly out-of-sample.

| Asset | Week | Baseline MtM ($) | Adaptive MtM ($) | Improvement ($) | 95% CI | Toxic Fill Rate — Base | Toxic Fill Rate — Adaptive | Avg Spread Ratio |
|-------|------|-----------------|-----------------|-----------------|--------|----------------------|--------------------------|-----------------|
| BTC | week2 | -4,726 | -4,271 | +455 | [-5,155, +2,376] | 1.1% | 0.5% | 1.044 |
| BTC | week3 | +349 | +2,339 | +1,991 | [-2,149, +7,002] | 6.2% | 4.2% | 1.209 |
| ETH | week2 | -368 | -87 | +280 | [-1,182, +542] | 3.2% | 2.5% | 1.098 |
| ETH | week3 | -107 | +12 | +118 | [+78, +178] | 12.6% | 11.1% | 1.402 |
| SOL | week2 | -34 | +52 | +86 | [-152, +344] | 5.3% | 2.5% | 1.156 |
| SOL | week3 | -35 | +52 | +87 | [-432, +487] | 16.2% | 13.7% | 1.628 |

CIs computed via block bootstrap (B=200, daily blocks, percentile method). ~4–7 daily blocks per week — the evaluation is genuinely underpowered.

### Key Findings

All six out-of-sample cells show positive improvement. Five of six CIs span zero, reflecting ~7 daily bootstrap blocks per week — the evaluation cannot distinguish signal from noise at this data volume. The honest claim is directional, not conclusive.

**ETH week 3 is the only statistically distinguishable result.** CI [+$78, +$178] lies entirely above zero despite only 4 daily blocks. ETH week 3 is the stress/correction regime with 12.6% baseline toxic fill rate and average spread ratio of 1.40 — the highest-toxicity cell in the evaluation. The signal has most value precisely when informed flow is most concentrated.

**The adaptive strategy consistently reduces toxic fill rate in every cell.** The mechanism is working: wider spreads on toxic bars prevent fills at adverse prices. The largest reduction is in the highest-toxicity cells (SOL week 3: 16.2% → 13.7%).

**BTC dollar improvements are larger in absolute terms due to notional, not signal strength.** BTC week 3 (+$1,991) reflects larger fill sizes at higher prices, not a stronger classifier signal. SOL and ETH show comparable percentage improvements.

---

## Limitations

**The evaluation is underpowered.** Bootstrap CIs span zero in 5/6 backtest cells due to ~7 daily blocks per week. A production evaluation requires months of out-of-sample data, not weeks.

**Recall is low at any viable precision threshold.** The classifier captures at most 16% of toxic flow at breakeven precision. The fundamental constraint is data resolution: without millisecond-level queue position, order-to-trade ratios, or participant identifiers, individual trade toxicity cannot be resolved with high confidence from public data.

**The classifier is least calibrated where it matters most.** Underconfidence in the high-toxicity regime means spread widening will systematically under-react to the signal. This is unrecognised risk — worse than overconfidence, which would merely cost revenue.

**k is calibrated on one asset in one regime.** k=5 was chosen from BTC week 1 only. Per-asset calibration would be more principled but risks overfitting to the single in-sample week available.

**Three structural gaps prevent direct application of equity microstructure frameworks:**

1. *Anonymity.* Frameworks like Cartea & Sánchez-Betancourt (2025) assume named counterparties. Crypto order flow is anonymous by construction.

2. *Venue structure.* Crypto perpetual futures combine spot and derivatives features, with funding rate arbitrage creating informed-appearing flow that is mechanically rather than informationally driven.

3. *Data resolution.* Without millisecond queue position, order-to-trade ratios, or participant identifiers, individual trade toxicity cannot be resolved from public data alone.

**What a production system would need:** co-location and nanosecond timestamps; order-to-trade ratios to distinguish aggressive informed flow from layering; cross-venue signal aggregation across spot, perp, and options; real transaction costs that make the spread-widening tradeoff economically precise.

---

## Reproducing Results

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Feature pipeline (requires raw Bybit data)
python src/features/build_features.py

# Classifier evaluation
jupyter notebook notebooks/03_classifier_evaluation.ipynb

# VPIN robustness grid
python src/evaluations/vpin_robustness.py

# Backtest + bootstrap CIs
python src/evaluations/bootstrap_eval.py
```

Raw data: Bybit public data portal (data.bybit.com). Assets: BTCUSDT, ETHUSDT, SOLUSDT USDT perpetual contracts. Weeks: Sep 9–15 2024, Oct 28 – Nov 3 2024, Feb 24 – Mar 2 2025.