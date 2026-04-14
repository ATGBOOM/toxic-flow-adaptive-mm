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
application) → time-series of derived features sampled at regular intervals → parquet. 
Implemented in Session 7.

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

## Status
- [x] Phase 0: Prerequisites (Sessions 1-3)
- [x] Phase 1: Data pipeline (Session 4)
- [x] Phase 2: EDA (Session 5)
- [x] Phase 3: VPIN implementation (Session 6) ← current
- [ ] Phase 4: Feature engineering + LOB reconstruction (Session 7)
- [ ] Phase 5: Toxicity classifier (Sessions 8-9)
- [ ] Phase 6: Adaptive market-making (Sessions 10-11)
- [ ] Phase 7: Writeup and packaging (Sessions 12-13)