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
regime-dependent, rather than fitting to one market condition. Also allows cross-regime 
comparison of OFI signal decay, spread dynamics, and VPIN behaviour.

**Sampling bias caveat:** These weeks were selected for regime variation, not randomly. 
Findings should not be generalised as representative of typical market conditions.

## Pipeline

### Trade Data
Raw CSV → typed DataFrame (timestamp auto-detection for ms/μs, direction as +1/-1 sign, 
gap detection) → parquet. One file per asset per week.

### Order Book Data
Raw JSON lines → order book reconstruction (snapshot initialisation + sequential delta 
application) → time-series of derived features sampled at regular intervals → parquet. 
To be implemented in Session 7.

## EDA Findings (Session 5)

### 1. Sampling frequency dominates distributional statistics
Tick-level return statistics are dominated by microstructure artefacts. Kurtosis drops 
from ~10,000 at tick level to 5-10 at 1-minute bars — the correct range for analysis. 
Skewness at tick level contains no interpretable signal. This directly motivates the 
volume clock in VPIN: time-based sampling produces statistics that reflect trade arrival 
rates more than genuine price dynamics.

### 2. OFI signal decays at dramatically different rates across regimes
Correlation between order flow imbalance (OFI) and next-period returns varies by both 
frequency and regime:

| | Raw OFI 1min | Raw OFI 10s | Normalised OFI 1min | Normalised OFI 10s |
|--|--|--|--|--|
| Week 1 (consolidation) | 0.0009 | 0.054 | -0.002 | 0.071 |
| Week 2 (breakout) | 0.006 | 0.045 | 0.020 | 0.085 |
| Week 3 (stress) | 0.015 | 0.017 | 0.014 | 0.061 |

Decay ratio (10s correlation / 1min correlation): Week 1 = 54x, Week 2 = 8x, Week 3 = 1x.

The stress/correction week shows persistent OFI — the signal survives temporal aggregation, 
consistent with sustained directional selling pressure. The consolidation week shows fast 
decay, consistent with short bursts of noise trading that rapidly reverse.

### 3. Normalisation sharpens signal at short frequencies, hurts at long
Volume-normalised OFI (signed volume / total volume, range [-1,+1]) outperforms raw OFI 
at 10-second bars across all regimes. At 1-minute bars normalisation adds noise — the 
denominator becomes dominated by a few large trades. Week 2 (breakout) shows strongest 
normalised OFI predictability once volume activity is controlled for, pointing toward 
momentum-driven rather than information-driven flow.

### 4. Correlation magnitudes are small throughout
Maximum observed correlation: 0.085 (normalised OFI, 10s, breakout week). All findings 
above are directional/relative — OFI alone is a weak predictor. This motivates the 
richer feature set and classifier in Phase 3.

## Status
- [x] Phase 0: Prerequisites (Sessions 1-3)
- [x] Phase 1: Data pipeline (Session 4)
- [x] Phase 2: EDA (Session 5)
- [ ] Phase 3: VPIN implementation (Session 6)
- [ ] Phase 4: Feature engineering + LOB reconstruction (Session 7)
- [ ] Phase 5: Toxicity classifier (Sessions 8-9)
- [ ] Phase 6: Adaptive market-making (Sessions 10-11)
- [ ] Phase 7: Writeup and packaging (Sessions 12-13)