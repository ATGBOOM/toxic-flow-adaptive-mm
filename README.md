Project purpose:
Implementing VPIN-based toxic order flow detection on crypto LOB data, extended with a Bayesian classifier and adaptive market-making strategy. Framed as a methodological test of whether equity microstructure patterns transfer to crypto venues. Not a trading bot.
Papers this is based on:

Easley, López de Prado, O'Hara (2012) — VPIN
Avellaneda & Stoikov (2008) — optimal market making
Cartea & Sánchez-Betancourt (2025) — analytical toxic flow adjustment
Cartea & Sánchez-Betancourt (2023) — PULSE feature set and evaluation framework
Bieganowski (2026) — crypto microstructure feature engineering

Data:

Source: Binance public data portal (data.binance.vision), spot trades
Assets: BTCUSDT, ETHUSDT, SOLUSDT
Three deliberately chosen weeks across distinct market regimes:

Week 1: Sep 9-15 2024 — low volatility consolidation ~$55k
Week 2: Oct 28 - Nov 3 2024 — directional breakout above $70k
Week 3: Feb 24 - Mar 2 2025 — stress/correction from $100k+ highs



Why three regimes:
To test whether toxicity patterns and classifier performance are regime-dependent, rather than overfitting to one market condition.
Pipeline:
Raw CSV → typed DataFrame (ms/μs timestamp auto-detection, direction as +1/-1 sign, gap detection) → parquet. One file per asset per week.