# Defense Questions and Answers

## Research Design

### Why study perpetual futures instead of spot?

Perpetuals concentrate leveraged trading and often contribute strongly to crypto
price discovery. They are therefore a relevant venue for adverse-selection
research. The tradeoff is that funding and basis arbitrage can produce flow that
looks informed without reflecting private directional information.

### Why only three weeks?

The weeks were selected to contrast consolidation, breakout, and correction
regimes under a manageable reconstruction workload. This is a regime case study,
not a representative population sample. Months of untouched data are required
for a production claim.

### Is selecting regimes after observing them a bias?

Yes, if the goal is unconditional performance estimation. Here it is defensible
as stress-test design, provided results are described as conditional case studies
and not typical-market estimates.

## Label

### Why call the label toxic?

It captures the market maker's operational definition of toxicity: an aggressor
is followed by a same-direction move large enough to make passive liquidity
adverse. It should be called an adverse-move toxicity proxy because participant
identity and private information are unobserved.

### Why 8 bps and 10 seconds?

Eight bps was near the 90th percentile of absolute ten-second BTC returns in the
threshold study. Ten seconds is intended to capture short-run impact rather than
long-horizon drift. Both are modeling choices and should be sensitivity-tested
across assets and regimes.

### Does the trade cause the future move?

The label does not establish causality. The trade may cause impact, anticipate a
common public signal, or simply occur during an existing trend.

### Why are toxic trades so clustered?

The market has bursty activity, but the label construction also creates overlap:
many trades inside one ten-second price move receive the same label. A Hawkes
interpretation needs a null model that controls for trade intensity and
overlapping horizons.

## Features

### Why exclude trade sign?

The intention was to avoid learning that buys or sells are inherently more toxic.
However, signed volume imbalance remains directional, so symmetry is incomplete.
A cleaner design would express flow relative to the current trade direction or
include sign interactions explicitly.

### Why use trade intensity?

Informed or urgent execution often arrives in bursts before a signal decays.
Intensity is observable from public data and transferred better than static book
shape in this sample. It is not uniquely informed flow; liquidations, news
reaction, and algorithmic slicing can create the same pattern.

### Why use book depth at several levels?

The levels distinguish immediate top-of-book pressure from broader visible
liquidity. They may capture different resilience horizons. Correlation among
levels also means coefficient-level interpretations are conditional.

## VPIN

### What is VPIN measuring?

It is a rolling normalized absolute buy-sell imbalance on a volume clock. It
measures one-sided flow, not informed trading directly.

### Why can VPIN fail during stress?

If aggressive informed selling and aggressive opportunistic buying occur
together, absolute net imbalance can remain modest even while adverse selection
and volatility are high. A long rolling window can average transient imbalances
away. This is a plausible mechanism, not a causal finding from the current code.

### Why does shorter-window VPIN improve?

Short windows react before opposing episodes are averaged together. They also
increase variance and sensitivity to noise, so improvement may be
sample-specific. That is why the parameter grid matters.

### Is this exactly Easley et al. VPIN?

No. The code uses the venue's observed aggressor side rather than the original
bulk-volume classification approach. It is best described as an
aggressor-classified VPIN variant.

## Models and Metrics

### Why logistic regression?

It provides a low-variance, interpretable baseline. In the selected stress week,
its simpler relationship transferred better than CatBoost's nonlinear
interactions. That is evidence for this sample, not proof that linear models are
always superior.

### Why did CatBoost win on week 2 but lose on week 3?

Week 2 was closer to the calm training distribution. CatBoost could exploit
regime-specific nonlinearities there. Under the larger week-3 distribution
shift, those relationships did not transfer as well.

### Why AP rather than accuracy?

With a low positive rate, predicting every trade as non-toxic yields high
accuracy. AP measures the precision-recall ranking of the minority class. Its
no-skill reference is the positive prevalence.

### Why report AUC as well?

AUC is prevalence-insensitive and measures ranking across the entire threshold
range. It can still look acceptable when high-precision operation is poor, so it
must be paired with AP and threshold metrics.

### What does the Brier score show?

It measures squared probability error. In week 3, both models were worse than a
constant predictor using the test prevalence, indicating weak probability
calibration and resolution under distribution shift.

### Are the AP confidence intervals valid?

They are exploratory. The current IID bootstrap ignores overlapping labels and
temporal clustering. A defensible comparison uses a paired temporal-block
bootstrap of the AP difference.

### Why is week 3 not a final test?

It was used both to compare models and to state that logistic regression should
be preferred. Once a dataset influences model choice, it becomes validation
data. A later untouched period is needed for final evaluation.

## Market Making

### What is Avellaneda-Stoikov?

It is a stochastic control framework balancing spread revenue against inventory
risk. Inventory moves the reservation price, while optimal spread depends on
risk aversion, volatility, time, and the decay of fill intensity with quote
distance.

### Is the implemented strategy truly Avellaneda-Stoikov?

It is A-S-inspired. It retains inventory-based reservation-price skew but
replaces the theoretically calibrated spread with the prevailing market spread
and a heuristic toxicity multiplier. The code's `kappa` is trade count per bar,
not the A-S fill-intensity slope.

### Why widen spreads when toxicity rises?

Wider quotes reduce fill probability and demand more compensation for adverse
selection. The cost is fewer benign fills and lower spread-capture volume.

### Did the classifier improve PnL?

The historical experiment does not answer that. It used the realized
forward-looking label, so it is an oracle upper bound. The reusable path now
consumes timestamped out-of-sample probabilities known before each quote, but
the historical PnL and intervals have not been rerun.

### Why is the current fill model optimistic?

It assumes fills from trade-price crossing without queue position, latency, or
available quantity. It also omits fees, rebates, funding, and hedging costs.

### How should the backtest be corrected?

The reusable code now uses interval `t` state and prediction to place quotes for
`t+1`, then evaluates only later trades. The next redesign should track each
fill, quantity, queue assumption, fee, and inventory transition explicitly and
compare paired daily PnL over a much longer untouched period.

## Engineering

### Why Parquet?

Columnar storage reduces repeated I/O and allows expensive order-book
reconstruction to be separated from feature and model experiments.

### How do you prevent leakage?

Training and evaluation weeks are time ordered, forward return is excluded from
features, rolling-feature truncation tests verify that future rows do not alter
past values, invalid terminal labels are dropped, and the reusable backtest uses
prior-bar predictions rather than realised targets. Historical PnL artifacts
were not rerun after that fix.

### What would you improve first?

1. Rerun the prediction-driven backtest over a final untouched period.
2. Add fees, queue assumptions, and event-level fill accounting.
3. Fix SHAP and dependence-aware uncertainty.
4. Add a final untouched multi-month test.
5. Consolidate VPIN implementations and expand synthetic integration tests.
