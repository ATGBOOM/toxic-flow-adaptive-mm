# Presentation Guide

This is a 12-slide structure for a 12-15 minute technical presentation.

## Slide 1: Research Question

Title: **Do Equity Toxic-Flow Tools Transfer to Crypto Perpetuals?**

Say:

> I tested whether VPIN and an adaptive market-making framework transfer from
> equity microstructure to anonymous crypto perpetual futures. The project is a
> methodological study, not a production trading strategy.

## Slide 2: Why This Market Is Different

Show:

- anonymous counterparties;
- continuous trading;
- high message and trade intensity;
- leverage and funding;
- fragmented spot, perpetual, and options price discovery.

Core point: transfer is not automatic because the market structure and available
information differ from the assumptions behind the original models.

## Slide 3: Data and Regimes

Show the three assets and three selected weeks.

Say explicitly:

> I deliberately selected contrasting regimes. This improves stress testing but
> prevents me from claiming that the sample represents a typical year.

Mention missing ETH and SOL order-book days.

## Slide 4: Pipeline

Use one diagram:

```text
trades + L2 updates -> reconstructed book -> features -> adverse-move label
                    -> VPIN and ML scores -> evaluation
```

Explain the event clock: book features are sampled around trade arrivals rather
than arbitrary calendar snapshots.

## Slide 5: What "Toxic" Means Here

Show:

```text
buy + price rises > 8 bps within 10s  -> toxic
sell + price falls > 8 bps within 10s -> toxic
```

Say:

> This is an ex-post adverse-selection proxy. It does not identify private
> information or prove causality.

Defend ten seconds as a short impact horizon and eight bps as an empirically
chosen BTC threshold. Admit that the same threshold has a different percentile
meaning for ETH and SOL.

## Slide 6: VPIN

Show the formula:

```text
VPIN = sum |buy volume - sell volume| / total volume
```

Explain equal-volume buckets and the rolling window.

Result:

> At the chosen standard parameters, VPIN ranked the adverse-move label at about
> chance. Shorter windows recovered some signal in stress data, so the precise
> conclusion is parameter-sensitive: long-window VPIN transferred poorly, not
> that every VPIN specification always fails.

Do not say the two-sided-flow mechanism was proven. Call it a plausible
explanation consistent with the observed pattern.

## Slide 7: Feature Set

Group the 16 features rather than listing all of them:

- book shape: spread, microprice, depth imbalance, pressure;
- flow speed: 1/5/10-second trade intensity and acceleration;
- flow direction/size: signed volume imbalance and normalized quantity;
- legacy signal: VPIN.

Explain why raw price and the forward return were excluded.

## Slide 8: Walk-Forward Design

Show:

```text
train W1       -> evaluate W2 and W3
train W1 + W2  -> evaluate W3
```

Say:

> Time ordering is preserved across regimes. However, because I used week 3 to
> choose the preferred model, it is best viewed as validation rather than a final
> untouched test set.

## Slide 9: Classifier Results

Show only the key week-3 table:

| Model | AP | AUC |
|---|---:|---:|
| VPIN | 0.177 | 0.507 |
| Logistic | 0.294 | 0.664 |
| CatBoost | 0.244 | 0.600 |

Explain:

- no-skill AP is approximately the toxic prevalence;
- logistic regression transferred better than CatBoost;
- higher-capacity nonlinear structure did not generalize to the selected stress
  regime;
- probability calibration remained poor.

Avoid presenting the current SHAP ranking until it is recomputed correctly.

## Slide 10: Economic Interpretation

Show the precision/recall tradeoff, but do not claim a universal 0.38 breakeven
precision.

Say:

> Higher thresholds isolate a more toxic subset but miss most toxic episodes.
> Whether a threshold is economically viable depends on fill-conditioned losses,
> spread capture, fees, rebates, and queue position, none of which are fully
> modeled here.

## Slide 11: Market-Making Experiment

Present this as an audit-corrected limitation:

> The current simulator widens spreads using the realized forward toxicity
> label. It therefore estimates an oracle upper bound in a simplified fill
> model. It does not yet establish classifier-driven PnL improvement.

Show the quote logic, not the existing confidence-interval table.

Explain what a valid rerun needs: timestamped predictions, lagged causal quote
placement, later fills, fees, and paired daily evaluation.

## Slide 12: Conclusions

Use three claims:

1. Standard long-window VPIN transferred poorly to this selected crypto sample.
2. Simple trade-intensity relationships transferred better than the nonlinear
   CatBoost model into the selected stress week.
3. A deployable market-making conclusion requires a causally ordered,
   prediction-driven backtest over much more out-of-sample data.

End with:

> The contribution is not a profitable strategy claim. It is identifying which
> parts of an equity-microstructure pipeline survive the transfer, which fail,
> and why the available public data limits the answer.

## Visuals to Prepare

1. Pipeline diagram.
2. One market-regime price chart.
3. VPIN AUC parameter heatmap.
4. Precision-recall curves with prevalence lines.
5. Reliability diagram.
6. Per-asset week-3 metric table.
7. A causal timing diagram for the corrected backtest.

## Claims to Avoid

- "VPIN is universally useless in crypto."
- "The classifier identifies informed traders."
- "SHAP proves trade intensity causes toxicity."
- "The classifier produced the reported PnL improvement."
- "ETH week 3 is statistically significant."
- "The system is production-ready."
- "More data made CatBoost worse" without saying this was one fixed sampling
  comparison rather than a learning-curve study with repeated seeds.
