# Presentation Notes — Self-Rehearsal Reference

Not a script to read verbatim — you're showing the actual code on screen, so
this is here to tell you **which file to have open, what to point at, and
what points to hit**, in your own words. Every stop has: the file(s) to
show, the **flow** (what happens, in order — read this first to know where
you are), then **talking points** as short bullets, then **Reserve** —
deeper material, not spoken unless asked.

~19–22 min total. No cold open. Two flagship deep-dives carry the weight;
everything else is a fast pass, a few bullets per stop, depth held in
Reserve.

Known doc staleness, for your own awareness only, never say this unprompted:
`AUDIT_FINDINGS.md` and `CODE_WALKTHROUGH.md` both predate the `a0ca776`
hardening commit — they still describe `vpin_robustness.py` as containing a
drifted duplicate VPIN implementation and claim an exact-fill bucket doesn't
flush until the next trade. Neither is true of the current code. If asked
about either, answer from the current code, not the older docs.

---

## 1. Problem statement (~30–45s)

**No file — just talk.**

- This pipeline reconstructs a crypto order book, derives a toxicity signal
  and classifier from it, and backtests a market-making strategy.
- The throughline for the whole talk: a strict no-look-ahead guarantee, at
  every boundary — one violation anywhere silently inflates every result
  downstream of it.

---

## 2. Architecture diagram (~90–120s)

**FILE:** `docs/presentation.html` — slide 2 (the diagram)

**Flow (walk it left to right):**
1. Bybit data splits into two cleaning stages — trades, L2 order book.
2. VPIN comes off cleaned trades directly (parallel to the book, not
   downstream of it).
3. Feature matrix is the merge point — trailing features, label, VPIN,
   book features come together here.
4. Classifier → predictions export → causal backtest. Notebooks are
   reporting only, off to the side.

**Talking points:**
- The merge point is also the **caching boundary** (now marked on the
  diagram): book replay is expensive, compute once; everything below it is
  cheap and iterated on — re-labelling is seconds, not a replay.
- Every stage writes **Parquet**, not CSV or a database — columnar/typed,
  a stage needing 3 columns doesn't pay to read the rest. A database is
  the real alternative, overkill for a batch pipeline; DuckDB over these
  files is the natural next step if that changed.
- One invariant enforced right at entry, before anything else: the
  **loader sorts every trade file by timestamp on load**, defensively,
  not an assert-and-fail. Everything downstream silently assumes ascending
  time. First instance of the invariant the rest of the talk keeps coming
  back to.

**Reserve — `src/data/loader.py`:**
- Sort was found unguarded during audit, now guarded by a monotonic-
  timestamp regression test.
- Gap detection (`diff() > 60s`) only prints — a smoke test, not gap
  handling.
- `side.map({"Buy": 1, "Sell": -1})` turns anything unexpected into `NaN`,
  silently poisoning signed-volume features. No validation.

---

## 3. Flagship #1 — order book (~3–4 min)

**FILE:** `src/features/reconstructor.py` — functions: `apply_update()`,
`compute_book_features()`, `reconstruct_and_extract_from_state()`,
`process_week()`

> "Let's go deep on the first real engineering decision — the order book."

**Flow:**
1. `apply_update` — snapshot clears + reloads both sides; delta upserts
   only the named levels, size 0 deletes.
2. `compute_book_features` — best bid/ask, spread, microprice, depth
   imbalance at 4 levels, pressure ratio.
3. `reconstruct_and_extract_from_state` — replay loop: for each book
   message, flush any pending trade *first* (sample book state), then
   apply the message. Strict `<` on timestamp.
4. `process_week` — loops days, book state (`SortedDict`) created once
   outside the loop, carried by reference across the whole week.

**Talking points:**
- Book is two `SortedDict`s now, not plain dicts — bids/asks.
- **The invariant that matters most:** loop samples the book for a
  pending trade *before* applying the next message. A trade never sees a
  book message after its own. Enforced by loop order, not a filter —
  can't be silently bypassed.
- **Test found a real bug:** two float representations of the same price
  should collapse to one key; without canonicalizing to 8 decimals, a
  size-0 delete could silently miss. Now guarded by a regression test.
- **Continuity:** `SortedDict`s created once, outside the day loop — the
  book doesn't treat "day" as a real boundary, because it isn't one.
- **dict → SortedDict tradeoff:** dict was O(1) upsert but O(L·logL)
  resort on every feature read. SortedDict: O(logL) upsert, O(logL) best
  bid/ask via `peekitem`, O(N) depth via a slice — no full resort.
  **Benchmarked, not assumed:** ~1.6x faster reads, ~6.7x slower writes at
  500 levels. Reads happen once per trade, writes once per message —
  messages vastly outnumber trades, so the real net win for *this batch
  job* is genuinely uncertain; the complexity argument is unambiguous for
  a live system instead.
- Book replay is stateful/imperative by necessity (sequential messages);
  everything downstream is pure functions over DataFrames (vectorizable).
  Different paradigm per stage, matched to what each needs.

**Reserve:**
- Crossed book unhandled — spread can go negative, undetected.
- Missing order-book day carries stale state into the next day until the
  next snapshot resyncs it. Known, still open.
- `namelist()[0]` assumes exactly one inner file per zip, unguarded.
- **Full structure comparison** (dict / SortedDict / tick-indexed array):
  tick-indexed array is O(1) everywhere, the fastest option, but needs a
  fixed tick size and bounded price range up front — fine for one
  instrument, a worse fit replaying three differently-scaled assets.

---

## 4. Fast pass (~4–5 min total)

> "That's the book. In parallel — off the same cleaned trades, not
> downstream of the book — VPIN runs independently. Moving faster through
> the next few stages; depth is there if asked."

### 4a. VPIN

**FILE:** `src/models/vpin.py` — functions: `build_volume_bucket()`,
`compute_rolling_vpins()`, `compute_vpin()`. Also:
`src/evaluations/vpin_robustness.py` (the unified import).

**Flow:**
1. `build_volume_bucket` — accumulate trades into fixed-volume buckets,
   splitting a trade across a boundary if needed; closes the instant a
   bucket exactly fills.
2. `compute_rolling_vpins` — cumsum-based rolling sum over completed
   buckets only.

**Talking points:**
- Volume clock, not calendar time — fixed time intervals overweight quiet
  periods, underweight bursts. Same causal discipline as the book, applied
  to a different axis: only *completed* buckets ever enter a rolling
  value.
- Rolling window used to re-sum the whole window per bucket (quadratic);
  now a single cumsum pass, linear, verified behavior-identical.
- **Found a real bug via testing:** hand-computed VPIN against a synthetic
  sequence, checked it against both the production path and the
  parameter-grid script — caught the grid script running a second,
  drifted implementation. Unified onto one canonical function.

### 4b. Feature matrix

**FILE:** `src/features/build_features.py` — functions:
`add_trade_features()`, `add_toxicity_label()`, `add_vpin_feature()`,
`build_full_features()`

**Flow:**
1. Load week's trades + per-day book features.
2. Merge trades + book state (positional, row-count guarded).
3. `add_trade_features` — trailing windows.
4. `add_toxicity_label` — forward-only horizon.
5. `add_vpin_feature` — join VPIN.
6. Save `full_features.parquet`.

**Talking points:**
- Trailing windows use `searchsorted` + prefix sums instead of a per-row
  scan — O(log n) per lookup vs O(window size) per row, matters at tens
  of millions of rows.
- Forward-only label: a row without a full future horizon gets *dropped*,
  not clipped.
- Merge is positional with a row-count guard — mismatched days get
  skipped entirely rather than silently misaligned. Timestamp-keyed join
  is the fix I'd still make.

**Reserve:** `daily_vol / 7` in `add_vpin_feature` is hardcoded, doesn't
account for weeks with missing days (ETH/SOL week 3) — still open.

### 4c. Classifier

**FILE:** `src/models/classifier/data_loader.py` — `load_asset_week()`,
`subsample_stratified()`, `prepare_split()`. Also:
`src/models/classifier/classifier.py` — `VPINBaseline`,
`ToxicityClassifier.fit()`/`.evaluate()`

**Flow:**
1. `load_asset_week` — load features, add 3 more (spread_bps,
   microprice_minus_mid, qty_normalised), drop VPIN-warmup/invalid rows.
2. `prepare_split` — pool assets for train weeks, stratified subsample;
   test weeks loaded full, no subsampling.
3. `ToxicityClassifier.fit` — fit VPIN baseline (no-op), logreg (scaled),
   GBT (unscaled).
4. `.evaluate` — AP/AUC/Brier + precision at fixed thresholds.

**Talking points:**
- The 3 extra features live here, not upstream in `build_features.py` —
  cross-asset normalization is a *modeling* decision, not a market fact,
  so it stays at the point of consumption. Same caching-boundary
  principle as the book-replay split, one layer downstream.
- Three models behind one `predict_proba` call — **Strategy pattern**.
  `VPINBaseline.fit()` does nothing at all, purely so it fits the same
  loop as the sklearn models — an **Adapter**.
- Only logreg gets scaled (fit train-only, never refit on test) — it
  optimizes via gradient descent, scale distorts the optimizer and the L2
  penalty. Trees split on raw thresholds — scaling is a no-op for them.
- Chose this 3-model ladder over a neural net/random forest — ~15 tabular
  features is exactly where boosting already matches deep learning with
  less tuning. Real question was whether complexity earns its keep over
  the raw signal, and survives a regime shift — same no-look-ahead
  discipline applied to model evaluation: train only on weeks strictly
  before the test week.
- Training capped at 500k rows, stratified to true prevalence, not
  rebalanced — full pooled set is tens of millions of rows, reruns need
  to stay fast. Test sets never subsampled.

**Reserve:** `spread_bps`/`qty_normalised` are properly normalized;
`microprice_minus_mid` is **not** — raw dollar difference, still-open gap
that undercuts the pooling argument for that one feature. Own it if asked.

### 4d. Predictions export

**FILE:** `src/evaluations/save_predictions.py`

**Flow:**
1. `prepare_split` on weeks 1+2 pooled (all 3 assets).
2. Fit `ToxicityClassifier`, save logreg/scaler/GBT via `joblib`.
3. Per asset: load week 3, predict with all 3 models, export parquet with
   `asset, week, source_row, timestamp, y_true, p_logreg, p_gbt, p_vpin`.

**Talking points:**
- Exported per-asset, not pooled, deliberately — a paired statistical
  test needs one observation per asset, not one blended number.
- Identity preserved (`source_row`, `timestamp`) — this is what lets the
  backtest safely reattach a prediction to its trade next.
- **Memory management, explicit, not incidental:** `del split2;
  gc.collect()` right after training, and `del df, X, y, p_logreg, p_gbt,
  p_vpin; gc.collect()` at the end of every per-asset loop iteration.
  Pooling weekly data across assets is multi-GB, and this loop loads
  several such frames back-to-back. Honest framing, tested not assumed:
  I checked whether this is cleaning up genuine reference cycles by
  reproducing the same derive-columns/slice pattern and calling
  `gc.collect()` explicitly — it found zero cyclic garbage; refcounting
  alone had already freed everything. So the real justification isn't
  "refcounting misses this" — it's forcing collection to run at a chosen
  moment, defensively, rather than trusting the automatic allocation-count
  trigger to fire before the next multi-GB load starts. I don't have
  evidence this specific code needs it; it's disciplined practice, not a
  proven leak.

**Reserve:** the saved `.joblib` model artifacts aren't loaded back
anywhere in the current repo — a live-scoring path is exactly what would
consume them.

---

## 5. Flagship #2 — causal backtest (~4–5 min)

**FILE:** `src/evaluations/bootstrap_eval.py` — functions:
`load_backtest_data()`, `build_bars()`, `run_backtest()`,
`block_bootstrap_improvement()`, `compute_bootstrap_ci()`

> "That's the pipeline up to a prediction. Now the second deep dive — this
> is where the sharpest engineering problem in the whole project lived."

**Flow — read this first, this is the whole shape of the file:**
1. `load_backtest_data` — merge features + predictions by `source_row`
   identity, cross-check timestamps.
2. `build_bars` — aggregate ticks into 1-second bars.
3. `run_backtest` — **the core algorithm.** Simulate one strategy run:
   quote, check fills, track cash/inventory, mark to market.
4. `block_bootstrap_improvement` / `compute_bootstrap_ci` — resample daily
   blocks, rerun step 3 many times, take percentiles for a CI.
5. `build_regime_table` / `main` — package it all into the reported table.

**The algorithm, `run_backtest` specifically:**
- Split the work by what's path-dependent and what isn't. Sigma, fill
  size, gamma, spread — none of these depend on the running simulation
  state, so they're **precomputed as numpy arrays, once, vectorized**.
  Only cash and inventory are genuinely sequential (each bar's value
  depends on every fill before it) — those run in a tight Python loop,
  and nothing else does.
- Real number, not a guess: **~0.5–1s per backtest on 600k bars, vs. ~30s
  with `iterrows()`.** Matters because the bootstrap reruns this hundreds
  of times per asset-week.

**The invariant — say this plainly, don't over-explain it:**
- State/signal from bar `t` can only place quotes for bar `t+1`; only
  bar `t+1`+ trades can fill them. Enforced three ways: structurally
  (`i-1`/`i` indexing), at the interface (hard KeyError without a named
  prediction column), and by two synthetic tests (label vs. prediction
  pointing opposite ways; same-bar fill rejected).
- This replaced an earlier version that computed a bar's quote from that
  *same bar's* realized label — exactly the class of error those two
  tests now catch.

**What's still open, honestly:**
- Block bootstrap concatenates resampled days into one continuous
  simulated path, inventory carried straight across — artificial
  transitions between days that were never adjacent.
- Historical PnL table predates the causal fix, not rerun — historical,
  oracle-signal result only.

**Reserve — the history, if asked "was there a bug" or "how'd you find it":**
- Earlier `build_bars()` computed `toxic_rate` straight from the realized
  label; backtest used that same bar's value as its own signal — the
  strategy's signal was, literally, the future. Notebook text claimed
  previous-bar; code never shifted anything.
- Even a naive one-bar shift wouldn't fix it — label horizon is 10s,
  longer than a 1s bar, so the previous bar's label still depends on
  prices up to 9s after the current bar begins.
- One more fix, in-code comment: original notebook credited an ask fill
  at the bid price. One-line accounting error.

**Reserve (Q&A only):**
- `fill_size = alpha * market_spread / (sigma * mid)` is dimensionally
  dollars over (dimensionless × dollars) = dimensionless, used as if it
  were base-asset units.
- `gamma`'s formula algebraically collapses to `1 / (2*M*alpha)` — a
  constant depending only on two fixed hyperparameters, despite looking
  dynamic.
- Inventory limit check happens *before* a fill — can overshoot by up to
  one fill size.
- `toxic_fill_rate()` uses `inventory.diff()` — same-bar two-sided fills
  that net to zero are invisible to it.
- Merge uses `validate="one_to_one"` plus the timestamp cross-check — real
  defense against silent misalignment.
- `kappa` here is trades-per-bar, not the A-S fill-intensity sensitivity —
  A-S-inspired, not a calibrated A-S solution.

---

## 6. Test suite — proving the invariants (~2 min)

**FILE:** `tests/test_core.py`

> "Before I wrap up — every invariant I just claimed has a synthetic test
> behind it, not just a description. Let me run a few, live."

**Run these, in the same order as the talk — each one proves a specific
claim already made, not a new one:**

1. `pytest tests/test_core.py -k load_trades_timestamps_are_monotonic -v`
   → the loader's sort invariant (Architecture section).
2. `pytest tests/test_core.py -k apply_update_canonicalizes_price_key -v`
   → the float-key fix (Flagship #1).
3. `pytest tests/test_core.py -k reconstruct_samples_book_state_at_or_before_trade -v`
   → the book replay's causal ordering (Flagship #1's core invariant).
4. `pytest tests/test_core.py -k no_lookahead_leakage_rolling_features -v`
   → trailing features never see a future trade (feature matrix, 4b).
5. `pytest tests/test_core.py -k backtest_ignores_realised_target_column -v`
   → a realized label can never drive the quote (Flagship #2, test 1 of 2).
6. `pytest tests/test_core.py -k backtest_quotes_from_previous_bar_and_fills_on_current_bar -v`
   → same-bar fills get rejected (Flagship #2, test 2 of 2).

**Talking points:**
- All 14 tests run in well under a second, synthetic, no external data
  needed — safe to run live at any point in the talk, not just here.
- These are boundary-constructed or hand-computed scenarios, not smoke
  tests — e.g. the float-drift test literally constructs `0.1 + 0.2` to
  get the exact representation-error case, not a fuzzed random input.
- 8 more tests beyond these 6 cover PnL arithmetic in isolation, inclusive
  trailing-window boundaries, label-horizon validity, and identity
  preservation through the classifier loader — mention if asked "is that
  all of them."

**Reserve — if asked "what's NOT tested":** order-book timestamp ordering
across days, data gaps inside an otherwise-valid label horizon,
prediction-to-bar alignment beyond the merge's identity check, inventory
limits and two-sided same-bar fills, a full deterministic end-to-end
synthetic integration test. Real gaps, not covered here.

---

## 7. Close (~60–90s)

**No file.**

- To close out: two things I'd still fix — timestamp-keyed merge instead
  of positional, and rerunning every PnL number/CI now the signal path is
  causally correct (currently historical, not evidence of anything).
- What I'd add for production: event-time streaming instead of batch
  replay, real queue position and transaction costs, months of untouched
  data instead of three selected weeks.
- The value isn't a profitable-strategy claim — it's the discipline of
  finding exactly where a pipeline like this breaks, and being honest
  about what's still unverified.

---

## 8. Anticipated hiring-manager probes

Earlier self-review flagged this as bug-heavy and light on the
classifier/ML work — both reworked since (Flagship #2 is invariant-first,
classifier stop covers scaling/model-choice/caching-boundary). Still true:
close doesn't mention detection performance; worth a real answer on where
test coverage's blind spots actually are.

- "You've told me things you found wrong. Tell me one thing you got right
  the first time — no audit needed."
- "Walk me through the classifier's three-model comparison — defend the
  walk-forward split, not just name it."
- "What fraction of toxic trades does this actually catch, at a tolerable
  false-alarm rate?"
- "How did you produce these numbers — anything AI-assisted? Pick a line
  in `bootstrap_eval.py`, explain why it's written that way."
- "Merge is positional with a row-count guard, timestamp-keyed join is the
  fix you'd still make — why not now?"
- "Gamma collapses to a constant — is inventory skew doing anything
  dynamic, or is it a fixed parameter dressed up as a control law?"
- "What's your test coverage story — not the bugs you found, the tests
  that would've caught them before interview prep, if any did?"
- "`microprice_minus_mid`'s normalization bug is in Reserve — what does it
  actually do to pooled model behavior on SOL vs. BTC? Checked?"
- "One more week before this interview — what would you actually fix vs.
  leave documented, and why that split?"
- "Data loading — what happens if `side` has an unexpected value, a typo
  or different casing?"
- "You benchmarked SortedDict and found it might be a wash for this exact
  workload — why keep the change instead of reverting?"
- "Neither of today's two bugs was caught by a test, because neither
  `main()` nor the CatBoost branch is exercised by the suite — where are
  the real blind spots?"
- "CatBoost doesn't load in your current environment — how confident are
  you the historical CatBoost numbers are still reproducible?"
