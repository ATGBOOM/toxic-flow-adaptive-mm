# Presentation Script — Final Structure

Engineering-led, ~16–19 min. No cold open. Two flagship deep-dives carry the
weight; everything else is a short, tradeoff-flavored pass — a few sentences,
not a single line — with the deeper material held for Q&A rather than spoken
in the walkthrough itself.

Structure: problem statement → architecture diagram → Flagship #1
(`reconstructor.py`) → fast pass (VPIN, feature matrix, classifier,
predictions export) → Flagship #2 (`bootstrap_eval.py`) → close.

Each section below has **Say** (the actual talk track) and **Reserve** (not
spoken — pulled out only if the interviewer asks). This mirrors how the two
flagships work: don't front-load everything, hold depth for defense.

Known doc staleness, for your own awareness only, never say this unprompted:
`AUDIT_FINDINGS.md` and `CODE_WALKTHROUGH.md` both predate the `a0ca776`
hardening commit — they still describe `vpin_robustness.py` as containing a
drifted duplicate VPIN implementation and claim an exact-fill bucket doesn't
flush until the next trade. Neither is true of the current code (verified
directly against `vpin_robustness.py` and `vpin.py`). If asked about either,
answer from the current code, not the older docs.

---

## 1. Problem statement (~30–45s)

**Say:**

> This pipeline reconstructs a crypto order book from raw exchange data,
> derives a toxicity signal and classifier from it, and backtests a
> market-making strategy against it. The engineering problem throughout was
> maintaining a strict no-look-ahead guarantee at every single boundary —
> because one violation anywhere silently inflates every result downstream of
> it. That discipline is what this talk is actually about.

---

## 2. Architecture diagram (~60–90s)

**Say**, walking the diagram left to right:

> Raw Bybit data splits into two cleaning stages — trades and the L2 order
> book — each cached separately. VPIN comes off cleaned trades. The feature
> matrix is the merge point: trailing trade features, the forward toxicity
> label, VPIN, and book features come together here — and it's the caching
> boundary. Book replay is expensive and deterministic, compute once;
> everything downstream is cheap and iterated on, so re-labelling is
> seconds, not a replay. From there: classifier, a durable predictions
> export, then the causal backtest. Notebooks are reporting only, not part
> of the causal chain.
>
> Every stage writes Parquet, not CSV or a database. Columnar and typed — a
> stage needing three columns doesn't pay to read the rest, and dtypes
> survive instead of round-tripping through strings. A database is the real
> alternative, reasonable for ad hoc querying, overkill for a batch
> pipeline where each stage just hands a typed table to the next; DuckDB
> over these same files is the natural next step if that changed.

---

## 3. Flagship #1 — `src/features/reconstructor.py` (~3–4 min)

**Say:**

> The book is two SortedDicts, price to size — bids and asks. A snapshot
> clears both and reloads from scratch, the resync mechanism for a client
> that missed deltas. A delta touches only the levels it names: nonzero
> size upserts, zero deletes, everything else untouched.
>
> From that state: best bid and ask give spread and midprice; microprice
> weights toward whichever side has less resting size, since more bid depth
> pushes it toward the ask — that's the side more likely to move next. I
> compute depth imbalance at four levels and a pressure ratio comparing
> top-five to top-25 depth per side, separating immediate top-of-book
> pressure from the broader visible book.
>
> I added a test that builds two floating-point representations of the same
> price and asserts they resolve to one dictionary key. That's what caught
> the real risk: uncanonicalized keys could split one book level into two,
> or miss a size-zero delete on a near-match. Fixed by rounding every price
> key to eight decimal places before it touches either dict — the test
> guards it now.
>
> The invariant that matters most: the replay loop samples the book for any
> pending trade before applying the next message — strict less-than on
> timestamp. A trade never sees a book message after its own. Violate that
> once and a feature is built from book state the market maker couldn't
> have observed yet — the classifier looks good in testing and fails
> exactly where it matters, live. It's enforced by loop order, not a filter
> checked afterward, so it can't be silently bypassed.
>
> State persists across day boundaries by construction — the SortedDicts are
> created once, outside the day loop, passed by reference. The book doesn't
> treat a day as a real boundary, because it isn't one; it's just how the
> historical files happen to be chunked.
>
> The operations needed are upsert, delete, best bid/ask, top-N depth. I
> started with a plain dict — O(1) upsert and delete, but best bid/ask and
> every depth level meant sorting the entire side from scratch on every
> single feature call, O(L log L). I replaced it with a SortedDict, which
> keeps prices ordered incrementally as each update lands: upsert and
> delete become O(log L), and best bid/ask is a direct lookup at either end
> of the structure instead of a full resort — depth-N sums just take a
> slice off either end, so nothing past the first N entries gets touched at
> all.
>
> That's not a free upgrade, and I benchmarked it rather than assumed it:
> at 500 levels per side, matching Bybit's real book depth, reads come out
> meaningfully faster, but each individual write is measurably slower than
> a plain dict's O(1) insert. Whether that's a net win depends on how often
> you read relative to how often you write — and in this pipeline, a read
> happens once per trade while a write happens once per book message, and
> messages outnumber trades by a wide margin. I worked out the actual
> breakeven point rather than wave that away.
>
> One design note: the book is imperative and stateful by necessity — it's
> inherently sequential, message by message. Everything downstream, the
> feature engineering, is pure functions over whole DataFrames instead,
> because those transforms are vectorizable and don't need memory beyond
> the current window. Different paradigm per stage, matched to what each
> one actually needs.

**Reserve (Q&A only):**

- Crossed book is unhandled — if best bid ≥ best ask, spread goes negative,
  undetected.
- A missing order-book day carries stale state into the next day until the
  next snapshot resyncs it. Known, still open.
- `namelist()[0]` assumes exactly one inner file per zip, unguarded.

**Reserve — the full data-structure decision (dict considered, SortedDict
chosen), Q&A only:**

- **Why I moved off dict:** O(1) upsert/delete, but best bid/ask and every
  depth level required sorting the whole side from scratch on every single
  feature call — O(L log L), paid once per sampled trade. That was
  genuinely the dominant cost in the whole feature build before the swap.
- **SortedDict** (`sortedcontainers.SortedDict`) — keeps prices ordered
  continuously via an internal sorted structure. Upsert/delete: O(log L).
  Best bid/ask: O(log L) via `peekitem` at either end, instead of a full
  O(L log L) resort. Depth-N: a slice off either end, O(N) with small
  constant N, nothing past it touched. No sorted structure in the Python
  standard library, so this is a real third-party dependency
  (`sortedcontainers`), not a stdlib swap — genuine added complexity a
  plain dict doesn't carry.
- **The honest benchmark, not assumed:** at 500 levels per side (Bybit's
  real ob500 depth), measured directly — reads are about 1.6x faster with
  SortedDict; individual writes are about 6.7x slower than a plain dict's
  O(1) insert. Net effect depends entirely on the actual read:write ratio.
  Working the algebra on the measured per-call costs: SortedDict only wins
  in aggregate once reads are at least roughly 12% as frequent as writes.
  In this pipeline, a read happens once per trade and a write happens once
  per book message, and messages (~863k/day per asset, per the README)
  vastly outnumber trades — so for this specific historical batch replay,
  the real net effect is genuinely uncertain without the exact trades/day
  figure, quite possibly close to a wash. The complexity argument is
  unambiguous for a *live* system, where reads happen far more relative to
  writes than in a one-time batch reconstruction — that's the honest scope
  of the claim, not "it's just faster."
- **Tick-indexed array** — the structure I didn't move to, and why: a flat
  array indexed by (price − reference) / tick size gives true O(1)
  everywhere, the fastest of any option. The cost is a fixed architectural
  commitment — you need to know the tick size and a bounded plausible
  price range up front. Fine for one instrument in a known band; a worse
  fit here, replaying arbitrary historical data across three
  differently-scaled assets without hard-coding that per-asset.

---

## 4. Fast pass (~3–4 min total)

**Say**, a few sentences per stop, tradeoff-flavored:

> VPIN samples on a volume clock, not calendar time — fixed time intervals
> overweight quiet periods and underweight real bursts, so a volume clock
> stays calibrated to actual activity. It accumulates trades into
> fixed-volume buckets and takes the rolling mean absolute buy/sell
> imbalance; a bucket closes the instant it's exactly filled, timestamped
> by the trade that filled it. The rolling window is a cumulative sum in
> one pass — the original re-summed the whole window per bucket, which is
> quadratic; the cumsum version is linear, verified behavior-identical by
> regression test. I also hand-compute VPIN against a synthetic sequence
> and check it against both the production implementation and the
> parameter-grid script — that comparison is what caught the grid script
> running a second, drifted implementation that kept a trailing partial
> bucket production discards. Unified onto one canonical function.
>
> The feature matrix merges trades, book state, and VPIN — trailing
> windows for features, forward-only horizon for the label, so a row
> without a full future horizon gets dropped, not clipped. Trailing windows
> use searchsorted plus prefix sums instead of a per-row scan — O(log n)
> per lookup instead of O(window size) per row, which matters at tens of
> millions of rows. The merge itself is positional with a row-count guard —
> mismatched days get skipped entirely rather than silently misaligned; a
> timestamp-keyed join is the fix I'd still make.
>
> Before the model comparison: the loader adds three more features —
> spread in basis points, microprice offset, normalized quantity —
> specifically at the classifier's load time, not upstream in the shared
> feature build. Cross-asset pooling needs scale-invariant inputs, but
> that's a modeling decision, not a fact about the market, so it stays at
> the point of consumption instead of getting baked into the shared cache
> every other consumer reads from — same caching-boundary principle as the
> book-replay split, one layer downstream.
>
> The classifier compares three models behind one interface — call
> `predict_proba`, get a probability — a Strategy pattern. VPIN's `fit`
> does nothing at all, purely so it sits in the same evaluation loop as the
> two real sklearn models without special-casing — an adapter, making a
> heuristic that was never trained conform to an interface built for models
> that are. Only logistic regression gets its inputs standardized, fit on
> train only, never refit on test — it optimizes a weighted sum via
> gradient descent, so features on wildly different scales distort both the
> optimizer and the L2 penalty; trees split on raw thresholds, so scaling
> would be a no-op for them. I chose this three-model ladder over a neural
> net or random forest because the feature set is around fifteen tabular
> columns, where gradient boosting already matches or beats deep learning
> with far less tuning — the real question wasn't "what's the best possible
> model," it was whether the added complexity earns its keep over the raw
> signal and survives a regime shift, which is what the walk-forward
> comparison tests. Training is capped at 500k rows, stratified to preserve
> true toxic prevalence rather than rebalanced to 50/50, because the full
> pooled set is tens of millions of rows and reruns need to stay fast; test
> sets are never subsampled, for an unbiased read.
>
> Predictions export with asset, week, timestamp, and row identity
> preserved — not just good practice, it's what lets the backtest safely
> reattach a prediction to its trade, which is next. Exported per-asset
> rather than pooled, deliberately: a paired statistical test comparing
> model configurations needs one observation per asset, not one blended
> number across all three.

**Reserve (Q&A only):**

- VPIN: bucket closes immediately on an exact fill, timestamped by the
  filling trade — this *is* correct in the current code; don't let anyone's
  older notes about a delayed-flush bug stand, it's fixed.
- Feature matrix: `daily_vol / 7` in `add_vpin_feature` is hardcoded and
  doesn't account for weeks with missing days (ETH/SOL week 3) — still open.
- Classifier: `spread_bps` and `qty_normalised` are properly cross-asset
  normalized (computed per-asset, before pooling); `microprice_minus_mid` is
  **not** — it's a raw dollar difference, not divided by midprice. Real,
  still-open normalization gap, and it directly undercuts the "pooling
  requires per-asset-invariant features" argument for that one feature
  specifically. Own this if asked, don't paper over it.
- Classifier: training is stratified to preserve true toxic prevalence, not
  rebalanced to 50/50; test sets are never subsampled, for an unbiased read.
- Predictions export: the saved `.joblib` model artifacts aren't currently
  loaded back anywhere in the repo — a live-scoring path is exactly what
  would consume them.

---

## 5. Flagship #2 — `src/evaluations/bootstrap_eval.py` (~4–5 min)

**Say:**

> This is the causal backtest. Adaptive spread is the market spread times
> one plus a multiplier times the toxicity signal; skew is negative
> inventory times a risk-aversion term times volatility times price — so
> heavier inventory or higher predicted toxicity both widen the quotes or
> shift them to protect against getting run over. A sell aggressor fills
> the bid if its price reaches it, a buy fills the ask the same way,
> mark-to-market is cash plus inventory times current mid.
>
> This is vectorized for performance: sigma, fill size, gamma, and spreads
> are precomputed as numpy arrays, and only cash and inventory — the
> genuinely path-dependent part — run in a tight Python loop. That's about
> half a second to a second per backtest on 600k bars, versus roughly
> thirty seconds with `iterrows()` — it matters because the bootstrap runs
> this hundreds of times per asset-week.
>
> The invariant this whole module exists to protect: state and signal known
> at the end of bar `t` can only place quotes for bar `t+1`, and only
> trades in bar `t+1` or later can fill those quotes. A realized label is
> never a valid decision signal, under any circumstance. That's enforced
> three separate ways, not just asserted once and hoped for. First,
> structurally in the code: the loop indexes state and signal at `i-1`
> while checking fills against bar `i`, so the one-bar offset is built into
> the indexing itself, not a convention someone has to remember to follow.
> Second, at the interface: the function requires an explicit named
> prediction column and hard-fails with a KeyError if it's missing, so
> there's no code path where a realized label could get passed in as the
> signal by accident. Third, by test: two synthetic tests guard this
> directly — one builds bars where the realized label and the prediction
> deliberately point in opposite directions and checks that only the
> prediction ever drives the quote; one builds a bar where a fill-capable
> trade arrives in the *same* bar the quote was set, and checks that the
> fill gets rejected, since a fill is only valid on a strictly later bar
> than the one that produced the quote.
>
> This replaced an earlier version that didn't enforce any of that — it
> computed a bar's quote from that same bar's realized, forward-looking
> label, which is exactly the class of error those two tests are shaped to
> catch if it ever creeps back in.
>
> What's still open, honestly: the confidence intervals use a block
> bootstrap over daily blocks, but resampled days get concatenated into one
> continuous path with inventory carried straight across them — that
> creates artificial transitions between days that were never actually
> adjacent, including backward timestamp jumps at the block boundaries. And
> the historical PnL table hasn't been rerun since the causal fix went in,
> so it's reported as a historical, oracle-signal result — not evidence of
> what the corrected strategy would actually do.

**Reserve — the full history behind the invariant, if asked "was there a bug
here" or "how did you find it":**

- The earlier version's `build_bars()` computed each bar's `toxic_rate`
  directly from the realized, forward-looking label, and the backtest used
  that *same bar's* value to set its own spread — the strategy's signal
  was, literally, the future. The notebook's own text claimed the previous
  bar was used; the code never actually shifted anything — a real
  claim-vs-code mismatch, not just an oversight in the math.
- Even a naive one-bar shift wouldn't have been enough to fix it: the
  label's own horizon is ten seconds, longer than a single one-second bar,
  so the *previous* bar's label still depends on prices up to nine seconds
  after the current bar begins. That's why the fix had to be a real
  redesign (named prediction column, `i-1`/`i` indexing, hard failure on a
  missing signal) rather than a one-line shift.
- One more concrete fix sitting right in the code as a comment: the
  original notebook credited an ask fill at the bid price instead of the
  ask price — a one-line accounting error that's invisible unless you're
  checking cash flow against the actual economics of the trade, not just
  running the code and watching it execute without error.

**Reserve (Q&A only):**

- `fill_size = alpha * market_spread / (sigma * mid)` is dimensionally
  dollars over (dimensionless times dollars) — dimensionless — but it's
  used as if it were a quantity of the base asset. Not dimensionally
  anchored to a real risk budget.
- `gamma` is defined via a formula referencing spread, fill size, sigma, and
  mid, but algebraically those all cancel — it collapses to `1 / (2 * M *
  alpha)`, a constant depending only on two fixed hyperparameters, despite
  looking dynamic.
- The inventory limit check happens *before* a fill is applied, so inventory
  can overshoot the stated limit by up to one fill size.
- `toxic_fill_rate()` detects fills via `inventory.diff()`; if both a buy
  and sell fill land in the same bar and net to zero inventory change, that
  bar's fills are invisible to it — undercounts.
- The prediction/feature merge uses `validate="one_to_one"` on the
  `source_row` join, plus an explicit timestamp cross-check that raises if
  the merged prediction's timestamp doesn't match the feature row's own —
  real defense against silent misalignment, not just a convention.
- `kappa` in this code is trades-per-bar, not the Avellaneda-Stoikov
  fill-intensity sensitivity to quote distance — this is A-S-inspired, not
  a calibrated A-S solution.

---

## 6. Close (~60–90s)

**Say:**

> The two things I'd still fix are a timestamp-keyed merge instead of the
> positional one, and rerunning every PnL number and confidence interval now
> that the signal path is causally correct — right now they're historical
> and shouldn't be read as evidence of anything the classifier did. What I'd
> add for production: event-time streaming instead of batch replay, real
> queue position and transaction costs, and months of untouched
> out-of-sample data instead of three selected weeks. The value here isn't a
> profitable strategy claim — it's the discipline of finding out exactly
> where a pipeline like this breaks, and being honest about what's still
> unverified.

---

## 7. Anticipated hiring-manager probes

Self-review flagged this script as bug-heavy (almost everything is framed as
"found X, fixed X"), light on the classifier/ML work (one sentence for
genuinely substantial modeling decisions), missing the scale/memory
engineering material entirely, and missing any economic reality check in the
close. Have real answers ready for these, not just the flagship material:

- "You've told me four things you found wrong. Tell me one thing in this
  codebase you got right the first time and would build the same way
  again — no audit needed."
- "Walk me through the classifier's three-model comparison — I want to see
  you defend the walk-forward split, not just name it."
- "Your close doesn't mention detection performance. What fraction of toxic
  trades does this actually catch, at a threshold where the false-alarm
  rate is tolerable?"
- "How did you produce the numbers in this project — anything AI-assisted?
  Pick any line in `bootstrap_eval.py` and explain exactly why it's written
  that way."
- "You said the merge is positional with a row-count guard, and a
  timestamp-keyed join is the fix you'd still make. Why didn't you just
  make it now?"
- "Your gamma formula collapses to a constant — so is inventory skew
  actually doing anything dynamic in this strategy, or is it effectively a
  fixed parameter dressed up as a control law?"
- "What's your test coverage story overall — not the bugs you found, the
  tests that would have caught them *before* interview prep, if any did?"
- "You dropped `microprice_minus_mid`'s normalization bug into the reserve
  material — if that's broken, what does it actually do to the pooled
  model's behavior on SOL versus BTC? Have you checked?"
- "If I gave you one more week before this interview, what would you
  actually go fix, versus what would you leave as a documented
  limitation — and why that split?"
