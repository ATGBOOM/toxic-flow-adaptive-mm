# Presentation Narrative

This is the talk written out as continuous prose, file by file, in the order
you'd actually present it — for reading through and internalizing the story,
not for glancing at while coding live (that's what `PRESENTATION_SCRIPT.md`
is for). Read this to know *why one thing leads to the next*; use the other
file to know *what to click on*.

---

## Problem statement

This pipeline reconstructs a crypto order book from raw exchange data,
derives a toxicity signal and a classifier from it, and backtests a
market-making strategy against it. The engineering problem running through
every single stage of it is maintaining a strict no-look-ahead guarantee —
because one violation anywhere silently inflates every result downstream of
it. That discipline is what the whole talk is actually about, and you'll see
it recur, in a slightly different form, at almost every stage that follows.

## Architecture diagram — `docs/presentation.html`

Raw Bybit data splits into two cleaning stages right at the start: trades
and the L2 order book, each cached separately. VPIN is computed directly off
the cleaned trades, in parallel with the book — not downstream of it. The
feature matrix is where everything meets: trailing trade features, the
forward toxicity label, VPIN, and book features all come together there,
and that merge point is also the caching boundary in the diagram — book
replay is expensive and deterministic, so it's computed once, while
everything below that line is cheap and gets iterated on, which is why
re-labelling takes seconds rather than a full replay. From the feature
matrix it goes to the classifier, then a durable predictions export, then
the causal backtest. The notebooks sitting off to the side are reporting
only — they're not part of the causal chain at all.

Every stage in that chain writes Parquet, not CSV and not a database.
Columnar and typed matters here specifically because a stage that only
needs three columns doesn't pay to read the rest, and dtypes survive
instead of round-tripping through strings the way CSV forces. A database
would be the reasonable alternative if this needed ad hoc querying, but
it's overkill for a batch pipeline where each stage just hands a typed
table to the next one — DuckDB directly over these same Parquet files would
be the natural next step if that ever changed.

One invariant gets enforced right at the very first entry point, before
anything else touches the data at all: `src/data/loader.py` sorts every
trade file by timestamp on load, and it does that defensively — sorting
rather than asserting-and-failing if the input is already out of order.
Everything downstream, the rolling windows, the VPIN bucketing, the
forward-return labels, silently assumes ascending time, and a research
batch pipeline should recover from a bad input rather than halt on it. That
sort was actually found unguarded during audit and is now guarded by a
monotonic-timestamp regression test. Two more things worth knowing about
that same file if pressed: gap detection there only prints a warning, it's
a smoke test, not gap handling, and the `side.map({"Buy": 1, "Sell": -1})`
call turns anything unexpected — a typo, different casing, a null — into
`NaN`, which then silently poisons every signed-volume feature downstream
with no validation catching it. That's the first instance of the no-look-
ahead discipline in the chain — everything else in the talk keeps coming
back to it.

## Flagship #1 — the order book, `src/features/reconstructor.py`

Let's go deep on the first real engineering decision — the order book.

The book itself is two `SortedDict`s, price to size, bids and asks. A
snapshot message clears both sides and reloads them from scratch, which is
the resync mechanism for a client that missed some deltas. A delta message
only touches the price levels it explicitly names — a nonzero size upserts
that level, a size of zero deletes it, and everything else in the book is
left completely untouched.

From that state, computing features means finding best bid and best ask,
deriving spread and midprice from them, and computing a microprice that
weights toward whichever side has less resting size — more bid depth pushes
the microprice toward the ask, because that's the side more likely to move
next. Depth imbalance gets computed at four levels, plus a pressure ratio
comparing the top five levels of depth against the top twenty-five, which
separates immediate top-of-book pressure from the broader visible book.

A regression test I wrote builds two different floating-point
representations of the exact same price and asserts they resolve to one
dictionary key. That test is what caught a real risk: without
canonicalizing price keys, representation drift could silently split one
real book level into two, or cause a size-zero delete to miss its target
because the key didn't match exactly. It's fixed now by rounding every
price key to eight decimal places before it ever touches either side of the
book, and the test guards that permanently.

The invariant that matters most in this whole file is causal ordering. The
replay loop samples the book for any pending trade *before* it applies the
next book message — strict less-than on timestamp, not less-than-or-equal.
A trade never sees a book message that arrived after its own. If that
invariant were ever violated, a feature would get built from book state the
market maker couldn't actually have observed yet, and the classifier would
look artificially good in testing while failing exactly where it matters,
live. It's enforced by the order the loop runs in, not by a filter checked
afterward, which is what makes it structural rather than something that
could be silently disabled by a refactor.

State also persists across day boundaries by construction. The `SortedDict`s
are created once, outside the day loop, and passed by reference into every
call — the book doesn't treat a calendar day as a real boundary, because it
isn't one in the market. It's just an artifact of how the historical data
happens to be chunked into files.

On the data structure itself: the operations actually needed are upsert,
delete, best bid and ask, and top-N depth. I started with a plain
dictionary — constant-time upsert and delete, but best bid, best ask, and
every depth level meant sorting the entire side from scratch on every
single feature call, which is O(L log L), paid once per sampled trade. I
replaced that with a `SortedDict`, which keeps prices ordered incrementally
as each update lands. Upsert and delete become O(log L), best bid and ask
become a direct lookup at either end of the structure instead of a full
resort, and depth sums just take a slice off either end, so nothing past
the first N entries ever gets touched. That's not a free upgrade, though,
and I benchmarked it rather than assumed it — at five hundred levels per
side, matching Bybit's real book depth, reads come out meaningfully faster
but each individual write is measurably slower than a plain dictionary's
constant-time insert. Whether that's a net win in aggregate depends on how
often you read relative to how often you write, and in this pipeline a
read happens once per trade while a write happens once per book message,
and messages vastly outnumber trades. Working the algebra on the measured
per-call costs, the sorted structure only wins in total time once reads are
at least roughly twelve percent as frequent as writes — for this specific
historical batch replay, the real net effect is genuinely uncertain without
the exact trades-per-day figure, quite possibly close to a wash. The
complexity argument becomes unambiguous the moment you imagine this running
live instead, where reads aren't tied to trade frequency at all and can
happen as often as the system wants to check the book.

There's also a tick-indexed array as an option I considered and didn't take
— a flat array indexed by price offset from a reference, divided by tick
size, gives true constant time everywhere, the fastest of any option. The
cost is a fixed architectural commitment: you need to know the tick size
and a bounded plausible price range up front, which is fine for one
instrument in a known band and a worse fit here, replaying historical data
across three differently-scaled assets without hard-coding that assumption
per asset.

One last design note on this file: the book itself is imperative and
stateful by necessity, because message replay is inherently sequential.
Everything downstream of it, all the feature engineering, is written as
pure functions over whole DataFrames instead, because those transforms are
vectorizable and don't need to remember anything beyond the current window.
Different paradigm per stage, matched to what each one actually needs, not
one style imposed everywhere for its own sake.

## The fast pass

That's the book. In parallel, off the same cleaned trades and not
downstream of the book at all, VPIN runs independently — from here I'll
move faster through the next few stages, though the depth is there if
asked about.

### VPIN — `src/models/vpin.py`

VPIN samples on a volume clock instead of calendar time. Fixed time
intervals overweight quiet periods and underweight real bursts of
activity, so a volume clock stays calibrated to how busy the market
actually is rather than to the clock. Mechanically, it accumulates trades
into fixed-volume buckets and takes the rolling mean absolute imbalance
between buy and sell volume within them. A bucket closes the instant it's
exactly filled, even mid-trade, and it gets timestamped by the trade that
filled it. This is the same causal discipline as the book, just applied on
a different axis — a rolling VPIN value only ever averages buckets that
completed strictly before it, so no partial or future bucket ever leaks
in.

The rolling window itself used to re-sum the whole window on every new
bucket, which is quadratic in the number of buckets. It's a single
cumulative-sum pass now, linear, and verified behavior-identical to the
original by regression test. Separately, I hand-compute VPIN against a
small synthetic sequence and check it against both the production
implementation and the parameter-sensitivity grid script — and that
comparison is exactly what caught the grid script quietly running a second,
drifted implementation of its own, one that kept a trailing partial bucket
the production path deliberately discards. Both call sites are unified onto
one canonical function now, so there's no way for them to diverge again.

### Feature matrix — `src/features/build_features.py`

The feature matrix merges cleaned trades, replayed book state, and VPIN
into one table per asset-week. Trailing windows are used for the trade
features, so nothing ever sees a future trade, and a strictly
forward-looking horizon is used for the label, so a row that doesn't have a
full future horizon available gets dropped entirely rather than clipped to
whatever price happened to be last. Those trailing windows use
`searchsorted` plus prefix sums instead of scanning each row individually —
that's logarithmic time per lookup instead of linear in the window size per
row, which genuinely matters at tens of millions of rows. The merge itself
that pastes book features onto trades is positional, guarded only by a
row-count check — if the two sides don't come out to the same length, that
whole day gets skipped rather than silently misaligned. A timestamp-keyed
join is the fix I'd still make here; it isn't done yet.

### Classifier — `src/models/classifier/data_loader.py` and `classifier.py`

Before getting to the model comparison itself: the loader adds three more
features at classifier load time — spread in basis points, a microprice
offset, and a normalized trade quantity — and it does that here rather than
upstream in the shared feature build. Cross-asset pooling needs
scale-invariant inputs, but that's a modeling decision specific to this
classifier, not a fact about the market itself, so it stays at the point of
consumption instead of getting baked permanently into the shared cache
every other consumer reads from. That's the same caching-boundary principle
as the book-replay split, just one layer further downstream.

The classifier itself compares three models behind one shared interface —
call `predict_proba`, get back a probability — which is a Strategy pattern.
The VPIN baseline needed its own hand-written class specifically because
nothing off-the-shelf could do what it needs: its `fit` method does
nothing at all, and its `predict_proba` just reads the raw VPIN column
straight out of the feature matrix and returns it as if it were a trained
model's output. Logistic regression and the boosted trees didn't need new
classes, because scikit-learn's and CatBoost's or XGBoost's classes already
behave exactly right natively — real training, real learned predictions —
so there was nothing to adapt. All three, the hand-written heuristic and
the two real models, get stored in the same dictionary and treated
identically by everything that compares them afterward; that uniform
interface is what makes genuinely different kinds of things interchangeable
to the orchestrator. Only logistic regression gets its inputs standardized,
fit on the training data only and never refit on test — it optimizes a
weighted sum of features through gradient descent, so features on wildly
different scales distort both the optimizer and the regularization penalty.
Trees split on raw thresholds, so scaling would be a complete no-op for
them.

I chose this three-model ladder over a neural network or a random forest
because the feature set here is only around fifteen tabular columns, and at
that scale gradient boosting already matches or beats deep learning with
far less tuning. The real question was never which model is theoretically
best — it was whether the added complexity of a nonlinear model actually
earns its keep over the raw hand-built signal, and whether it survives a
regime shift, which is exactly what the walk-forward comparison tests. And
that walk-forward split is the same no-look-ahead discipline again, applied
this time to model evaluation: train only on weeks strictly before the test
week, never mixed, so nothing from the future leaks into training either.
Training itself is capped at five hundred thousand rows, stratified to
preserve the true toxic prevalence rather than rebalanced to fifty-fifty,
because the full pooled set across all three assets is tens of millions of
rows and reruns need to stay fast. Test sets are never subsampled, so
evaluation stays unbiased.

### Predictions export — `src/evaluations/save_predictions.py`

Predictions get exported with asset, week, timestamp, and row identity
preserved — not just as good practice, but because that's exactly what
lets the backtest safely reattach a prediction to the trade it belongs to,
later. They're exported per-asset rather than pooled, deliberately, because
a paired statistical test comparing model configurations needs one
observation per asset, not one number blended across all three.

Memory management here is explicit, not incidental, and worth calling out
on its own: right after training, the code does `del split2` followed by
`gc.collect()`, and the same pattern — deleting the loaded frame and its
derived arrays, then forcing collection — runs at the end of every
per-asset loop iteration while generating predictions. Pooling weekly data
across three assets is a multi-gigabyte load, and this loop loads several
such frames back-to-back. Worth being honest about the actual
justification here rather than assuming one: I tested whether this pattern
was cleaning up genuine reference cycles, by reproducing the same
derive-columns-then-slice pattern and calling `gc.collect()` explicitly —
it found zero cyclic garbage, meaning refcounting alone had already freed
everything on its own. So the real reason to call it explicitly isn't that
refcounting misses this case; it's forcing collection to happen at a
chosen moment, defensively, rather than trusting Python's own automatic
allocation-count trigger to fire before the next multi-gigabyte load
starts. That's disciplined practice, not a proven leak in this code.

## Flagship #2 — the causal backtest, `src/evaluations/bootstrap_eval.py`

That's the pipeline up to a prediction. Now the second deep dive — what
actually happens with that prediction, and this is where the sharpest
engineering problem in the whole project lived.

The shape of this file, in order, is: load and align the predictions,
aggregate into one-second bars, run the strategy simulation, then bootstrap
that simulation many times over resampled days to get a confidence
interval, and finally package all of it into a reported table. The loading
step merges features and predictions by `source_row` identity, not
position, and cross-checks that the merged timestamp actually matches the
feature row's own timestamp — real defense against silent misalignment, not
just a convention. Bars are built by aggregating ticks into one-second
windows.

The simulation itself, `run_backtest`, is the core algorithm, and the way
it's built is a genuinely deliberate performance decision: everything that
doesn't depend on the running simulation state — the volatility estimate,
the fill size, the risk-aversion term, the spread — gets precomputed once
as numpy arrays, fully vectorized. Only cash and inventory are genuinely
sequential, because each bar's value depends on every fill that happened
before it, and those two run in a tight Python loop, nothing else does.
That split is what gets this down to about half a second to a second per
backtest run on six hundred thousand bars, versus roughly thirty seconds
running the same thing with `iterrows`, and it matters because the
bootstrap reruns this hundreds of times per asset-week.

Mechanically, it widens the quoted spread as the toxicity signal rises, and
skews the quote by current inventory — heavier inventory pushes the quote
to encourage trading it back down. A sell aggressor fills the bid if its
price reaches it, a buy fills the ask the same way, and mark-to-market
values the current inventory at today's price.

The invariant this whole module exists to protect is that state and signal
known at the end of bar t can only ever place quotes for bar t plus one,
and only trades in bar t plus one or later can fill those quotes — a
realized label is never, under any circumstance, a valid decision signal.
That's enforced three separate ways, not just asserted and hoped for.
Structurally, the loop indexes state and signal at the previous bar while
checking fills against the current one, so the one-bar offset is built
into the indexing itself, not a convention someone has to remember. At the
interface level, the function requires an explicit named prediction column
and hard-fails with a KeyError if it's missing, so there's no code path
where a realized label could get passed in as the signal by accident. And
by test, two synthetic tests guard this directly — one builds bars where
the realized label and the prediction deliberately point in opposite
directions and checks that only the prediction ever drives the quote, and
one builds a bar where a fill-capable trade arrives in the same bar the
quote was set and checks that the fill gets rejected, since a fill is only
ever valid on a strictly later bar than the one that produced the quote.

This replaced an earlier version that enforced none of that — it computed
a bar's quote from that same bar's realized, forward-looking label, which
is exactly the class of error those two tests are now shaped to catch if
it ever creeps back in. The earlier `build_bars` computed each bar's toxic
rate directly from the realized label, and the backtest used that same
bar's value as its own signal — the strategy's signal was, literally, the
future. The notebook's own text claimed the previous bar was used; the code
never actually shifted anything. And even a naive one-bar shift wouldn't
have been enough to fix it, because the label's own horizon is ten seconds,
longer than a single one-second bar, so the previous bar's label still
depends on prices up to nine seconds after the current bar begins — which
is why the fix had to be a real redesign rather than a one-line shift. One
more concrete fix sits right in the code as a comment: the original
notebook credited an ask fill at the bid price instead of the ask price, a
one-line accounting error invisible unless you're checking cash flow
against the actual economics of the trade rather than just watching the
code execute without error.

What's still genuinely open: the confidence intervals use a block bootstrap
over daily blocks, but resampled days get concatenated into one continuous
simulated path with inventory carried straight across them, which creates
artificial transitions between days that were never actually adjacent,
including backward timestamp jumps at the block boundaries. And the
historical PnL table hasn't been rerun since the causal fix went in, so
it's reported as a historical, oracle-signal result, not evidence of what
the corrected strategy would actually do.

## Proving it — `tests/test_core.py`

Before wrapping up, every invariant just described has a synthetic test
behind it, not just a description, and they're worth running live. All
fourteen tests run in well under a second, entirely synthetic, no external
data needed. Six of them map directly onto claims made earlier in this
exact talk: the loader's sort invariant, the float-key canonicalization
fix, the book replay's causal ordering, the no-lookahead guarantee on
trailing features, and the two tests behind the backtest's invariant —
target-signal separation and same-bar fill rejection. The other eight cover
things like PnL arithmetic in isolation, inclusive trailing-window
boundaries, label-horizon validity, and identity preservation through the
classifier loader. What's genuinely not tested yet: order-book timestamp
ordering across days, data gaps inside an otherwise-valid label horizon,
prediction-to-bar alignment beyond the merge's own identity check,
inventory limits and two-sided same-bar fills, and a full deterministic
end-to-end synthetic integration test.

## Close

To close out: the two things I'd still fix are a timestamp-keyed merge
instead of the positional one, and rerunning every PnL number and
confidence interval now that the signal path is causally correct — right
now they're historical and shouldn't be read as evidence of anything the
classifier did. What I'd add for production is event-time streaming
instead of batch replay, real queue position and transaction costs, and
months of untouched out-of-sample data instead of three selected weeks.
The value here was never a profitable-strategy claim — it's the discipline
of finding out exactly where a pipeline like this breaks, and being honest
about what's still unverified.
