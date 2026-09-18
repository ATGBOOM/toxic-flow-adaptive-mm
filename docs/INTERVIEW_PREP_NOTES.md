# Interview Prep Notes — Stage 1 Repository Mastery

Working notes for the 45-min project presentation + code defence. Organised by
module in dataflow order. Each entry = what it does (correct wording), the
landmines an interviewer will hit, and how to defend it.

---

## Group 1 — Data source & trade cleaning (`src/data/loader.py`, `src/data/pipeline.py`)

### What it does
- **`loader.load_trades`** adapts one raw Bybit `.csv.gz` file to our schema:
  - Reads only `timestamp, side, size, price`.
  - `side → sign`: `Buy → +1`, `Sell → −1`; drops `side`.
  - `size → qty` rename.
  - Converts timestamp **from Unix seconds (float) → `datetime64[ns]`** via
    `pd.to_datetime(unit="s")` (millisecond precision survives).
  - Gap smoke-test: warns (prints) when consecutive `timestamp.diff() > 60s`.
  - Returns `[timestamp, price, qty, sign]`.
- **`pipeline.process_all`**: for each asset dir, parse date from filename,
  bucket days into `week1/2/3` via `date_to_week` (string range compare), load
  each day with `load_trades`, `concat` per week, write
  `data/processed/<asset>/<week>.parquet`.

### Corrections / precise wording
- **Timestamp direction:** raw is **Unix seconds** (Bybit's format, not our
  choice); we convert *up* to a pandas datetime. Downstream we go back to
  **int64 ms** (`astype("int64") // 1e6`) for fast `searchsorted`.

### Landmines (own these before they're asked)
1. **Sort invariant.** `diff()` gaps + all downstream `searchsorted`/rolling
   windows assume **ascending time**. Out-of-order input → silently wrong
   features, no error. **FIXED:** `loader.load_trades` now
   `sort_values("timestamp")` after the datetime conversion, guarded by
   `tests/test_core.py::test_load_trades_timestamps_are_monotonic`.
   *Design choice to defend:* we **sort** (defensive, silently reorders) rather
   than **assert-and-fail** (loud). Sort is the safer default for a research
   batch — say it's deliberate.
2. **`sign` mapping fails silently.** `.map({"Buy":1,"Sell":-1})` turns any
   other/None/mis-cased value into `NaN`, poisoning signed-volume features. No
   validation.
3. **Gap check only prints.** No drop/split/logged artifact; in a batch run the
   warning is lost. It's a smoke-test, not gap *handling* — say so.

### Instincts, refined
- **Not a per-row dataclass** — that kills vectorised pandas at 1M+ rows. Want a
  **schema/dtype contract at the DataFrame boundary** (assert columns+dtypes, or
  `pandera`). Phrase: "schema contract, not row objects."
- The "assume chronological" point is landmine #1 — state it as a hard
  invariant, not a footnote.

### Efficiency (honest answer)
- Not maximally efficient: holds a whole asset's daily frames in memory then
  `pd.concat` (full copy). Fine at this scale.
- Defensible framing: **one-time batch → optimise for correctness + simplicity;
  the parquet cache makes expensive downstream stages cheap to re-run.**
- If it had to scale: write **per-day parquet** + lazy dataset scan
  (`pyarrow.dataset`/DuckDB) instead of concat-in-RAM. Filename parsing should
  be a **regex on the date**, not `str.replace`. *(regex fix being added.)*

---

## Group 2 — Order-book mechanics (`src/features/reconstructor.py`: `apply_update`, `compute_book_features`)

### What it does
- **`apply_update(bids, asks, message)`**: book = two dicts `price→size`,
  mutated in place.
  - `type == "snapshot"` → **clear both sides, then reload** (this is the
    **resync/recovery** mechanism — a client that dropped a delta recovers on
    the next snapshot).
  - `type == "delta"` → **upsert only the named price levels**; unnamed levels
    persist untouched. `size == 0` → `pop(price, None)` (level removed —
    cancelled or filled; safe no-op if absent).
- **`compute_book_features(bids, asks)`**: returns `None` if **either side is
  empty** (replay loop drops those rows). Otherwise **sorts all price keys** →
  best bid/ask, then spread, midprice, microprice, depth_imbalance at 1/5/10/25,
  bid/ask pressure, pressure_imbalance.

### Precise wording
- Delta = **upsert of named levels**, not "replace the side."
- Snapshot = **resync**, not just "replace all."

### Landmines
- **Crossed book unhandled.** If `best_bid ≥ best_ask`, spread goes negative and
  nothing detects it — volunteer this as a known limitation.
- **Re-sorts every call.** `sorted(keys)` on every trade sample → the dominant
  cost of the whole feature build (see complexity below), *not* the message
  replay.
- **Float price keys** — theoretical drift could split one real level into two.
  **FIXED:** key canonicalized via `round(float(price), 8)` in `apply_update`,
  guarded by a test. Production form = integer ticks (below).

### Data-structure tradeoff (prepare this — the "why a dict?" question)

Operations needed: **upsert**, **delete**, **best bid/ask**, **top-N**.

| Structure | upsert / delete | best bid/ask | top-N | Note |
|---|---|---|---|---|
| **`dict` price→size** (chosen) | O(1) | **O(L log L)** (sort every call) | O(L log L) | Simple + correct; the sort per trade is the cost |
| **Sorted container** (SortedDict / balanced BST / skiplist) | O(log L) | O(1) (ends) | O(N) | The natural upgrade — cheap best/top-N |
| **Two heaps** (max bids / min asks) | O(log L) push | O(1) | hard | Arbitrary-level **cancels are awkward** (lazy deletion) → poor fit for a LOB |
| **Tick-indexed array** (`price/tick → idx`) | O(1) | O(1) w/ best pointers | O(N) | What real matching engines use; huge memory if price range is wide/sparse |

- **Our choice + why:** `dict` + full sort — simplicity and correctness for a
  **one-time batch** that only samples per trade; the re-sort cost is tolerated.
- **Complexity headline:** feature build is **O(T · L log L)** (T trades ×
  sorting L levels), not O(messages).
- **Production answer:** maintain a **sorted structure or tick-indexed array
  incrementally**, tracking best bid/ask on each message so features are O(1).

### Fix applied during prep (working tree, not committed)
- `reconstructor.py`: `apply_update` canonicalizes price keys (`round(float(price), 8)`).
- `tests/test_core.py`: test that an insert with float-representation noise is deleted correctly.

---

## Group 3 — Book replay & weekly driver (`reconstructor.py`: `reconstruct_and_extract_from_state`, `process_week`)

### What it does
- **`reconstruct_and_extract_from_state(ob_zip_path, trades_df, bids, asks) → DataFrame`**:
  streams one day's zipped NDJSON book file and samples the book at each trade.
  - **Streaming read:** `ZipFile → namelist()[0] → zf.open → for line in f → json.loads`.
    Decompress-one-line-discard → near-constant memory on multi-GB files. Single
    forward pass in ts order.
  - **The causal loop:** for each book message, flush every pending trade with
    `trade_ts < book_ts`, sampling the **current (pre-update)** book, *then*
    `apply_update`. Net invariant → **a trade sees every book message with
    `ts ≤ its own ts`, nothing after.**
  - **`<` (strict) is the same-millisecond rule:** a trade at exactly a message's
    ts is sampled *after* that message applies (included). `<=` would exclude it.
    No strictly-future leakage either way — that's what protects the label.
  - **Returns a `pd.DataFrame`, not an array.** `compute_book_features` returns
    `None` when a side is empty → that trade produces **no row** → `len(out) ≤
    len(trades)`. This is *why* the merge later filters to `ts ≥ first snapshot`.
  - **Tail loop:** trades after the last message get the **frozen final book**
    (stale state).
- **`process_week(...)`**: loops the week's dates; per day selects the book zip
  **by filename** (`<date>_<asset>_ob500.data.zip`), masks the **weekly trades**
  to `[midnight, next-midnight)` (half-open, `<` end), calls the replay, writes
  **one parquet per day** (`<asset>_<date>_book_features.parquet`).

### Precise wording (corrections to own)
- The day boundary comes from the **book filename**, not from masking the book.
  We mask **trades**, not the order book.
- **The book is stateful across days by design.** `bids={}` / `asks={}` are
  created **once before the day loop** and passed by reference into every call —
  so day N+1 opens on day N's **closing book** (the point of `..._from_state`).
- Output is **per-day**, not per-week.

### Landmines
1. **`namelist()[0]` is an unguarded assumption** — reads whatever is first;
   assumes exactly one inner file. Harden with `assert len(namelist())==1` or a
   manifest check.
2. **Missing day (`if not os.path.exists: continue`)** — README: ETH/SOL week 3
   miss days. Two failure modes at once: (a) that day's **trades produce no rows**
   (silently gone), and (b) the persistent book **carries stale state across the
   gap** — the resumed day opens on a book 2+ days old until the first snapshot
   resyncs it; combined with strict-`<`, the earliest resumed trades sample a
   badly stale book. One-liner: *"stateful book by design; failure mode is a
   missing day — stale carryover + dropped trades, mitigated only by the next
   snapshot resync."*
3. **Stale tail** — final-message-to-last-trade gap freezes one book state.

### Probes to pre-load
- "Two book messages at the same ms?" → both applied in stream order before any
  trade at/after them is sampled.
- "A trade before the first message?" → book empty → `None` → dropped.
- "Where's the cost?" → the `sorted(keys)` inside `compute_book_features`, run
  per flushed trade → O(T · L log L), dominated by trades not messages.

---

## Group 4 — Feature matrix (`features/build_features.py`)

### Data lineage — the two inputs are NOT duplicates
Two files describe the **same trades from two sides** (both one-row-per-trade):
- **Weekly trades parquet** (`trades_path`, from Stage 1) → the *trade* side:
  `timestamp, price, qty, sign`.
- **Per-day book-feature parquets** (from Stage 2 `process_week`) → the
  *order-book* side: `spread, mid, microprice, depth_imbal…, timestamp`.

`build_full_features` **does not recompute book features** — it reads the cached
per-day files and pastes them on. (The `# step 1: run book reconstruction if not
already done` comment is **stale** — the code only reads trades; reconstruction
must have been run separately.)

The trades parquet is read **twice**: reconstructor needed only the trade
**timestamps** (when to snapshot the book); build_features needs the full trade
rows to attach labels/features and merge. The only real column overlap is
`timestamp`, used for the alignment filter then skipped in the paste.

### Why book features are a separate stage (own this)
The cache boundary is drawn between **expensive+frozen** and **cheap+iterated**:
- Book reconstruction = O(T·L log L), **minutes/asset-week** (code times it per
  day), and **deterministic** — compute once, ever.
- Trade features / toxic label / VPIN = vectorised, **seconds**, and the things
  you *tune* (threshold, horizon, windows, bucket size).

One monolithic pass would put the cache at the final output → tweaking the label
threshold would force a **full book replay**. Splitting means re-labelling is
seconds, not hours. Secondary wins: **per-day parquets = fault isolation /
resumability**; the two stages have different execution shapes (streaming stateful
replay vs whole-week-in-memory rolling windows). One-liner: *"book features are an
expensive deterministic compute-once artifact; labels are cheap and I iterate on
them — caching at that boundary makes re-labelling seconds not a replay."*

### The merge (step 2) — landmine
Per day: slice trades to `[midnight, next-midnight)` → **drop trades before the
first book snapshot** (`ts >= book_day.timestamp.iloc[0]`) → **`if len(day_trades)
!= len(book_day): skip the whole day`** → paste book cols by **`.values`**
(pure positional, no timestamp join).
- The length-guard **silently drops an entire day** on any mismatch.
- Positional paste is correct **only because** both passes slice/order trades
  identically — nothing enforces it. A **timestamp-keyed join** removes this
  silent-drift risk (this is slide-3 "merge robustness").

### The feature adders — causality (interviewer will hammer this)
- **`add_trade_features`** — *trailing* windows via `searchsorted` on sorted `ts`;
  inclusive both ends; no future trade. `trade_intensity` = count in
  `[ts-w, ts]`; `volume_acceleration` = last-5s vol ÷ (last-30s vol prorated to
  5s), fallback 1.0 when denom 0; `signed_vol_imbalance_10s` = Σ(sign·qty) over 10s.
- **`add_toxicity_label`** — *forward-looking* 10s; `valid` mask NaNs trades
  without a full horizon (tail of day unlabelled); **sign-aware**: buy toxic if
  `fwd>+8bps`, sell toxic if `fwd<-8bps` (adverse to the maker). Label is
  **nullable boolean** (`True/False/pd.NA`) so unlabelled ≠ False → `dropna`
  before training, else fake negatives poison class balance.
- **`add_vpin_feature`** — forward-fill join: `searchsorted(vpin_ts, trade_ts,
  "right") - 1` = most recent completed VPIN ≤ trade; `np.clip(...,0,len-1)`
  stops the `-1` from **negative-wrapping to the last (future) value**; trades
  before the first bucket set to NaN.

### Landmine — `daily_vol / 7` (real bug)
`add_vpin_feature` computes `daily_vol = trades_raw["qty"].sum() / 7`, hardcoding
7 days. On a week **missing days** (ETH/SOL week 3) this **understates daily_vol →
bucket_size too small → too many undersized buckets → VPIN distorted on exactly
the gap weeks**. Should divide by the actual number of present days. Also:
`bucket_size`/`n_buckets` are derived internally — should be parameters (enables
the sensitivity analysis VPIN needs).

---

## Next steps (revise before mock)

1. **Schema validation at the DataFrame boundary.** Enforce columns + dtypes
   after load (assert or `pandera` schema). This is the correct form of the
   "typing/dataclass" instinct.
2. **Storage-format defence — full tradeoff (CSV vs Parquet vs DuckDB/SQLite).**
   Guaranteed "why this format, what alternatives?" question.

   | Option | Verdict | Tradeoff |
   |---|---|---|
   | **CSV** | ✗ | Readable + universal, but **untyped** (timestamps reparse to strings), large, slow, no column pushdown |
   | **Parquet** | ✅ chosen | Columnar + compressed + **dtype-preserving** intermediate cache; reads only needed columns; splittable. Not human-readable, needs pyarrow, weak for tiny files |
   | **DuckDB / SQLite** | next step if scaled | Queryable, SQL over the data. A batch research pipeline doesn't need a DB — **but DuckDB-over-parquet is the natural evolution** (lazy, out-of-core, columnar SQL) |

   One-liner: *"Columnar, compressed, dtype-preserving cache so each stage
   re-runs independently and reads only the columns it needs; CSV loses types
   and is slow; a DB is overkill for a batch pipeline (DuckDB-over-parquet if it
   scaled)."*

### Code fixes applied during prep (working tree, not committed)
- `pipeline.py`: date parsing → regex (`\d{4}-\d{2}-\d{2}`) instead of `str.replace`.
- `loader.py`: `sort_values("timestamp")` enforces the chronological invariant.
- `tests/test_core.py`: `test_load_trades_timestamps_are_monotonic` guards it (suite 12 passed).
