"""
benchmark_book_structure.py — measures the actual dict-vs-SortedDict
tradeoff at realistic order-book depth, instead of asserting it.

Not part of the pipeline. Compares:
  - reads:  compute_book_features(), old dict+sort() vs. new SortedDict
  - writes: apply_update()-style single-level upsert, dict O(1) vs.
            SortedDict O(log L)

at L=500 levels per side (Bybit ob500 depth), which is the scale the
"dominant cost" claim is actually about — the real mock fixture (30 levels)
is too shallow to show the effect honestly.
"""

import random
import timeit
from sortedcontainers import SortedDict

L = 500          # levels per side, matches Bybit ob500
N_READS = 2000    # simulated feature-read calls (one per sampled trade)
N_WRITES = 20000  # simulated single-level upserts (one per delta message)

random.seed(42)


def build_plain_dict(n: int, base: float) -> dict:
    return {round(base - i * 0.5, 2): random.random() * 3 for i in range(n)}


def build_sorted_dict(n: int, base: float) -> SortedDict:
    return SortedDict(build_plain_dict(n, base))


def read_old(bids: dict, asks: dict) -> None:
    sorted_bids = sorted(bids.keys(), reverse=True)
    sorted_asks = sorted(asks.keys())
    best_bid, best_ask = sorted_bids[0], sorted_asks[0]
    for n in (1, 5, 10, 25):
        sum(bids[p] for p in sorted_bids[:n])
        sum(asks[p] for p in sorted_asks[:n])


def read_new(bids: SortedDict, asks: SortedDict) -> None:
    best_bid, _ = bids.peekitem(-1)
    best_ask, _ = asks.peekitem(0)
    bid_vals, ask_vals = bids.values(), asks.values()
    for n in (1, 5, 10, 25):
        sum(bid_vals[-n:])
        sum(ask_vals[:n])


def write_old(book: dict, price: float, size: float) -> None:
    book[price] = size


def write_new(book: SortedDict, price: float, size: float) -> None:
    book[price] = size


def main() -> None:
    print(f"Book depth: {L} levels/side | reads: {N_READS} | writes: {N_WRITES}\n")

    # ── reads (the "dominant cost" claim) ──────────────────────────────
    bids_d = build_plain_dict(L, 60_000)
    asks_d = build_plain_dict(L, 60_050)
    bids_s = build_sorted_dict(L, 60_000)
    asks_s = build_sorted_dict(L, 60_050)

    t_old = timeit.timeit(lambda: read_old(bids_d, asks_d), number=N_READS)
    t_new = timeit.timeit(lambda: read_new(bids_s, asks_s), number=N_READS)
    print(f"READ  (best bid/ask + depth-1/5/10/25), {N_READS} calls:")
    print(f"  dict + sort() every call : {t_old:.4f}s  ({t_old/N_READS*1e6:.1f} us/call)")
    print(f"  SortedDict                : {t_new:.4f}s  ({t_new/N_READS*1e6:.1f} us/call)")
    print(f"  speedup: {t_old/t_new:.1f}x\n")

    # ── writes (the flip side — SortedDict is NOT free) ────────────────
    book_d = build_plain_dict(L, 60_000)
    book_s = build_sorted_dict(L, 60_000)
    prices = [round(60_000 - random.random() * 250, 2) for _ in range(N_WRITES)]
    sizes = [random.random() * 3 for _ in range(N_WRITES)]

    t_old_w = timeit.timeit(
        lambda: [write_old(book_d, p, s) for p, s in zip(prices, sizes)], number=1
    )
    t_new_w = timeit.timeit(
        lambda: [write_new(book_s, p, s) for p, s in zip(prices, sizes)], number=1
    )
    print(f"WRITE (single-level upsert), {N_WRITES} calls:")
    print(f"  dict       : {t_old_w:.4f}s  ({t_old_w/N_WRITES*1e6:.2f} us/call)")
    print(f"  SortedDict : {t_new_w:.4f}s  ({t_new_w/N_WRITES*1e6:.2f} us/call)")
    print(f"  SortedDict is {t_new_w/t_old_w:.1f}x slower per write — the honest cost "
          f"of the read-side win")


if __name__ == "__main__":
    main()
