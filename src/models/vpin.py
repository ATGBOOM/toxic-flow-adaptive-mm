from pathlib import Path
from numpy import roll
import pandas as pd


ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
WEEKS = ['WEEK1', 'WEEK2', 'WEEK3']
DATA_DIR = Path("data/processed/BTCUSDT/week1.parquet")


def build_volume_bucket(df, bucket_size):
    buckets = []
    vol = {"v_buy": 0.0, "v_sell": 0.0, "timestamp": None}
    
    # extract to numpy arrays — much faster than iterrows
    qtys = df['qty'].to_numpy()
    signs = df['sign'].to_numpy()
    timestamps = df['timestamp'].to_numpy()
    
    for i in range(len(qtys)):
        remaining_qty = qtys[i]
        ts = timestamps[i]
        sign = signs[i]
        
        while remaining_qty > 0:
            capacity = bucket_size - (vol['v_buy'] + vol['v_sell'])
            
            if remaining_qty <= capacity:
                if sign == 1:
                    vol['v_buy'] += remaining_qty
                else:
                    vol['v_sell'] += remaining_qty
                vol['timestamp'] = ts
                remaining_qty = 0
            else:
                if sign == 1:
                    vol['v_buy'] += capacity
                else:
                    vol['v_sell'] += capacity
                vol['timestamp'] = ts
                buckets.append(vol.copy())
                vol = {'v_buy': 0.0, 'v_sell': 0.0, 'timestamp': None}
                remaining_qty -= capacity
    
    return buckets


def compute_rolling_vpins(volume_buckets, n_buckets, bucket_size):
    vpins = []
    for i in range(n_buckets, len(volume_buckets) + 1):
        window = volume_buckets[i - n_buckets:i]
        vpin = sum(abs(b['v_buy'] - b['v_sell']) for b in window) / (n_buckets * bucket_size)
        vpins.append({
            'timestamp': volume_buckets[i - 1]['timestamp'],
            'vpin': vpin
        })
    return pd.DataFrame(vpins)



def compute_vpin(df, bucket_size, n_buckets=50):
    """
    df: DataFrame with columns [timestamp, price, qty, sign]
    bucket_size: total volume per bucket (in BTC)
    n_buckets: rolling window length for VPIN average
    
    returns: DataFrame with columns [timestamp, vpin]
    """
    
    # initialise a an array and put in subset of the trades with volumes adding up to bucket_size - store the v_b, v_s, timestamp
    # if a single trade goes over 2 buckets then split it
    volume_buckets = build_volume_bucket(df, bucket_size)

    # iterate through each bucket and compute the rolling VPIN over the n_buckets, using sliding window technnique
     # formula is 1/n * sum of (V_B - V_S)/V
    rolling_vpins = compute_rolling_vpins(volume_buckets, n_buckets, bucket_size)

    return rolling_vpins
  


# total volume over the week
for asset in ASSETS:
  df1 = pd.read_parquet(f"data/processed/{asset}/week1.parquet")
  fixed_bucket_size = df1['qty'].sum() / 7 / 50
  print("Asset:", asset)
  for week in ['week1', 'week2', 'week3']:
      df = pd.read_parquet(f"data/processed/{asset}/{week}.parquet")
      result = compute_vpin(df, fixed_bucket_size, n_buckets=50)
      print(f"{week}: mean VPIN = {result['vpin'].mean():.4f}, "
            f"max VPIN = {result['vpin'].max():.4f}, "
            f"buckets = {len(result)}")



# Session 6 Summary — VPIN Implementation

# Implementation:
# - Built VPIN from scratch using numpy array extraction (iterrows was 109s, numpy loop 2s)
# - Bucket filling with trade splitting via while loop — handles trades spanning multiple buckets
# - Bucket size set to 1/50 of daily volume following Easley et al. (2012)
# - BTC week 1 daily volume: ~118,825 BTC → bucket size: 2376 BTC → ~300 buckets per week
# - qty (units of BTC) used as volume, NOT price × qty — each unit of BTC traded is one
#   information event regardless of dollar value

# Key methodological decision — fixed vs per-week bucket size:
# - Per-week calibration (Easley method): each bucket = 1/50 daily volume, VPIN values
#   comparable within asset but not across weeks with different volume
# - Fixed bucket size (week 1 as reference): comparable bucket counts across weeks but
#   week 3 buckets represent smaller fraction of daily activity (produces 400-969 buckets
#   vs intended 300-350)
# - For classifier (Phase 3): use per-week calibration — each VPIN value should be
#   meaningful relative to that week's activity
# - For cross-regime comparison: use fixed bucket size

# Results (fixed bucket size = 2376 BTC):
# BTCUSDT:
#   week1 (consolidation): mean VPIN = 0.1619, max = 0.1920, buckets = 300
#   week2 (breakout):      mean VPIN = 0.1707, max = 0.2113, buckets = 350
#   week3 (correction):    mean VPIN = 0.1487, max = 0.1933, buckets = 400

# ETHUSDT:
#   week1: mean VPIN = 0.1753, max = 0.2075, buckets = 301
#   week2: mean VPIN = 0.1936, max = 0.2649, buckets = 407
#   week3: mean VPIN = 0.1859, max = 0.2544, buckets = 969

# SOLUSDT:
#   week1: mean VPIN = 0.1644, max = 0.1905, buckets = 300
#   week2: mean VPIN = 0.1659, max = 0.1970, buckets = 329
#   week3: mean VPIN = 0.1546, max = 0.2184, buckets = 746

# Key finding — stress regime has lowest VPIN, not highest:
# Week 3 (correction) has the lowest mean VPIN for BTC and SOL despite being the most
# volatile regime. This is not a bug. In a correction, aggressive sellers AND opportunistic
# buyers are simultaneously active — high volume on both sides means low net imbalance.
# VPIN detects directional informed trading well (week 2 breakout).
# It struggles with volatility regimes where both sides react simultaneously (week 3).
# This is a real documented limitation worth including in the limitations section.

# ETH anomaly — week 2 has higher max VPIN (0.265) relative to mean (0.194):
# ETH has higher beta to BTC moves. In the breakout, informed traders seeking leveraged
# crypto exposure may concentrate in ETH perps, creating sharper imbalance spikes.
# Hypothesis only — not tested rigorously.

# Validation:
# Correlation between VPIN and absolute 30-minute forward return in BTC week 2: 0.145
# VPIN predicts magnitude of price moves, not direction (hence absolute return).
# 0.145 is modest but consistent with Easley et al. equity market findings.
# This is the baseline benchmark — the classifier in Phase 3 must beat this.

# VPIN interpretation guide:
# 0.15 - 0.25 → calm market, uninformed flow dominating
# 0.25 - 0.45 → elevated, some directional pressure
# 0.45 - 0.65 → high toxicity, market makers historically widen spreads
# 0.65+       → extreme, rare, major news events or dislocations

# Limitations:
# 1. Bucket size sensitivity — smaller buckets amplify noise, larger buckets mute signal.
#    Must justify choice and report sensitivity.
# 2. Cross-asset comparison unreliable — different assets have different volume scales and
#    baseline imbalance levels. BTC week 3 bucket (2376 BTC) represents much smaller
#    fraction of daily activity than intended.
# 3. VPIN detects directional informed trading but fails in volatility regimes where both
#    sides are simultaneously active — imbalance collapses even as volatility spikes.
# 4. VPIN is a lagging signal by construction — it summarises the last n_buckets, not
#    current conditions.