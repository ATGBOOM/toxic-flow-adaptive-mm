# src/models/data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path


# Features we pull directly from parquet
RAW_FEATURES = [
    'depth_imbalance_1', 'depth_imbalance_5', 'depth_imbalance_10', 'depth_imbalance_25',
    'bid_pressure', 'ask_pressure', 'pressure_imbalance',
    'trade_intensity_1s', 'trade_intensity_5s', 'trade_intensity_10s',
    'volume_acceleration', 'signed_vol_imbalance_10s', 'vpin',
]

# Columns we need from parquet to engineer new features
LOAD_COLS = RAW_FEATURES + ['spread', 'microprice', 'midprice', 'qty', 'toxic', 'timestamp']

ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
WEEKS = ['week1', 'week2', 'week3']


def load_asset_week(data_dir, asset, week):
    """Load a single asset-week parquet and engineer features."""
    path = Path(data_dir) / f"{asset}_{week}_full_features.parquet"
    df = pd.read_parquet(path, columns=LOAD_COLS)

    # --- Engineer new features ---

    # Spread in basis points (cross-asset comparable)
    df['spread_bps'] = df['spread'] / df['midprice'] * 10_000

    # Book asymmetry without raw price level
    df['microprice_minus_mid'] = df['microprice'] - df['midprice']

    # Normalised trade size (rolling 1000-trade mean)
    df['qty_normalised'] = df['qty'] / df['qty'].rolling(1000, min_periods=1).mean()

    # Asset identifier (for the "with asset indicator" experiment)
    df['asset_id'] = ASSETS.index(asset)

    # Drop rows with NaN VPIN (warmup period)
    df = df.dropna(subset=['vpin'])

    # Drop raw columns we no longer need
    df = df.drop(columns=['spread', 'microprice', 'midprice', 'qty', 'timestamp'])

    # Drop any remaining NaNs in features
    df = df.dropna()

    return df


def load_weeks(data_dir, weeks, assets=ASSETS):
    """Load and pool multiple asset-weeks."""
    dfs = []
    for asset in assets:
        for week in weeks:
            df = load_asset_week(data_dir, asset, week)
            print(f"  {asset} {week}: {len(df):,} rows, toxic={df['toxic'].mean():.3f}")
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  TOTAL: {len(combined):,} rows, toxic={combined['toxic'].mean():.3f}")
    return combined


def get_feature_columns(include_asset=False):
    """Return the list of feature column names."""
    features = RAW_FEATURES + ['spread_bps', 'microprice_minus_mid', 'qty_normalised']
    if include_asset:
        features.append('asset_id')
    return features


def subsample_stratified(df, n=500_000, seed=42):
    """Stratified subsample preserving toxic/non-toxic ratio."""
    toxic = df[df['toxic']]
    non_toxic = df[~df['toxic']]

    toxic_ratio = len(toxic) / len(df)
    n_toxic = int(n * toxic_ratio)
    n_non_toxic = n - n_toxic

    toxic_sample = toxic.sample(n=min(n_toxic, len(toxic)), random_state=seed)
    non_toxic_sample = non_toxic.sample(n=min(n_non_toxic, len(non_toxic)), random_state=seed)

    result = pd.concat([toxic_sample, non_toxic_sample]).sample(frac=1, random_state=seed)
    print(f"  Subsampled: {len(result):,} rows, toxic={result['toxic'].mean():.3f}")
    return result


def prepare_split(data_dir, train_weeks, test_weeks_dict, n_train=500_000, seed=42):
    """
    Prepare a full train/test split.

    Args:
        data_dir: path to feature parquets
        train_weeks: list of weeks for training, e.g. ['week1', 'week2']
        test_weeks_dict: dict of {name: [weeks]}, e.g. {'week2': ['week2'], 'week3': ['week3']}
        n_train: subsample size for training
        seed: random seed

    Returns:
        dict with 'train' and test set DataFrames
    """
    features = get_feature_columns(include_asset=False)

    print("Loading training data...")
    train_full = load_weeks(data_dir, train_weeks)
    train = subsample_stratified(train_full, n=n_train, seed=seed)
    del train_full  # free memory

    X_train = train[features].values
    y_train = train['toxic'].values

    result = {
        'X_train': X_train,
        'y_train': y_train,
        'features': features,
    }

    for name, weeks in test_weeks_dict.items():
        print(f"\nLoading test data ({name})...")
        test = load_weeks(data_dir, weeks)
        result[f'X_test_{name}'] = test[features].values
        result[f'y_test_{name}'] = test['toxic'].values

    return result


if __name__ == "__main__":
    # Quick test: load week 1, check shapes and feature stats
    data_dir = "data/processed/features"

    print("=== Quick data check ===\n")
    df = load_asset_week(data_dir, 'BTCUSDT', 'week1')
    features = get_feature_columns()

    print(f"\nShape: {df.shape}")
    print(f"Features: {features}")
    print(f"Toxic rate: {df['toxic'].mean():.3f}")
    print(f"\nFeature stats:")
    print(df[features].describe().round(4).to_string())