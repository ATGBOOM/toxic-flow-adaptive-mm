# src/models/data_loader.py

import gc
from pathlib import Path

import numpy as np
import pandas as pd


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


def load_asset_week(data_dir: str | Path, asset: str, week: str) -> pd.DataFrame:
    """Load a single asset-week parquet and engineer derived features.

    Args:
        data_dir: Path to the directory containing full-feature parquets.
        asset: Asset symbol, e.g. 'BTCUSDT'.
        week: Week label, e.g. 'week1'.

    Returns:
        DataFrame with RAW_FEATURES plus spread_bps, microprice_minus_mid,
        qty_normalised, asset_id, and toxic columns. Rows with NaN VPIN or
        any remaining NaN features are dropped.
    """
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


def load_weeks(
    data_dir: str | Path,
    weeks: list[str],
    assets: list[str] = ASSETS,
) -> pd.DataFrame:
    """Load and pool multiple asset-weeks into a single DataFrame.

    Args:
        data_dir: Path to the directory containing full-feature parquets.
        weeks: List of week labels to load, e.g. ['week1', 'week2'].
        assets: List of asset symbols to include. Defaults to all three assets.

    Returns:
        Concatenated DataFrame of all requested asset-weeks with an index
        reset to 0..N-1.
    """
    dfs = []
    for asset in assets:
        for week in weeks:
            df = load_asset_week(data_dir, asset, week)
            print(f"  {asset} {week}: {len(df):,} rows, toxic={df['toxic'].mean():.3f}")
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  TOTAL: {len(combined):,} rows, toxic={combined['toxic'].mean():.3f}")
    return combined


def get_feature_columns(include_asset: bool = False) -> list[str]:
    """Return the ordered list of model input feature names.

    Args:
        include_asset: If True, append 'asset_id' as an additional feature.
            Defaults to False.

    Returns:
        List of feature column name strings.
    """
    features = RAW_FEATURES + ['spread_bps', 'microprice_minus_mid', 'qty_normalised']
    if include_asset:
        features.append('asset_id')
    return features


def subsample_stratified(
    df: pd.DataFrame, n: int = 500_000, seed: int = 42
) -> pd.DataFrame:
    """Stratified subsample that preserves the toxic/non-toxic class ratio.

    Args:
        df: DataFrame with a boolean 'toxic' column.
        n: Target number of rows in the sample. If the class has fewer rows
            than its proportional allocation, all rows of that class are used.
        seed: Random seed for reproducibility.

    Returns:
        Shuffled DataFrame with at most n rows, preserving the toxic rate.
    """
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


def prepare_split(
    data_dir: str | Path,
    train_weeks: list[str],
    test_weeks_dict: dict[str, list[str]],
    n_train: int = 500_000,
    seed: int = 42,
    include_asset: bool = False,
) -> dict:
    """Prepare a full train/test split as numpy arrays.

    Args:
        data_dir: Path to the directory containing full-feature parquets.
        train_weeks: List of week labels used for training, e.g. ['week1', 'week2'].
        test_weeks_dict: Mapping of split name to week list, e.g.
            {'week2': ['week2'], 'week3': ['week3']}. Each entry produces an
            X_test_<name> and y_test_<name> key in the result.
        n_train: Number of training rows after stratified subsampling.
        seed: Random seed for subsampling and shuffling.
        include_asset: If True, include asset_id as a feature.

    Returns:
        Dict with keys 'X_train', 'y_train', 'features', and one pair of
        'X_test_<name>' / 'y_test_<name>' for each entry in test_weeks_dict.
    """
    features = get_feature_columns(include_asset)

    print("Loading training data...")
    train_full = load_weeks(data_dir, train_weeks)
    train = subsample_stratified(train_full, n=n_train, seed=seed)
    del train_full  # free memory

    X_train = train[features].values
    y_train = train['toxic'].values

    result: dict = {
        'X_train': X_train,
        'y_train': y_train,
        'features': features,
    }

    for name, weeks in test_weeks_dict.items():
        print(f"\nLoading test data ({name})...")
        test = load_weeks(data_dir, weeks)
        result[f'X_test_{name}'] = test[features].values
        result[f'y_test_{name}'] = test['toxic'].values
        del test
        gc.collect()

    return result


if __name__ == "__main__":
    # Quick test: load week 1, check shapes and feature stats
    _data_dir = str(
        Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'features'
    )

    print("=== Quick data check ===\n")
    df = load_asset_week(_data_dir, 'BTCUSDT', 'week1')
    features = get_feature_columns()

    print(f"\nShape: {df.shape}")
    print(f"Features: {features}")
    print(f"Toxic rate: {df['toxic'].mean():.3f}")
    print(f"\nFeature stats:")
    print(df[features].describe().round(4).to_string())
