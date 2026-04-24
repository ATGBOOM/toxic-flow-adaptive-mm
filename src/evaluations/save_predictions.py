# src/evaluation/save_predictions.py

"""
Run once to train models on Split 2 and save predictions + models.
Output:
  results/models/logreg_split2.joblib
  results/models/gbt_split2.joblib
  results/predictions/split2_week3_predictions.parquet  (per-asset)
"""

import gc
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.classifier.data_loader import prepare_split, get_feature_columns, load_asset_week, ASSETS
from src.models.classifier.classifier import ToxicityClassifier

DATA_DIR   = "data/processed/features"
MODELS_DIR = Path("results/models")
PREDS_DIR  = Path("results/predictions")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR.mkdir(parents=True, exist_ok=True)


def run():
    features = get_feature_columns(include_asset=False)

    # ── Train on weeks 1+2 ──────────────────────────────────────────
    print("=" * 60)
    print("Split 2: Train weeks 1+2")
    print("=" * 60)

    split2 = prepare_split(
        DATA_DIR,
        train_weeks=['week1', 'week2'],
        test_weeks_dict={},      # no pooled test — we do per-asset below
        n_train=500_000,
    )

    clf = ToxicityClassifier(features)
    clf.fit(split2['X_train'], split2['y_train'])

    del split2
    gc.collect()

    # ── Save models ─────────────────────────────────────────────────
    joblib.dump(clf.models['logreg'], MODELS_DIR / 'logreg_split2.joblib')
    joblib.dump(clf.scaler,           MODELS_DIR / 'scaler_split2.joblib')
    joblib.dump(clf.models['gbt'],    MODELS_DIR / 'gbt_split2.joblib')
    print("\nModels saved.")

    # ── Save per-asset predictions on week 3 ────────────────────────
    # We save per-asset so the paired Wilcoxon test has three observations
    print("\nGenerating per-asset week 3 predictions...")

    all_preds = []

    for asset in ASSETS:
        print(f"  {asset}...")
        df = load_asset_week(DATA_DIR, asset, 'week3')

        X = df[features].values
        y = df['toxic'].values

        p_logreg = clf.predict_proba(X, 'logreg')
        p_gbt    = clf.predict_proba(X, 'gbt')
        p_vpin   = clf.predict_proba(X, 'vpin')

        preds = pd.DataFrame({
            'asset':   asset,
            'y_true':  y,
            'p_logreg': p_logreg,
            'p_gbt':    p_gbt,
            'p_vpin':   p_vpin,
        })

        all_preds.append(preds)

        del df, X, y, p_logreg, p_gbt, p_vpin
        gc.collect()

    predictions = pd.concat(all_preds, ignore_index=True)
    out_path = PREDS_DIR / 'split2_week3_predictions.parquet'
    predictions.to_parquet(out_path, index=False)

    print(f"\nPredictions saved to {out_path}")
    print(f"Shape: {predictions.shape}")
    print(f"\nPer-asset toxic rates:")
    print(predictions.groupby('asset')['y_true'].mean().round(3).to_string())



if __name__ == "__main__":
    run()