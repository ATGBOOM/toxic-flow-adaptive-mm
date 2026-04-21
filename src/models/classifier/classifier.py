# src/models/classifier.py

import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    brier_score_loss, roc_auc_score, log_loss
)
import time
import warnings

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]
warnings.filterwarnings('ignore')

# Try importing catboost, fall back to xgboost
try:
    from catboost import CatBoostClassifier
    BOOST_LIB = 'catboost'
except ImportError:
    from xgboost import XGBClassifier
    BOOST_LIB = 'xgboost'
    

class VPINBaseline:
    """Threshold classifier using VPIN only."""
    
    def __init__(self, vpin_index):
        """
        Args:
            vpin_index: column index of VPIN in the feature matrix
        """
        self.vpin_index = vpin_index
    
    def fit(self, X, y):
        """No fitting needed — VPIN is precomputed."""
        return self
    
    def predict_proba(self, X):
        """
        Use VPIN as raw probability score.
        VPIN ranges ~0.13-0.19, so it's not a calibrated probability,
        but precision-recall curves only need ranking, not calibration.
        """
        vpin = X[:, self.vpin_index]
        # Return as 2-column array to match sklearn convention [P(not toxic), P(toxic)]
        return np.column_stack([1 - vpin, vpin])


class ToxicityClassifier:
    """Wraps all three models and handles training/evaluation."""
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.vpin_index = feature_names.index('vpin')
        self.scaler = StandardScaler()
        self.models = {}
        self.train_time = {}
    
    def fit(self, X_train, y_train):
        """Fit all three models."""
        
        # 1. VPIN baseline (no fitting)
        print("Fitting VPIN baseline...")
        vpin_model = VPINBaseline(self.vpin_index)
        vpin_model.fit(X_train, y_train)
        self.models['vpin'] = vpin_model
        self.train_time['vpin'] = 0
        print("  Done (no fitting needed)")
        
        # 2. Logistic regression (needs standardised features)
        print("Fitting logistic regression...")
        X_scaled = self.scaler.fit_transform(X_train)
        
        t0 = time.time()
        lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        lr.fit(X_scaled, y_train)
        self.train_time['logreg'] = time.time() - t0
        self.models['logreg'] = lr
        print(f"  Done in {self.train_time['logreg']:.1f}s")
        
        # 3. Gradient boosted trees
        print(f"Fitting gradient boosted trees ({BOOST_LIB})...")
        t0 = time.time()
        
        if BOOST_LIB == 'catboost':
            gbt = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                verbose=100,
                random_seed=42,
            )
            gbt.fit(X_train, y_train)
        else:
            gbt = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                reg_lambda=3,
                verbosity=1,
                random_state=42,
                scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
            )
            gbt.fit(X_train, y_train)
        
        self.train_time['gbt'] = time.time() - t0
        self.models['gbt'] = gbt
        print(f"  Done in {self.train_time['gbt']:.1f}s")
    
    def predict_proba(self, X, model_name):
        """Get probability predictions from a specific model."""
        if model_name == 'logreg':
            X_scaled = self.scaler.transform(X)
            return self.models[model_name].predict_proba(X_scaled)[:, 1]
        elif model_name == 'vpin':
            return self.models[model_name].predict_proba(X)[:, 1]
        else:
            return self.models[model_name].predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test, dataset_name="test"):
        """Evaluate all models on a test set."""
        print(f"\n{'='*60}")
        print(f"Evaluation: {dataset_name}")
        print(f"  Rows: {len(y_test):,}  |  Toxic: {y_test.sum():,} ({y_test.mean():.3f})")
        print(f"{'='*60}")
        
        results = {}
        
        for name in ['vpin', 'logreg', 'gbt']:
            probs = self.predict_proba(X_test, name)
            
            # Core metrics
            ap = average_precision_score(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            brier = brier_score_loss(y_test, probs)
            
            # Precision-recall at specific recall targets
            precision, recall, thresholds = precision_recall_curve(y_test, probs)
            
            # Find precision at 50% and 80% recall
            idx_50 = np.where(recall >= 0.50)[0]
            p_at_50 = precision[idx_50[-1]] if len(idx_50) > 0 else 0.0
            idx_80 = np.where(recall >= 0.80)[0]
            p_at_80 = precision[idx_80[-1]] if len(idx_80) > 0 else 0.0
            
            results[name] = {
                'avg_precision': ap,
                'auc_roc': auc,
                'brier_score': brier,
                'precision_at_50_recall': p_at_50,
                'precision_at_80_recall': p_at_80,
                'precision_curve': precision,
                'recall_curve': recall,
                'thresholds': thresholds,
                'probs': probs,
            }
            
            print(f"\n  {name:>8s}:  AP={ap:.4f}  AUC={auc:.4f}  Brier={brier:.4f}")
            print(f"           Precision@50%recall={p_at_50:.3f}  Precision@80%recall={p_at_80:.3f}")
            for threshold in THRESHOLDS:
                predicted_toxic = (probs >= threshold)  # boolean array, True where model is "confident enough"
                tp = (predicted_toxic & (y_test == 1)).sum()
                fp = (predicted_toxic & (y_test == 0)).sum()
                fn = (~predicted_toxic & (y_test == 1)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn)
                flag_rate = predicted_toxic.sum() / len(y_test)
                print (f"Threshold: {threshold}  Precision : {precision}  Recall : {recall}  Flag Rate: {flag_rate}")
        
        return results
    
    def get_logreg_coefficients(self):
        """Return logistic regression coefficients for interpretation."""
        lr = self.models['logreg']
        coefs = pd.Series(lr.coef_[0], index=self.feature_names)
        return coefs.sort_values(key=abs, ascending=False)


if __name__ == "__main__":
    from data_loader import prepare_split, get_feature_columns, load_weeks
    
    data_dir = "data/processed/features"
    
    # === Split 1: Train week 1, test week 2 ===
    print("\n" + "="*60)
    print("SPLIT 1a: Train week1 -> Test week2")
    print("="*60)
    
    split1a = prepare_split(
        data_dir,
        train_weeks=['week1'],
        test_weeks_dict={'week2': ['week2']},
        n_train=500_000,
    )
    
    features = split1a['features']
    clf1 = ToxicityClassifier(features)
    clf1.fit(split1a['X_train'], split1a['y_train'])
    
    results1_w2 = clf1.evaluate(split1a['X_test_week2'], split1a['y_test_week2'], "Split1 → Week2")
    
    # Free test data, keep model
    del split1a
    gc.collect()
    
    # Now test on week 3 with the same model
    print("\n" + "="*60)
    print("SPLIT 1b: Train week1 -> Test week3")
    print("="*60)
    
    print("Loading test data (week3)...")
    test_w3 = load_weeks(data_dir, ['week3'])
    X_test_w3 = test_w3[features].values
    y_test_w3 = test_w3['toxic'].values
    del test_w3
    gc.collect()
    
    results1_w3 = clf1.evaluate(X_test_w3, y_test_w3, "Split1 → Week3")
    
    print("\n\nLogistic regression coefficients:")
    print(clf1.get_logreg_coefficients().to_string())
    
    del clf1, X_test_w3, y_test_w3, results1_w2, results1_w3
    gc.collect()
    
    
    # === Split 2: Train weeks 1+2, test week 3 ===
    print("\n\n" + "="*60)
    print("SPLIT 2: Train week1+week2 -> Test week3")
    print("="*60)
    
    split2 = prepare_split(
        data_dir,
        train_weeks=['week1', 'week2'],
        test_weeks_dict={'week3': ['week3']},
        n_train=500_000,
    )
    
    clf2 = ToxicityClassifier(features)
    clf2.fit(split2['X_train'], split2['y_train'])
    
    results2_w3 = clf2.evaluate(split2['X_test_week3'], split2['y_test_week3'], "Split2 → Week3")
    
    print("\n\nLogistic regression coefficients (Split 2):")
    print(clf2.get_logreg_coefficients().to_string())
    
    # === Summary comparison ===
    print("\n\n" + "="*60)
    print("SUMMARY: Week 3 performance comparison")
    print("="*60)
    print(f"{'Model':<10} {'Split1 AP':>12} {'Split2 AP':>12} {'Improvement':>12}")
    print("-"*48)
    for name in ['vpin', 'logreg', 'gbt']:
        ap1 = results1_w3[name]['avg_precision']
        ap2 = results2_w3[name]['avg_precision']
        diff = (ap2 - ap1) / ap1 * 100
        print(f"{name:<10} {ap1:>12.4f} {ap2:>12.4f} {diff:>+11.1f}%")