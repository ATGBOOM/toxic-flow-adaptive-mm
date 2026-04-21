# ## Classifier Investigation Results — Session 8

# ### Key Numbers

# | Investigation | Model | Split | AP | Best Precision | Threshold |
# |---|---|---|---|---|---|
# | Inv1 - Balanced | CatBoost balanced | week1→week2 | 0.0949 | 0.083 | t=0.30 |
# | Inv1 - Unbalanced | CatBoost unbalanced | week1→week2 | 0.0925 | 0.107 | t=0.10 |
# | Inv1 - Balanced | CatBoost balanced | week1+2→week3 | 0.2436 | 0.188 | t=0.30 |
# | Inv1 - Unbalanced | CatBoost unbalanced | week1+2→week3 | 0.2476 | 0.300 | t=0.20 |
# | Inv2 - 2M rows | CatBoost unbalanced | week1→week2 | 0.0891 | ~0.102 | t=0.10 |
# | Inv3 - asset_id | CatBoost unbalanced | week1+2→week3 | 0.2495 | 0.310 | t=0.20 |

# ### Conclusions
# 1. Use unbalanced CatBoost — better calibrated, AP equivalent, meaningful thresholds
# 2. Training size does not help — feature-limited not data-limited
# 3. Asset identity is redundant — microstructure features already capture per-asset differences
# 4. Breakeven toxicity probability for market maker = S/(S+L) = 5/(5+8) = 0.38
# 5. Best classifier precision = 0.30 — does not clear breakeven for binary pull-quotes decision
# 6. Recommended use: continuous spread widening proportional to p_toxic, not binary on/off


from catboost import CatBoostClassifier
from joblib import register_compressor
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    brier_score_loss, roc_auc_score, log_loss
)
from data_loader import prepare_split, get_feature_columns, load_weeks
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]
class Investigate:

  def __init__(self):
    self.models = {}
    self.data_dir = "data/processed/features"


  def balvsunbal(self):
   
    # split1a = prepare_split(
    #     self.data_dir,
    #     train_weeks=['week1'],
    #     test_weeks_dict={'week2': ['week2']},
    #     n_train=500_000,
    # )
    # x_train = split1a['X_train']
    # y_train = split1a['y_train']
    # x_test, y_test = split1a['X_test_week2'], split1a['y_test_week2']
    split1a = prepare_split(
        self.data_dir,
        train_weeks=['week1', 'week2'],
        test_weeks_dict={'week3': ['week3']},
        n_train=500_000,
    )
    x_train = split1a['X_train']
    y_train = split1a['y_train']
    x_test, y_test = split1a['X_test_week3'], split1a['y_test_week3']
    bal_gbt = CatBoostClassifier(
          iterations=500,
          depth=6,
          learning_rate=0.05,
          l2_leaf_reg=3,
          verbose=0,
          random_seed=42,
          auto_class_weights='Balanced',  # handle class imbalance
      )

    self.models['balanced_gbt'] = bal_gbt

    unbal_gbt = CatBoostClassifier(
          iterations=500,
          depth=6,
          learning_rate=0.05,
          l2_leaf_reg=3,
          verbose=0,
          random_seed=42,
      )

    self.models['unbalanced_gbt'] = unbal_gbt
 
    
    for name in ['balanced_gbt', 'unbalanced_gbt']:
      self.models[name].fit(x_train, y_train)
      probs = self.models[name].predict_proba(x_test)[:, 1]
  
      ap = average_precision_score(y_test, probs)
      brier = brier_score_loss(y_test, probs)

      #lower brier is better, higher ap is better
      print(f"\n  {name:>8s}:  AP={ap:.4f}  Brier={brier:.4f}")
      
      for threshold in THRESHOLDS:
        predicted_toxic = (probs >= threshold)  # boolean array, True where model is "confident enough"
        tp = (predicted_toxic & (y_test == 1)).sum()
        fp = (predicted_toxic & (y_test == 0)).sum()
        fn = (~predicted_toxic & (y_test == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn)
        flag_rate = predicted_toxic.sum() / len(y_test)
        print (f"Threshold: {threshold}  Precision : {precision}  Recall : {recall}  Flag Rate: {flag_rate}")
  
  def increase_training_data(self):
    for n in [500_000, 2_000_000]:
      for split in [(['week1'], ['week2']), (['week1', 'week2'], ['week3'])]:
        split1a = prepare_split(
            self.data_dir,
            train_weeks=split[0],
            test_weeks_dict={'test': split[1]},
            n_train=n,
        )
        x_train = split1a['X_train']
        y_train = split1a['y_train']
        x_test, y_test = split1a['X_test_test'], split1a['y_test_test']

        unbal_gbt = CatBoostClassifier(
              iterations=500,
              depth=6,
              learning_rate=0.05,
              l2_leaf_reg=3,
              verbose=0,
              random_seed=42,
              # auto_class_weights='Balanced',  # handle class imbalance
          )

        self.models['gbt'] = unbal_gbt

    
        
        for name in ['gbt']:
          print(f"\n=== n_train={n:,}  train={split[0]}  test={split[1]} ===")
          self.models[name].fit(x_train, y_train)
          probs = self.models[name].predict_proba(x_test)[:, 1]
      
          ap = average_precision_score(y_test, probs)
          brier = brier_score_loss(y_test, probs)

          #lower brier is better, higher ap is better
          print(f"\n  {name:>8s}:  AP={ap:.4f}  Brier={brier:.4f}")
          
          for threshold in THRESHOLDS:
            predicted_toxic = (probs >= threshold)  # boolean array, True where model is "confident enough"
            tp = (predicted_toxic & (y_test == 1)).sum()
            fp = (predicted_toxic & (y_test == 0)).sum()
            fn = (~predicted_toxic & (y_test == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn)
            flag_rate = predicted_toxic.sum() / len(y_test)
            print (f"Threshold: {threshold}  Precision : {precision}  Recall : {recall}  Flag Rate: {flag_rate}")

  def asset_agnostic(self):
    for include in [True, False]:
      for split in [(['week1'], ['week2']), (['week1', 'week2'], ['week3'])]:
        split1a = prepare_split(
            self.data_dir,
            train_weeks=split[0],
            test_weeks_dict={'test': split[1]},
            n_train=500_000,
            include_asset=include
        )
        x_train = split1a['X_train']
        y_train = split1a['y_train']
        x_test, y_test = split1a['X_test_test'], split1a['y_test_test']

        unbal_gbt = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            verbose=0,
            random_seed=42,
  
        )
        self.models['gbt'] = unbal_gbt
        
        for name in ['gbt']:
          print(f"\n=== Asset Included={include}  train={split[0]}  test={split[1]} ===")
          print(f"Features: {x_train.shape[1]}")
          self.models[name].fit(x_train, y_train)
          probs = self.models[name].predict_proba(x_test)[:, 1]
      
          ap = average_precision_score(y_test, probs)
          brier = brier_score_loss(y_test, probs)

          #lower brier is better, higher ap is better
          print(f"\n  {name:>8s}:  AP={ap:.4f}  Brier={brier:.4f}")
          
          for threshold in THRESHOLDS:
            predicted_toxic = (probs >= threshold)  # boolean array, True where model is "confident enough"
            tp = (predicted_toxic & (y_test == 1)).sum()
            fp = (predicted_toxic & (y_test == 0)).sum()
            fn = (~predicted_toxic & (y_test == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn)
            flag_rate = predicted_toxic.sum() / len(y_test)
            print (f"Threshold: {threshold}  Precision : {precision}  Recall : {recall}  Flag Rate: {flag_rate}")

if __name__ == "__main__":
  investigate = Investigate()
  # investigate.balvsunbal()
  # investigate.increase_training_data()
  investigate.asset_agnostic()




# Summary

# Investigation 1 -> An unabalanced catboost is more predictive than balanced one as it overestimates the toxicity filter reducing the value over regimes
# Investigation 2 -> Training data size did not affect results as training set was not data limited but feature limited, the original vpin had many more features including users data which is helpful as informed traders are likely to repeat. We have mostly used pure trade and orderbook data in this. 
# Investigation 3 -> Asset Agnostic does not affect as our other features already display the changes in that come with different assets like trade volatility, imbalance etc. 

# For adaptive market we want the highest pnl. E[PnL] = (1 - p_toxic) × S - p_toxic × L. We calculate our spread profit and loss and calculate the probability to break even. We can then choose the precisions above that as our indicators.