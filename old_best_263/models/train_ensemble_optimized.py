"""
Train Optimized Ensemble (Power of 20).
Implements a Bagged Ensemble of HistGradientBoostingRegressors.
- 10 Folds x 2 Random Seeds = 20 Models.
- Aggressive Parameters (Depth 16, Low L2).
- Out-of-Fold (OOF) Validation for honest scoring.
"""

import sys
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np
import logging
import json
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from feature_engineering.transformers import engineer_features

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("EnsembleOptimized")

# Paths
TRAIN_PATH = BASE_DIR / "TRAIN.csv"
TEST_PATH = BASE_DIR / "KAGGLE.csv"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Best-in-Class Parameters
MODEL_PARAMS = dict(
    learning_rate=0.08,
    max_iter=500,
    max_depth=12,
    max_leaf_nodes=36,
    min_samples_leaf=15,
    max_bins=160,
    l2_regularization=0.07,
    early_stopping=False,
    scoring='loss'
)

# Ensemble Config
N_FOLDS = 5
SEEDS = [2050]

def train_ensemble():
    LOGGER.info("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    test_df = pd.read_csv(TEST_PATH, parse_dates=['date_time'])
    
    # Store IDs for submission
    test_ids = test_df['ID'] if 'ID' in test_df.columns else test_df.index
    
    # 1. Feature Engineering
    LOGGER.info("Applying Feature Engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # 2. Prepare Features
    # Define feature groups
    numeric_features = [
        'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'time_index',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    
    categorical_features = ['weather_main', 'weather_description']
    
    boolean_features = [
        'is_holiday', 'is_severe_weather', 'is_morning_rush', 'is_evening_rush', 
        'rush_x_severe', 'is_daylight', 'is_weekend'
    ]
    
    feature_cols = numeric_features + categorical_features + boolean_features
    
    X = train_df[feature_cols]
    y = train_df['traffic_volume']
    X_test = test_df[feature_cols]
    
    # Log Transform Target (to handle the non-negative constraint and skew)
    y_log = np.log1p(y)
    
    # 3. Build Pipeline Template
    # HGBR handles NaNs and Categoricals natively, but we need to ensure categoricals are encoded if passed as objects
    # However, sklearn's HGBR expects categorical features to be explicitly flagged or OrdinalEncoded.
    # The easiest robust way is OneHotEncoder for low-cardinality cats, or Ordinal for high.
    # Given 'weather_description' has ~38 values, OneHot is fine.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('bool', 'passthrough', boolean_features)
        ]
    )
    
    # 4. Training Loop
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(X_test))
    
    total_models = len(SEEDS) * N_FOLDS
    model_count = 0
    
    LOGGER.info(f"Starting Ensemble Training: {len(SEEDS)} Seeds x {N_FOLDS} Folds = {total_models} Models")
    
    for seed in SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
            model_count += 1
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
            
            # Build Pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', HistGradientBoostingRegressor(random_state=seed, **MODEL_PARAMS))
            ])
            
            model.fit(X_train, y_train)
            
            # Predict Validation (OOF)
            val_pred_log = model.predict(X_val)
            oof_preds[val_idx] += val_pred_log # We sum here, but since we have multiple seeds, we need to average carefully
            
            # Predict Test
            test_pred_log = model.predict(X_test)
            test_preds_accum += test_pred_log
            
            rmse_log = np.sqrt(mean_squared_error(y_val, val_pred_log))
            LOGGER.info(f"Model {model_count}/{total_models} (Seed {seed} Fold {fold+1}) - Val RMSE (Log): {rmse_log:.4f}")
            
    # Average the predictions
    # For OOF: Each sample was in the validation set exactly 'len(SEEDS)' times.
    oof_preds_avg = oof_preds / len(SEEDS)
    
    # For Test: Each sample was predicted 'total_models' times.
    test_preds_avg = test_preds_accum / total_models
    
    # Inverse Transform
    oof_preds_final = np.expm1(oof_preds_avg)
    test_preds_final = np.expm1(test_preds_avg)
    
    # Clip negatives
    oof_preds_final = np.maximum(oof_preds_final, 0)
    test_preds_final = np.maximum(test_preds_final, 0)
    
    # Calculate Final OOF RMSE
    final_rmse = np.sqrt(mean_squared_error(y, oof_preds_final))
    LOGGER.info(f"Final Ensemble OOF RMSE: {final_rmse:.4f}")
    
    # Save Artifacts
    # 1. Submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'traffic_volume': test_preds_final
    })
    sub_path = ARTIFACTS_DIR / "ensemble_optimized_submission.csv"
    submission.to_csv(sub_path, index=False)
    LOGGER.info(f"Submission saved to {sub_path}")
    
    # 2. OOF Predictions (for plotting)
    oof_df = pd.DataFrame({
        'actual': y,
        'predicted': oof_preds_final,
        'resid': y - oof_preds_final
    })
    oof_path = ARTIFACTS_DIR / "ensemble_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    LOGGER.info(f"OOF Predictions saved to {oof_path}")
    
    # 3. Metrics
    metrics = {
        'rmse': final_rmse,
        'seeds': SEEDS,
        'folds': N_FOLDS,
        'params': MODEL_PARAMS
    }
    with open(ARTIFACTS_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_ensemble()
