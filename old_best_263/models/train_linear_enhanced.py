"""
Train Enhanced Linear Model (Ridge Regression).
Tests the hypothesis that a Linear Model with heavy transformations (Polynomials, Scaling, OHE)
can fit the data shape better than a simple linear model.
"""

import sys
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor

from feature_engineering.transformers import engineer_features

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("LinearEnhanced")

# Paths
TRAIN_PATH = BASE_DIR / "TRAIN.csv"
TEST_PATH = BASE_DIR / "KAGGLE.csv"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate():
    LOGGER.info("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    test_df = pd.read_csv(TEST_PATH, parse_dates=['date_time'])
    
    # 1. Apply Engineering
    LOGGER.info("Applying Feature Engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    # 2. Prepare Features
    # Drop non-feature columns
    drop_cols = ['traffic_volume', 'date_time', 'holiday', 'weather_main', 'weather_description']
    # We will use 'weather_main' and 'weather_description' for OHE, so we keep them in X but drop from raw input if needed
    # Actually, let's define explicit feature lists
    
    numeric_features = [
        'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'time_index',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    
    categorical_features = ['weather_main', 'weather_description']
    
    boolean_features = [
        'is_holiday', 'is_severe_weather', 'is_morning_rush', 'is_evening_rush', 
        'rush_x_severe', 'is_daylight', 'is_weekend'
    ]
    
    X = train_df[numeric_features + categorical_features + boolean_features]
    y = train_df['traffic_volume']
    
    X_test = test_df[numeric_features + categorical_features + boolean_features]
    
    # 3. Build Pipeline
    # Linear models need Scaling.
    # They also need OHE for categoricals.
    # Polynomials help fit curves (the "shape" of the data).
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        # ('poly', PolynomialFeatures(degree=2, include_bias=False)) # Removed to check if this was causing the explosion
    ])
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', 'passthrough', boolean_features)
        ]
    )
    
    # Ridge Regression (No Log Transform)
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Wrap in TransformedTargetRegressor to handle log1p automatically
    # final_model = TransformedTargetRegressor(
    #     regressor=model,
    #     func=np.log1p,
    #     inverse_func=np.expm1
    # )
    
    # 4. Cross-Validation (TimeSeriesSplit)
    LOGGER.info("Starting Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        final_model.fit(X_train_fold, y_train_fold)
        preds = final_model.predict(X_val_fold)
        
        # Clip predictions to valid range
        preds = np.maximum(preds, 0)
        
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)
        LOGGER.info(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
    avg_rmse = np.mean(rmse_scores)
    LOGGER.info(f"Average CV RMSE: {avg_rmse:.4f}")
    
    # 5. Train on Full Data and Predict
    LOGGER.info("Retraining on full dataset...")
    final_model.fit(X, y)
    
    test_preds = final_model.predict(X_test)
    test_preds = np.maximum(test_preds, 0) # Ensure no negative traffic
    
    # 6. Save Submission
    submission = pd.DataFrame({
        'date_time': test_df['date_time'],
        'traffic_volume': test_preds
    })
    
    sub_path = ARTIFACTS_DIR / "linear_enhanced_submission.csv"
    submission.to_csv(sub_path, index=False)
    LOGGER.info(f"Submission saved to {sub_path}")
    
    # Save coefficients (if possible, tricky with Pipeline + TransformedTarget)
    try:
        # Access the inner Ridge model
        ridge_model = final_model.regressor_.named_steps['regressor']
        feature_names = final_model.regressor_.named_steps['preprocessor'].get_feature_names_out()
        coefs = pd.DataFrame({
            'feature': feature_names,
            'coefficient': ridge_model.coef_
        })
        coefs['abs_coef'] = coefs['coefficient'].abs()
        coefs = coefs.sort_values('abs_coef', ascending=False)
        coefs.to_csv(ARTIFACTS_DIR / "linear_coefficients.csv", index=False)
        LOGGER.info("Coefficients saved.")
    except Exception as e:
        LOGGER.warning(f"Could not save coefficients: {e}")

if __name__ == "__main__":
    train_and_evaluate()
