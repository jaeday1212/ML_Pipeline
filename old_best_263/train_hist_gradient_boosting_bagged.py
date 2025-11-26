"""Bagged ensemble of HistGradientBoosting pipelines with differing seeds."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample

from feature_engineering.transformers import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("hgb-bagged")

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = WORKSPACE_DIR / "TRAIN.csv"
KAGGLE_PATH = WORKSPACE_DIR / "KAGGLE.csv"
ARTIFACTS = WORKSPACE_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

SUBMISSION_PATH = ARTIFACTS / "hist_gradient_boosting_bagged_submission.csv"
TRAIN_PRED_PATH = ARTIFACTS / "hist_gradient_boosting_bagged_train_predictions.csv"
METRICS_PATH = ARTIFACTS / "hist_gradient_boosting_bagged_metrics.json"

# Tuned Params (Conservative to prevent overfitting)
BASE_PARAMS = dict(
    learning_rate=0.05,
    max_iter=500,              # Reduced from 1000 for speed
    max_depth=8,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    max_bins=255,
    l2_regularization=1.5,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
CV_FOLDS = 5

def load_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and apply shared feature engineering pipeline."""
    LOGGER.info("Loading data...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    kaggle = pd.read_csv(KAGGLE_PATH, parse_dates=['date_time'])

    train['is_train'] = True
    kaggle['is_train'] = False
    df = pd.concat([train, kaggle], axis=0, ignore_index=True)

    LOGGER.info("Engineering features via feature_engineering.transformers...")
    df = engineer_features(df)

    train_processed = df[df['is_train']].copy().drop(columns=['is_train'])
    kaggle_processed = df[~df['is_train']].copy().drop(columns=['is_train'])

    return train_processed, kaggle_processed

def generate_plots(y_true, y_pred, X, title_prefix):
    """
    Generates diagnostic plots.
    """
    LOGGER.info(f"Generating plots for {title_prefix}...")
    residuals = y_true - y_pred
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=y_true, alpha=0.3, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Predicted Traffic Volume")
    plt.ylabel("Actual Traffic Volume")
    plt.title(f"{title_prefix}: Actual vs Predicted")
    plt.savefig(ARTIFACTS / f"{title_prefix}_actual_vs_predicted.png")
    plt.close()
    
    # 2. Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, color='green')
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Traffic Volume")
    plt.ylabel("Residuals")
    plt.title(f"{title_prefix}: Residuals vs Predicted")
    plt.savefig(ARTIFACTS / f"{title_prefix}_residuals_vs_predicted.png")
    plt.close()
    
    # 3. Residual Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title(f"{title_prefix}: Residual Distribution")
    plt.savefig(ARTIFACTS / f"{title_prefix}_residual_dist.png")
    plt.close()
    
    # 4. Residuals vs Hour
    if 'hour' in X.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=X['hour'], y=residuals)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f"{title_prefix}: Residuals by Hour")
        plt.savefig(ARTIFACTS / f"{title_prefix}_residuals_by_hour.png")
        plt.close()

def train_bagged_ensemble(train_df: pd.DataFrame) -> Tuple[List[HistGradientBoostingRegressor], List[str]]:
    """Train a bagged ensemble of HistGradientBoostingRegressors."""
    # Select features - exclude ID, target, and original date_time
    features = [c for c in train_df.columns if c not in ['traffic_volume', 'date_time', 'ID']]
    
    # Identify categorical features for HGBR
    categorical_features = [c for c in features if train_df[c].dtype.name == 'category']
    LOGGER.info(f"Features: {features}")
    LOGGER.info(f"Categorical Features: {categorical_features}")
    
    X = train_df[features]
    y = train_df['traffic_volume']
    
    # CRITICAL: Log transform to fix sigmoidal residuals and handle non-negative constraint
    y_log = np.log1p(y)
    
    models = []
    
    LOGGER.info(f"Training {CV_FOLDS} bagged models...")
    for i in range(CV_FOLDS):
        seed = 42 + i
        # Bootstrap sample (Bagging)
        X_sample, y_sample = resample(X, y_log, random_state=seed)
        
        # Update params with seed
        params = BASE_PARAMS.copy()
        params['random_state'] = seed
        
        model = HistGradientBoostingRegressor(
            categorical_features=categorical_features,
            **params
        )
        model.fit(X_sample, y_sample)
        models.append(model)
        if (i + 1) % 2 == 0:
            LOGGER.info(f"Model {i+1}/{CV_FOLDS} trained.")
        
    return models, features

def predict_ensemble(models: List[HistGradientBoostingRegressor], X: pd.DataFrame) -> np.ndarray:
    """Generate averaged predictions from the ensemble."""
    predictions = np.zeros(len(X))
    for model in models:
        pred_log = model.predict(X)
        pred = np.expm1(pred_log) # Inverse log transform
        predictions += pred
    return predictions / len(models)

def main():
    train_df, kaggle_df = load_and_preprocess()
    
    models, features = train_bagged_ensemble(train_df)
    
    # Evaluate on training data
    LOGGER.info("Generating training predictions...")
    X_train = train_df[features]
    y_train = train_df['traffic_volume']
    
    y_pred_train = predict_ensemble(models, X_train)
    
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae = mean_absolute_error(y_train, y_pred_train)
    
    LOGGER.info(f"Training RMSE: {rmse:.4f}")
    LOGGER.info(f"Training MAE: {mae:.4f}")
    
    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump({'rmse': rmse, 'mae': mae}, f)
        
    # Save training predictions for analysis (Residuals!)
    train_pred_df = pd.DataFrame({
        'ID': train_df['ID'],
        'actual': y_train,
        'predicted': y_pred_train,
        'residual': y_train - y_pred_train
    })
    train_pred_df.to_csv(TRAIN_PRED_PATH, index=False)
    LOGGER.info(f"Training predictions saved to {TRAIN_PRED_PATH}")
    
    # Generate Plots
    generate_plots(y_train, y_pred_train, X_train, "BoostedEnsemble")

    # Kaggle Submission
    LOGGER.info("Generating Kaggle predictions...")
    X_test = kaggle_df[features]
    y_pred_test = predict_ensemble(models, X_test)
    
    submission = pd.DataFrame({
        'ID': kaggle_df['ID'],
        'traffic_volume': y_pred_test
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    LOGGER.info(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()
