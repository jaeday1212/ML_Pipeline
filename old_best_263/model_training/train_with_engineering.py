"""
Training Script with Feature Engineering.
Imports engineering logic from feature_engineering/transformers.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import feature_engineering
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample

from feature_engineering.transformers import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("train-engineered")

# Paths
TRAIN_PATH = BASE_DIR / "TRAIN.csv"
KAGGLE_PATH = BASE_DIR / "KAGGLE.csv"
ARTIFACTS = BASE_DIR / "model_training" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SUBMISSION_PATH = ARTIFACTS / "submission.csv"
TRAIN_PRED_PATH = ARTIFACTS / "train_predictions.csv"
METRICS_PATH = ARTIFACTS / "metrics.json"

# Tuned Params (Balanced for Performance)
BASE_PARAMS = dict(
    learning_rate=0.05,
    max_iter=1500,             # Increased to allow more learning
    max_depth=12,              # Increased depth to capture interactions
    max_leaf_nodes=45,         # Increased complexity
    min_samples_leaf=20,       # Reduced slightly to allow fitting to smaller patterns
    max_bins=255,
    l2_regularization=0.5,     # Reduced regularization to reduce Bias (Underfitting)
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
CV_FOLDS = 2                   # Reduced from 5 for speed

def load_and_process():
    LOGGER.info("Loading data...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    kaggle = pd.read_csv(KAGGLE_PATH, parse_dates=['date_time'])
    
    train['is_train'] = True
    kaggle['is_train'] = False
    df = pd.concat([train, kaggle], axis=0, ignore_index=True)
    
    LOGGER.info("Applying Feature Engineering...")
    df = engineer_features(df)
    
    train_processed = df[df['is_train']].copy().drop(columns=['is_train'])
    kaggle_processed = df[~df['is_train']].copy().drop(columns=['is_train'])
    
    return train_processed, kaggle_processed

def generate_plots(y_true, y_pred, X, title_prefix):
    LOGGER.info(f"Generating plots for {title_prefix}...")
    residuals = y_true - y_pred
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=y_true, alpha=0.3, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f"{title_prefix}: Actual vs Predicted")
    plt.savefig(ARTIFACTS / f"{title_prefix}_actual_vs_predicted.png")
    plt.close()
    
    # 2. Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, color='green')
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.title(f"{title_prefix}: Residuals vs Predicted")
    plt.savefig(ARTIFACTS / f"{title_prefix}_residuals_vs_predicted.png")
    plt.close()
    
    # 3. Residuals by Hour
    if 'hour' in X.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=X['hour'], y=residuals)
        plt.axhline(0, color='r', linestyle='--')
        plt.title(f"{title_prefix}: Residuals by Hour")
        plt.savefig(ARTIFACTS / f"{title_prefix}_residuals_by_hour.png")
        plt.close()

def train_bagged_ensemble(train_df):
    features = [c for c in train_df.columns if c not in ['traffic_volume', 'date_time', 'ID']]
    categorical_features = [c for c in features if train_df[c].dtype.name == 'category']
    
    LOGGER.info(f"Features: {features}")
    
    X = train_df[features]
    y = train_df['traffic_volume']
    y_log = np.log1p(y)
    
    models = []
    LOGGER.info(f"Training {CV_FOLDS} bagged models...")
    
    for i in range(CV_FOLDS):
        seed = 42 + i
        X_sample, y_sample = resample(X, y_log, random_state=seed)
        
        params = BASE_PARAMS.copy()
        params['random_state'] = seed
        
        model = HistGradientBoostingRegressor(
            categorical_features=categorical_features,
            **params
        )
        model.fit(X_sample, y_sample)
        models.append(model)
        LOGGER.info(f"Model {i+1}/{CV_FOLDS} trained.")
        
    return models, features

def predict_ensemble(models, X):
    predictions = np.zeros(len(X))
    for model in models:
        pred_log = model.predict(X)
        pred = np.expm1(pred_log)
        predictions += pred
    return predictions / len(models)

def main():
    train_df, kaggle_df = load_and_process()
    
    models, features = train_bagged_ensemble(train_df)
    
    LOGGER.info("Evaluating...")
    X_train = train_df[features]
    y_train = train_df['traffic_volume']
    y_pred_train = predict_ensemble(models, X_train)
    
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae = mean_absolute_error(y_train, y_pred_train)
    
    LOGGER.info(f"Training RMSE: {rmse:.4f}")
    LOGGER.info(f"Training MAE: {mae:.4f}")
    
    with open(METRICS_PATH, 'w') as f:
        json.dump({'rmse': rmse, 'mae': mae}, f)
        
    generate_plots(y_train, y_pred_train, X_train, "EngineeredModel")
    
    LOGGER.info("Predicting Kaggle...")
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
