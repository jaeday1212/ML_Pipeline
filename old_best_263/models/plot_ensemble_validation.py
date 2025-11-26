"""Generate comprehensive validation plots for the optimized ensemble.

Outputs include:
1. Train residual diagnostics (scatter, distribution, by hour/day).
2. Prediction distributions comparing train vs Kaggle.
3. Residual proxy for Kaggle predictions vs train hourly baseline.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("EnsembleValidation")

TRAIN_PATH = BASE_DIR / "TRAIN.csv"
KAGGLE_PATH = BASE_DIR / "KAGGLE.csv"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
PLOTS_DIR = BASE_DIR / "models" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    kaggle_df = pd.read_csv(KAGGLE_PATH, parse_dates=['date_time'])
    oof_df = pd.read_csv(ARTIFACTS_DIR / "ensemble_oof_predictions.csv")
    submission_df = pd.read_csv(ARTIFACTS_DIR / "ensemble_optimized_submission.csv")

    if len(train_df) != len(oof_df):
        raise ValueError("Train dataframe and OOF predictions have differing lengths. Cannot align residuals.")

    # Align train + residuals
    train_with_resid = pd.concat([train_df.reset_index(drop=True), oof_df], axis=1)
    train_with_resid['hour'] = train_with_resid['date_time'].dt.hour
    train_with_resid['day_of_week'] = train_with_resid['date_time'].dt.dayofweek
    train_with_resid['month'] = train_with_resid['date_time'].dt.month

    # Attach predictions to Kaggle features
    kaggle_pred = kaggle_df.merge(submission_df, on='ID', suffixes=('_features', '_pred'))
    if 'traffic_volume_pred' in kaggle_pred.columns:
        kaggle_pred.rename(columns={'traffic_volume_pred': 'prediction'}, inplace=True)
    elif 'traffic_volume_y' in kaggle_pred.columns:
        kaggle_pred.rename(columns={'traffic_volume_y': 'prediction'}, inplace=True)
    elif 'traffic_volume' in kaggle_pred.columns:
        kaggle_pred.rename(columns={'traffic_volume': 'prediction'}, inplace=True)
    else:
        raise KeyError("Could not identify prediction column in Kaggle submission merge.")

    # Drop placeholder traffic column from Kaggle features if it exists
    drop_cols = [col for col in ['traffic_volume_features', 'traffic_volume_x'] if col in kaggle_pred.columns]
    kaggle_pred.drop(columns=drop_cols, inplace=True)

    kaggle_pred['hour'] = kaggle_pred['date_time'].dt.hour
    kaggle_pred['day_of_week'] = kaggle_pred['date_time'].dt.dayofweek

    return train_with_resid, kaggle_pred

def plot_train_residual_scatter(train_df):
    LOGGER.info("Plotting train residual scatter plots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=train_df.sample(frac=0.6, random_state=42), x='predicted', y='resid', alpha=0.25, s=15, ax=ax)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title('Train Residuals vs Predicted (OOF)')
    ax.set_xlabel('Predicted Traffic Volume')
    ax.set_ylabel('Residual (Actual - Predicted)')
    fig.savefig(PLOTS_DIR / 'ensemble_train_residuals_vs_predicted.png', dpi=150)
    plt.close(fig)

def plot_train_residual_distribution(train_df):
    LOGGER.info("Plotting train residual distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(train_df['resid'], bins=80, kde=True, color='purple', ax=ax)
    ax.set_title('Train Residual Distribution (OOF)')
    ax.set_xlabel('Residual')
    fig.savefig(PLOTS_DIR / 'ensemble_train_residual_distribution.png', dpi=150)
    plt.close(fig)

def plot_residuals_by_hour(train_df):
    LOGGER.info("Plotting residuals by hour...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=train_df, x='hour', y='resid', ax=ax, color='#6baed6')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title('Train Residuals by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Residual (Actual - Predicted)')
    fig.savefig(PLOTS_DIR / 'ensemble_train_residuals_by_hour.png', dpi=150)
    plt.close(fig)

def plot_residuals_by_day(train_df):
    LOGGER.info("Plotting residuals by day of week...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=train_df, x='day_of_week', y='resid', ax=ax, color='#fd8d3c')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_title('Train Residuals by Day of Week (0=Mon)')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Residual (Actual - Predicted)')
    fig.savefig(PLOTS_DIR / 'ensemble_train_residuals_by_day.png', dpi=150)
    plt.close(fig)

def plot_prediction_distributions(train_df, kaggle_df):
    LOGGER.info("Plotting prediction distributions (train vs Kaggle)...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(train_df['predicted'], label='Train Predictions (OOF)', fill=True, alpha=0.3, color='steelblue', ax=ax)
    sns.kdeplot(kaggle_df['prediction'], label='Kaggle Predictions', fill=True, alpha=0.3, color='darkorange', ax=ax)
    ax.set_title('Prediction Distribution: Train vs Kaggle')
    ax.set_xlabel('Predicted Traffic Volume')
    ax.legend()
    fig.savefig(PLOTS_DIR / 'ensemble_prediction_distribution_train_vs_kaggle.png', dpi=150)
    plt.close(fig)

def plot_kaggle_residual_proxy(train_df, kaggle_df):
    LOGGER.info("Plotting Kaggle residual proxy vs train residuals...")
    hourly_baseline = train_df.groupby('hour')['actual'].mean().rename('hourly_actual_mean')
    kaggle_aug = kaggle_df.merge(hourly_baseline, on='hour', how='left')
    kaggle_aug['residual_proxy'] = kaggle_aug['prediction'] - kaggle_aug['hourly_actual_mean']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(train_df['resid'], label='Train Residuals (actual)', fill=True, alpha=0.35, color='purple', ax=ax)
    sns.kdeplot(kaggle_aug['residual_proxy'], label='Kaggle Residual Proxy (pred - hourly train mean)', fill=True, alpha=0.35, color='green', ax=ax)
    ax.set_title('Residual Shape: Train vs Kaggle Proxy')
    ax.set_xlabel('Residual / Proxy Value')
    ax.legend()
    fig.savefig(PLOTS_DIR / 'ensemble_residual_shape_train_vs_kaggle_proxy.png', dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=kaggle_aug.groupby('hour')['residual_proxy'].mean().reset_index(), x='hour', y='residual_proxy', marker='o', label='Kaggle residual proxy', ax=ax2)
    sns.lineplot(data=train_df.groupby('hour')['resid'].mean().reset_index(), x='hour', y='resid', marker='o', label='Train residual mean', ax=ax2)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Mean Residuals by Hour: Train vs Kaggle Proxy')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Residual / Proxy')
    ax2.legend()
    fig2.savefig(PLOTS_DIR / 'ensemble_residuals_by_hour_train_vs_kaggle.png', dpi=150)
    plt.close(fig2)

def main():
    train_with_resid, kaggle_pred = load_data()

    plot_train_residual_scatter(train_with_resid)
    plot_train_residual_distribution(train_with_resid)
    plot_residuals_by_hour(train_with_resid)
    plot_residuals_by_day(train_with_resid)
    plot_prediction_distributions(train_with_resid, kaggle_pred)
    plot_kaggle_residual_proxy(train_with_resid, kaggle_pred)

    LOGGER.info("All validation plots saved to %s", PLOTS_DIR)

if __name__ == "__main__":
    main()
