"""
Plot Ensemble Diagnostics.
Visualizes the performance of the Optimized Ensemble using OOF predictions.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("EnsemblePlots")

ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
PLOTS_DIR = BASE_DIR / "models" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    LOGGER.info("Loading OOF predictions...")
    try:
        df = pd.read_csv(ARTIFACTS_DIR / "ensemble_oof_predictions.csv")
    except FileNotFoundError:
        LOGGER.error("OOF predictions not found. Run training first.")
        return

    # Plot 1: Actual vs Predicted
    LOGGER.info("Plotting Actual vs Predicted...")
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='predicted', y='actual', data=df, alpha=0.1, s=10, color='blue')
    
    # Perfect line
    max_val = max(df['actual'].max(), df['predicted'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    
    plt.title("Ensemble: Actual vs Predicted")
    plt.xlabel("Predicted Traffic Volume")
    plt.ylabel("Actual Traffic Volume")
    plt.savefig(PLOTS_DIR / "ensemble_actual_vs_predicted.png")
    plt.close()

    # Plot 2: Residuals vs Predicted
    LOGGER.info("Plotting Residuals vs Predicted...")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='predicted', y='resid', data=df, alpha=0.1, s=10, color='green')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Ensemble: Residuals vs Predicted")
    plt.xlabel("Predicted Traffic Volume")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.savefig(PLOTS_DIR / "ensemble_residuals_vs_predicted.png")
    plt.close()

    # Plot 3: Residual Distribution
    LOGGER.info("Plotting Residual Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['resid'], bins=100, kde=True, color='purple')
    plt.title("Ensemble: Residual Distribution")
    plt.xlabel("Residuals")
    plt.savefig(PLOTS_DIR / "ensemble_residual_dist.png")
    plt.close()
    
    LOGGER.info(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
