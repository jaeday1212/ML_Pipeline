"""
Plot Linear Model Diagnostics.
Visualizes why the Linear Model fails.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from feature_engineering.transformers import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("LinearPlots")

TRAIN_PATH = BASE_DIR / "TRAIN.csv"
PLOTS_DIR = BASE_DIR / "models" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    LOGGER.info("Loading data...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    df = engineer_features(df)
    
    numeric_features = [
        'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'time_index',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]
    categorical_features = ['weather_main', 'weather_description']
    boolean_features = [
        'is_holiday', 'is_severe_weather', 'is_morning_rush', 'is_evening_rush', 
        'rush_x_severe', 'is_daylight', 'is_weekend'
    ]
    
    X = df[numeric_features + categorical_features + boolean_features]
    y = df['traffic_volume']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('bool', 'passthrough', boolean_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    LOGGER.info("Training Linear Model...")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    
    residuals = y_val - preds
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=preds, y=y_val, alpha=0.3)
    plt.plot([0, 7500], [0, 7500], 'r--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Linear Model: Actual vs Predicted (RMSE ~887)")
    plt.savefig(PLOTS_DIR / "linear_actual_vs_predicted.png")
    plt.close()
    
    # Plot 2: Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=preds, y=residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Linear Model: Residuals vs Predicted")
    plt.savefig(PLOTS_DIR / "linear_residuals.png")
    plt.close()
    
    LOGGER.info(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
