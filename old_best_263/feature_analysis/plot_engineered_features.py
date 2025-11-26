"""
Plot Engineered Features.
Visualizes the specific features added in the latest engineering step.
"""

import sys
from pathlib import Path

# Add parent directory to path to import feature_engineering
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from feature_engineering.transformers import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("PlotEngineered")

# Paths
TRAIN_PATH = BASE_DIR / "TRAIN.csv"
PLOTS_DIR = BASE_DIR / "feature_analysis" / "plots_engineered"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_process():
    LOGGER.info("Loading and transforming data...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    df = engineer_features(df)
    return df

def plot_severe_weather(df):
    LOGGER.info("Plotting Severe Weather Impact...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_severe_weather', y='traffic_volume', data=df, palette='Reds')
    plt.title("Traffic Volume: Normal vs Severe Weather")
    plt.xticks([0, 1], ['Normal', 'Severe'])
    plt.savefig(PLOTS_DIR / "impact_severe_weather.png")
    plt.close()

def plot_rush_hour(df):
    LOGGER.info("Plotting Rush Hour Impact...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='is_morning_rush', y='traffic_volume', data=df, palette='Oranges')
    plt.title("Morning Rush Hour")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='is_evening_rush', y='traffic_volume', data=df, palette='Oranges')
    plt.title("Evening Rush Hour")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "impact_rush_hour.png")
    plt.close()

def plot_interaction(df):
    LOGGER.info("Plotting Interaction (Rush x Severe)...")
    # Create a readable label for the plot
    df['condition'] = 'Normal'
    df.loc[df['is_morning_rush'] == 1, 'condition'] = 'Morning Rush'
    df.loc[df['is_evening_rush'] == 1, 'condition'] = 'Evening Rush'
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='condition', y='traffic_volume', hue='is_severe_weather', data=df, palette='Set1')
    plt.title("Traffic Volume: Rush Hour vs Severe Weather Interaction")
    plt.legend(title='Severe Weather', labels=['No', 'Yes'])
    plt.savefig(PLOTS_DIR / "impact_interaction_rush_severe.png")
    plt.close()

def plot_trend(df):
    LOGGER.info("Plotting Time Trend...")
    plt.figure(figsize=(12, 6))
    # Downsample for scatter plot clarity if needed, or just plot all
    sns.scatterplot(x='time_index', y='traffic_volume', data=df, alpha=0.1, color='gray', s=10)
    
    # Add a rolling mean trend line
    df_sorted = df.sort_values('time_index')
    df_sorted['rolling_mean'] = df_sorted['traffic_volume'].rolling(window=168, center=True).mean() # 1 week window
    plt.plot(df_sorted['time_index'], df_sorted['rolling_mean'], color='red', linewidth=2, label='7-Day Rolling Mean')
    
    plt.title("Traffic Volume over Time (Time Index)")
    plt.xlabel("Hours since start")
    plt.legend()
    plt.savefig(PLOTS_DIR / "impact_time_trend.png")
    plt.close()

def plot_daylight(df):
    LOGGER.info("Plotting Daylight Impact...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_daylight', y='traffic_volume', data=df, palette='YlOrBr')
    plt.title("Traffic Volume: Night vs Daylight")
    plt.xticks([0, 1], ['Night', 'Daylight'])
    plt.savefig(PLOTS_DIR / "impact_daylight.png")
    plt.close()

def main():
    df = load_and_process()
    
    plot_severe_weather(df)
    plot_rush_hour(df)
    plot_interaction(df)
    plot_trend(df)
    plot_daylight(df)
    
    LOGGER.info(f"All plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
