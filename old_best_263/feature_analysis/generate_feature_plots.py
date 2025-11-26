"""
Feature Analysis Script.
Generates extensive plots to inform feature engineering decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("FeatureAnalysis")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "TRAIN.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def load_data():
    LOGGER.info("Loading dataset...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['date_time'])
    return df

def plot_target_distribution(df):
    LOGGER.info("Plotting Target Distribution...")
    plt.figure(figsize=(12, 6))
    sns.histplot(df['traffic_volume'], kde=True, color='teal', bins=50)
    plt.title("Distribution of Traffic Volume")
    plt.xlabel("Traffic Volume")
    plt.ylabel("Count")
    plt.savefig(PLOTS_DIR / "01_traffic_volume_distribution.png")
    plt.close()

    # Log Transformed Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(np.log1p(df['traffic_volume']), kde=True, color='purple', bins=50)
    plt.title("Distribution of Log(Traffic Volume + 1)")
    plt.xlabel("Log(Traffic Volume)")
    plt.ylabel("Count")
    plt.savefig(PLOTS_DIR / "02_log_traffic_volume_distribution.png")
    plt.close()

def plot_time_series_analysis(df):
    LOGGER.info("Plotting Time Series Analysis...")
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year

    # 1. Traffic by Hour
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='hour', y='traffic_volume', data=df, palette='viridis')
    plt.title("Traffic Volume by Hour of Day")
    plt.savefig(PLOTS_DIR / "03_traffic_by_hour.png")
    plt.close()

    # 2. Traffic by Day of Week
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='day_of_week', y='traffic_volume', data=df, palette='coolwarm')
    plt.title("Traffic Volume by Day of Week (0=Mon, 6=Sun)")
    plt.savefig(PLOTS_DIR / "04_traffic_by_day_of_week.png")
    plt.close()

    # 3. Traffic by Month
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='month', y='traffic_volume', data=df, palette='magma')
    plt.title("Traffic Volume by Month")
    plt.savefig(PLOTS_DIR / "05_traffic_by_month.png")
    plt.close()

    # 4. Traffic by Year (Trend)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='year', y='traffic_volume', data=df, marker='o')
    plt.title("Average Traffic Volume by Year")
    plt.savefig(PLOTS_DIR / "06_traffic_by_year_trend.png")
    plt.close()

def plot_categorical_analysis(df):
    LOGGER.info("Plotting Categorical Analysis...")
    
    # 1. Weather Main
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='weather_main', y='traffic_volume', data=df)
    plt.xticks(rotation=45)
    plt.title("Traffic Volume by Weather Main")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_traffic_by_weather_main.png")
    plt.close()

    # 2. Holiday
    # Filter out 'None' to see the impact of actual holidays
    holidays_only = df[df['holiday'] != 'None']
    if not holidays_only.empty:
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='holiday', y='traffic_volume', data=holidays_only)
        plt.xticks(rotation=90)
        plt.title("Traffic Volume by Holiday Type")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "08_traffic_by_holiday.png")
        plt.close()
    
    # Holiday vs Non-Holiday
    df['is_holiday'] = (df['holiday'] != 'None').astype(int)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='is_holiday', y='traffic_volume', data=df)
    plt.title("Traffic Volume: Holiday vs Non-Holiday")
    plt.savefig(PLOTS_DIR / "09_traffic_holiday_vs_non.png")
    plt.close()

def plot_numerical_correlations(df):
    LOGGER.info("Plotting Numerical Correlations...")
    # Select numerical columns
    numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
    numeric_df = df[numeric_cols]
    
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Numerical Features")
    plt.savefig(PLOTS_DIR / "10_numerical_correlation_matrix.png")
    plt.close()

    # Pairplot (Scatter matrix) - Sampled for speed
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    sns.pairplot(sample_df[numeric_cols])
    plt.savefig(PLOTS_DIR / "11_numerical_pairplot.png")
    plt.close()

def plot_variance_analysis(df):
    LOGGER.info("Plotting Variance Analysis...")
    numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
    numeric_df = df[numeric_cols]
    
    # Normalize
    normalized_df = (numeric_df - numeric_df.mean()) / numeric_df.std()
    variances = normalized_df.var().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=variances.values, y=variances.index, palette='viridis')
    plt.title("Normalized Feature Variance (Information Content)")
    plt.xlabel("Variance (Standardized)")
    plt.savefig(PLOTS_DIR / "12_feature_variance.png")
    plt.close()

def main():
    df = load_data()
    
    plot_target_distribution(df)
    plot_time_series_analysis(df)
    plot_categorical_analysis(df)
    plot_numerical_correlations(df)
    plot_variance_analysis(df)
    
    LOGGER.info(f"All plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
