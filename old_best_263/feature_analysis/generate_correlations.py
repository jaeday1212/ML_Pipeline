"""Correlation diagnostics for engineered feature set."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from feature_engineering.transformers import engineer_features

LOGGER = logging.getLogger("CorrelationAnalysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TRAIN_PATH = BASE_DIR / "TRAIN.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="white", font_scale=0.95)

RAW_NUMERIC = [
    "traffic_volume",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
]

TIME_CONTINUOUS = [
    "hour",
    "day_of_week",
    "month",
    "year",
    "time_index",
]

CYCLICAL = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]

BINARY_FLAGS = [
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_daylight",
    "is_raining",
    "is_snowing",
    "is_severe_weather",
    "is_holiday",
    "rush_x_severe",
]


def load_engineered_frame() -> pd.DataFrame:
    LOGGER.info("Loading training data and applying engineer_features()")
    df = pd.read_csv(TRAIN_PATH, parse_dates=["date_time"])
    df = engineer_features(df)
    return df


def _keep_existing(df: pd.DataFrame, columns):
    return [col for col in columns if col in df.columns]


def plot_heatmap(df: pd.DataFrame, columns, title: str, filename: str, method: str = "pearson"):
    cols = _keep_existing(df, columns)
    if len(cols) < 2:
        LOGGER.warning("Skipping %s heatmap because <2 columns available", title)
        return

    corr = df[cols].corr(method=method)
    plt.figure(figsize=(min(0.8 * len(cols) + 4, 14), 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"{title} ({method.title()} correlation)")
    plt.tight_layout()
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    LOGGER.info("Saved %s", out_path.name)


def plot_target_correlations(df: pd.DataFrame, method: str, filename: str):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "traffic_volume" not in numeric_cols:
        LOGGER.warning("No traffic_volume column found for correlation plot")
        return

    numeric_cols.remove("traffic_volume")
    corrs = df[numeric_cols].corrwith(df["traffic_volume"], method=method).dropna()
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=True).index)

    plt.figure(figsize=(10, max(6, len(corrs) * 0.3)))
    corrs.plot(kind="barh", color=np.where(corrs > 0, "#1f77b4", "#d62728"))
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Traffic Volume vs Features ({method.title()} correlation)")
    plt.xlabel("Correlation Coefficient")
    plt.tight_layout()
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    LOGGER.info("Saved %s", out_path.name)

    # Persist numeric values for quick reference
    corr_df = corrs.sort_values(key=lambda s: s.abs(), ascending=False).to_frame(name=f"corr_{method}")
    corr_df.to_csv(PLOTS_DIR / filename.replace(".png", ".csv"))


def main():
    df = load_engineered_frame()

    plot_heatmap(df, RAW_NUMERIC, "Raw Weather vs Target", "correlation_raw_numeric.png")
    plot_heatmap(df, TIME_CONTINUOUS, "Time Index Features", "correlation_time_components.png")
    plot_heatmap(df, CYCLICAL, "Cyclical Encodings", "correlation_cyclical.png", method="spearman")
    plot_heatmap(df, BINARY_FLAGS + ["traffic_volume"], "Binary Flags", "correlation_binary_flags.png")

    plot_target_correlations(df, method="pearson", filename="target_corr_pearson.png")
    plot_target_correlations(df, method="spearman", filename="target_corr_spearman.png")

    LOGGER.info("Correlation diagnostics saved to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
