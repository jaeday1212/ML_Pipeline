"""Scatter + smooth-line plot of log traffic volume by hour of day.

Run with: python plot_log_traffic_by_hour.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_CANDIDATES = [
    Path("TRAIN_CLEAN_V2.csv"),
    Path("BEST_MODEL_261") / "TRAIN_CLEAN_V2.csv",
    Path("bilal_pipeline") / "TRAIN_CLEAN_V2.csv",
]
DATE_COLUMN = "date_time"
TARGET_COLUMN = "traffic_volume"
LOG_FN = np.log1p  # change if you prefer plain np.log
MAX_POINTS = 100_000  # downsample for faster scatter drawing
OUTPUT_PATH = Path("log_traffic_volume_by_hour.png")


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the TRAIN_CLEAN_V2.csv candidates exist: "
        + ", ".join(str(p) for p in paths)
    )


def load_train_frame() -> pd.DataFrame:
    path = _first_existing(TRAIN_CANDIDATES)
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"'{TARGET_COLUMN}' missing from {path}")
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"'{DATE_COLUMN}' missing from {path}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df["hour_of_day"] = df[DATE_COLUMN].dt.hour
    df["log_traffic"] = LOG_FN(df[TARGET_COLUMN].clip(lower=0))
    return df


def make_plot(df: pd.DataFrame) -> None:
    if len(df) > MAX_POINTS:
        df_plot = df.sample(MAX_POINTS, random_state=42)
    else:
        df_plot = df

    hourly_mean = (
        df.groupby("hour_of_day")["log_traffic"]
        .mean()
        .reset_index()
        .sort_values("hour_of_day")
    )

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=df_plot,
        x="hour_of_day",
        y="log_traffic",
        alpha=0.2,
        s=10,
        edgecolor=None,
    )
    sns.lineplot(
        data=hourly_mean,
        x="hour_of_day",
        y="log_traffic",
        color="crimson",
        linewidth=2,
        label="Hourly mean (log scale)",
    )
    plt.title("Log traffic volume by hour of day")
    plt.xlabel("Hour of day (0 = midnight)")
    plt.ylabel("log1p(traffic_volume)")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"Saved plot -> {OUTPUT_PATH}")


def main() -> None:
    df = load_train_frame()
    print(f"Loaded {len(df):,} rows")
    make_plot(df)


if __name__ == "__main__":
    main()
