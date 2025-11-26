from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_TRAIN = Path("TRAIN.csv")  # this one
RAW_TEST = Path("KAGGLE.csv")
TRAIN_OUT = Path("TRAIN_CLEAN_V2.csv")
TEST_OUT = Path("KAGGLE_CLEAN_V2.csv")

CUBIC_BASE_COLS = ["hour", "temp_c", "precip_total", "clouds_all", "dayofweek"]
# *** FIX IS HERE: "ID" has been REMOVED from the redundant list ***
REDUNDANT_COLS = ["temp", "temp_k"]


def series_or_value(
    df: pd.DataFrame, column: str, fill_value: float = 0.0
) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(fill_value, index=df.index)


def series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    return series_or_value(df, column, 0.0)


def engineer_common_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["holiday"] = data["holiday"].fillna("None")
    dt = pd.to_datetime(data["date_time"], utc=False)

    data["year"] = dt.dt.year
    data["month"] = dt.dt.month
    data["day"] = dt.dt.day
    data["hour"] = dt.dt.hour
    data["dayofweek"] = dt.dt.dayofweek
    data["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    data["dayofyear"] = dt.dt.dayofyear
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["is_holiday"] = (data["holiday"] != "None").astype(int)

    data["is_peak_am"] = ((data["hour"] >= 6) & (data["hour"] <= 9)).astype(int)
    data["is_peak_pm"] = ((data["hour"] >= 15) & (data["hour"] <= 18)).astype(int)
    data["is_commute_hour"] = (data["is_peak_am"] | data["is_peak_pm"]).astype(int)

    data["temp_c"] = (data["temp"] - 32) * 5.0 / 9.0
    data["temp_k"] = data["temp_c"] + 273.15

    rain_1h = series_or_zero(data, "rain_1h")
    snow_1h = series_or_zero(data, "snow_1h")
    data["precip_total"] = rain_1h + snow_1h
    data["rain_flag"] = (rain_1h > 0).astype(int)
    data["snow_flag"] = (snow_1h > 0).astype(int)
    data["any_precip"] = (data["precip_total"] > 0).astype(int)

    precip_bins = [-0.1, 0.05, 1.0, 3.0, np.inf]
    precip_labels = ["none", "light", "moderate", "heavy"]
    data["precip_intensity_bin"] = pd.cut(
        data["precip_total"],
        bins=precip_bins,
        labels=precip_labels,
        right=True,
    ).astype(str)

    data["weather_main"] = data["weather_main"].str.lower()
    data["weather_description"] = data["weather_description"].str.replace(
        " ", "_", regex=False
    )
    data["holiday"] = data["holiday"].str.lower().str.replace(
        " ", "_", regex=False
    )

    important_weather_main = {
        "thunderstorm",
        "snow",
        "mist",
        "drizzle",
        "fog",
        "haze",
        "clear",
        "rain",
    }
    important_weather_desc = {
        "heavy_snow",
        "sky_is_clear",
        "heavy_intensity_rain",
        "light_intensity_drizzle",
        "drizzle",
        "broken_clouds",
        "light_rain",
        "proximity_thunderstorm_with_rain",
        "proximity_shower_rain",
        "shower_drizzle",
        "haze",
        "heavy_intensity_drizzle",
    }

    data["weather_main"] = data["weather_main"].where(
        data["weather_main"].isin(important_weather_main),
        "other",
    )
    data["weather_description"] = data["weather_description"].where(
        data["weather_description"].isin(important_weather_desc),
        "other",
    )

    data["is_washingtons_birthday"] = (
        data["holiday"] == "washingtons_birthday"
    ).astype(int)
    data["is_shoulder_hour"] = data["hour"].isin([10, 11, 19, 20]).astype(int)
    data["is_transition_month"] = data["month"].isin([3, 4, 9, 10]).astype(int)
    data["extreme_temp_flag"] = (
        (data["temp_c"] <= -10) | (data["temp_c"] >= 30)
    ).astype(int)
    data["is_night"] = ((data["hour"] <= 5) | (data["hour"] >= 22)).astype(int)
    data["is_precommute_hour"] = data["hour"].isin([4, 5]).astype(int)
    data["is_midday_block"] = data["hour"].between(10, 14).astype(int)
    data["is_evening_block"] = data["hour"].between(17, 21).astype(int)
    data["is_summer_month"] = data["month"].isin([6, 7, 8]).astype(int)
    data["is_winter_month"] = data["month"].isin([12, 1, 2]).astype(int)
    data["is_holiday_season"] = data["month"].isin([11, 12]).astype(int)
    data["is_month_start"] = (data["day"] <= 3).astype(int)
    data["is_month_end"] = (data["day"] >= 28).astype(int)
    data["weekofyear_quarter"] = (data["weekofyear"] // 13).astype(int)

    data["peak_am_rain"] = data["is_peak_am"] * data["rain_flag"]
    data["peak_pm_snow"] = data["is_peak_pm"] * data["snow_flag"]
    data["hour_weekend"] = data["hour"] * data["is_weekend"]
    data["weekend_precip"] = data["is_weekend"] * data["any_precip"]
    data["holiday_commute"] = data["is_holiday"] * data["is_commute_hour"]
    data["commute_precip"] = data["is_commute_hour"] * data["any_precip"]
    data["rain_commute"] = data["is_commute_hour"] * data["rain_flag"]
    data["snow_commute"] = data["is_commute_hour"] * data["snow_flag"]
    data["winter_precip"] = data["is_winter_month"] * data["any_precip"]
    data["holiday_precip"] = data["is_holiday"] * data["any_precip"]

    visibility = series_or_value(data, "visibility", np.nan)
    data["is_monthly_turn"] = data["is_month_start"] | data["is_month_end"]
    data["low_visibility"] = (visibility < 6000).astype(int)
    data["very_low_visibility"] = (visibility < 4000).astype(int)
    data["night_low_visibility"] = data["is_night"] * data["low_visibility"]
    data["commute_low_visibility"] = (
        data["is_commute_hour"] * data["low_visibility"]
    )

    if "humidity" in data.columns:
        data["humid_temp_index"] = data["temp_c"] * (data["humidity"] / 100.0)
        data["high_humidity_flag"] = (data["humidity"] >= 85).astype(int)
        data["low_humidity_flag"] = (data["humidity"] <= 40).astype(int)

    if {"wind_speed", "wind_direction"}.issubset(data.columns):
        data["wind_quadrant"] = pd.cut(
            data["wind_direction"],
            bins=[-1, 90, 180, 270, 361],
            labels=["N_E", "E_S", "S_W", "W_N"],
            right=True,
        ).astype(str)

    if "wind_speed" in data.columns:
        data["wind_speed_mph"] = data["wind_speed"] * 2.23694
        data["is_gusty_wind"] = (data["wind_speed"] >= 8).astype(int)

    data["rain_heavy_flag"] = (rain_1h >= 1.0).astype(int)
    data["snow_heavy_flag"] = (snow_1h >= 0.5).astype(int)
    data["rain_snow_mix"] = ((rain_1h > 0) & (snow_1h > 0)).astype(int)

    data = add_cubic_metrics(data, CUBIC_BASE_COLS)

    # "ID" is no longer in REDUNDANT_COLS, so it will be kept
    drop_cols = ["date_time", "holiday"] + REDUNDANT_COLS
    return data.drop(columns=drop_cols, errors="ignore")


def add_cubic_metrics(
    df: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        centered = df[col] - df[col].mean()
        df[f"{col}_centered"] = centered
        df[f"{col}_centered_sq"] = centered ** 2
        df[f"{col}_centered_cu"] = centered ** 3
    return df


def split_and_save(features: pd.DataFrame) -> None:
    train_rows = (
        features[features["dataset"] == "train"]
        .drop(columns=["dataset"], errors="ignore")
    )
    test_rows = (
        features[features["dataset"] == "test"]
        .drop(columns=["dataset"], errors="ignore")
    )

    if "traffic_volume" in train_rows.columns:
        train_rows["traffic_volume_log"] = np.log1p(
            train_rows["traffic_volume"]
        )

    train_rows.to_csv(TRAIN_OUT, index=False)
    test_rows.to_csv(TEST_OUT, index=False)


def main() -> None:
    if not RAW_TRAIN.exists() or not RAW_TEST.exists():
        missing = [
            path for path in [RAW_TRAIN, RAW_TEST] if not path.exists()
        ]
        raise FileNotFoundError(
            f"Missing required file(s): {', '.join(str(m) for m in missing)}"
        )

    train = pd.read_csv(RAW_TRAIN)
    test = pd.read_csv(RAW_TEST)

    combined = pd.concat(
        [train.assign(dataset="train"), test.assign(dataset="test")],
        ignore_index=True,
        sort=False,
    )

    engineered = engineer_common_features(combined)
    split_and_save(engineered)

    print(
        f"Wrote {TRAIN_OUT} with "
        f"{engineered[engineered['dataset']=='train'].shape[0]} rows."
    )
    print(
        f"Wrote {TEST_OUT} with "
        f"{engineered[engineered['dataset']=='test'].shape[0]} rows."
    )


if __name__ == "__main__":
    main()
