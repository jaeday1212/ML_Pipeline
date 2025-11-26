"""Feature engineering utilities for the hackathon traffic models."""

from typing import Iterable

import numpy as np
import pandas as pd

CUBIC_BASE_COLS: list[str] = [
    "hour",
    "temp",
    "clouds_all",
    "dayofweek_numeric",
]

PRECIP_BINS = [-0.1, 0.05, 1.0, 3.0, np.inf]
PRECIP_LABELS = ["none", "light", "moderate", "heavy"]

IMPORTANT_WEATHER_MAIN = {
    "thunderstorm",
    "snow",
    "mist",
    "drizzle",
    "fog",
    "haze",
    "clear",
    "rain",
}

IMPORTANT_WEATHER_DESC = {
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

SEVERE_KEYWORDS = ("heavy", "thunderstorm", "sleet", "shower", "squall")


def _series_or_value(df: pd.DataFrame, column: str, fill_value: float = 0.0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(fill_value, index=df.index)


def _series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    return _series_or_value(df, column, 0.0)


def _add_cubic_metrics(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        centered = df[col] - df[col].mean()
        df[f"{col}_centered"] = centered
        df[f"{col}_centered_sq"] = centered**2
        df[f"{col}_centered_cu"] = centered**3
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a rich feature bank, then prune it down to the seven locked model features."""

    data = df.copy()

    if "date_time" not in data.columns:
        raise KeyError("Expected 'date_time' column in input frame.")

    if not np.issubdtype(data["date_time"].dtype, np.datetime64):
        data["date_time"] = pd.to_datetime(data["date_time"], utc=False, errors="coerce")

    if "holiday" in data.columns:
        data["holiday"] = data["holiday"].fillna("None")
    else:
        data["holiday"] = "None"

    data["weather_main"] = (
        data.get("weather_main", "other").fillna("other").astype(str).str.lower()
    )
    data["weather_description"] = (
        data.get("weather_description", "other").fillna("other").astype(str)
    )

    dt = data["date_time"]
    data["hour"] = dt.dt.hour
    data["dayofweek_numeric"] = dt.dt.dayofweek
    data["year"] = dt.dt.year
    data["dayofyear"] = dt.dt.dayofyear
    data["month_numeric"] = dt.dt.month
    data["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    data["day"] = dt.dt.day

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["day_sin"] = np.sin(2 * np.pi * data["dayofweek_numeric"] / 7)
    data["day_cos"] = np.cos(2 * np.pi * data["dayofweek_numeric"] / 7)
    data["month_sin"] = np.sin(2 * np.pi * data["month_numeric"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month_numeric"] / 12)

    data["is_weekend"] = data["dayofweek_numeric"].isin([5, 6]).astype(int)
    data["is_morning_rush"] = (
        (data["hour"].isin([7, 8, 9])) & (data["is_weekend"] == 0)
    ).astype(int)
    data["is_evening_rush"] = (
        (data["hour"].isin([16, 17, 18])) & (data["is_weekend"] == 0)
    ).astype(int)
    data["is_daylight"] = data["hour"].between(6, 20).astype(int)

    data["is_shoulder_hour"] = data["hour"].isin([10, 11, 19, 20]).astype(int)
    data["is_precommute_hour"] = data["hour"].isin([4, 5]).astype(int)
    data["is_midday_block"] = data["hour"].between(10, 14).astype(int)
    data["is_evening_block"] = data["hour"].between(17, 21).astype(int)

    data["is_holiday_flag"] = (data["holiday"].str.lower() != "none").astype(int)
    data["is_washingtons_birthday"] = (
        data["holiday"].str.lower() == "washingtons_birthday"
    ).astype(int)

    temp_f = _series_or_value(data, "temp", np.nan)
    data["temp_c"] = (temp_f - 32.0) * 5.0 / 9.0
    data["temp_k"] = data["temp_c"] + 273.15

    rain_1h = _series_or_zero(data, "rain_1h")
    snow_1h = _series_or_zero(data, "snow_1h")

    data["precip_total"] = rain_1h + snow_1h
    data["rain_flag"] = (rain_1h > 0).astype(int)
    data["snow_flag"] = (snow_1h > 0).astype(int)
    data["any_precip"] = (data["precip_total"] > 0).astype(int)
    data["rain_heavy_flag"] = (rain_1h >= 1.0).astype(int)
    data["snow_heavy_flag"] = (snow_1h >= 0.5).astype(int)
    data["rain_snow_mix"] = ((rain_1h > 0) & (snow_1h > 0)).astype(int)

    data["precip_intensity_bin"] = pd.cut(
        data["precip_total"], bins=PRECIP_BINS, labels=PRECIP_LABELS, right=True
    ).astype(str)

    data["is_peak_am"] = (
        data["hour"].between(6, 9) & (data["is_holiday_flag"] == 0)
    ).astype(int)
    data["is_peak_pm"] = (
        data["hour"].between(15, 18) & (data["is_holiday_flag"] == 0)
    ).astype(int)
    data["is_commute_hour"] = (data["is_peak_am"] | data["is_peak_pm"]).astype(int)

    data["is_transition_month"] = data["month_numeric"].isin([3, 4, 9, 10]).astype(int)
    data["is_summer_month"] = data["month_numeric"].isin([6, 7, 8]).astype(int)
    data["is_winter_month"] = data["month_numeric"].isin([12, 1, 2]).astype(int)
    data["is_holiday_season"] = data["month_numeric"].isin([11, 12]).astype(int)
    data["is_month_start"] = (data["day"] <= 3).astype(int)
    data["is_month_end"] = (data["day"] >= 28).astype(int)
    data["is_monthly_turn"] = (data["is_month_start"] | data["is_month_end"]).astype(int)
    data["weekofyear_quarter"] = (data["weekofyear"] // 13).astype(int)

    data["hour_of_week"] = data["dayofweek_numeric"] * 24 + data["hour"]
    data["hour_weekend"] = data["hour"] * data["is_weekend"]
    data["weekend_precip"] = data["is_weekend"] * data["any_precip"]
    data["holiday_commute"] = data["is_holiday_flag"] * data["is_commute_hour"]
    data["commute_precip"] = data["is_commute_hour"] * data["any_precip"]
    data["rain_commute"] = data["is_commute_hour"] * data["rain_flag"]
    data["snow_commute"] = data["is_commute_hour"] * data["snow_flag"]
    data["winter_precip"] = data["is_winter_month"] * data["any_precip"]
    data["holiday_precip"] = data["is_holiday_flag"] * data["any_precip"]

    visibility = _series_or_value(data, "visibility", np.nan)
    data["low_visibility"] = (visibility < 6000).astype(int)
    data["very_low_visibility"] = (visibility < 4000).astype(int)
    data["night_low_visibility"] = (
        data["is_daylight"].eq(0).astype(int) * data["low_visibility"]
    )
    data["commute_low_visibility"] = data["is_commute_hour"] * data["low_visibility"]

    if "humidity" in data.columns:
        humidity = _series_or_value(data, "humidity", np.nan)
        data["humid_temp_index"] = data["temp_c"] * (humidity / 100.0)
        data["high_humidity_flag"] = (humidity >= 85).astype(int)
        data["low_humidity_flag"] = (humidity <= 40).astype(int)

    if {"wind_speed", "wind_direction"}.issubset(data.columns):
        data["wind_quadrant"] = pd.cut(
            data["wind_direction"],
            bins=[-1, 90, 180, 270, 361],
            labels=["N_E", "E_S", "S_W", "W_N"],
            right=True,
        ).astype(str)

    if "wind_speed" in data.columns:
        speed = _series_or_value(data, "wind_speed", 0.0)
        data["wind_speed_mph"] = speed * 2.23694
        data["is_gusty_wind"] = (speed >= 8).astype(int)

    data["is_night"] = ((data["hour"] <= 5) | (data["hour"] >= 22)).astype(int)
    data["time_index"] = (dt - dt.min()).dt.total_seconds() / 3600.0

    desc_lower = data["weather_description"].str.lower()
    conditions = [
        desc_lower.isin(["sky is clear", "overcast clouds"]),
        desc_lower.isin(["few clouds", "broken clouds", "scattered clouds", "haze"]),
        desc_lower.isin(["mist", "fog"]),
        desc_lower.isin([
            "light rain",
            "drizzle",
            "light intensity drizzle",
            "light rain and snow",
            "light intensity shower rain",
        ]),
        desc_lower.isin([
            "moderate rain",
            "heavy intensity rain",
            "freezing rain",
            "heavy intensity drizzle",
            "shower drizzle",
            "proximity shower rain",
        ]),
        desc_lower.isin(["light snow", "light shower snow"]),
        desc_lower.isin(["snow", "heavy snow", "sleet", "shower snow"]),
        desc_lower.str.contains("thunderstorm", na=False),
    ]
    labels = [
        "Best_Conditions",
        "Cloudy_Hazy",
        "Low_Viz",
        "Rain_Light",
        "Rain_ModHeavy",
        "Snow_Light",
        "Snow_ModHeavy",
        "Thunderstorm",
    ]
    data["weather_final"] = np.select(conditions, labels, default="Other")

    data["weather_main"] = data["weather_main"].where(
        data["weather_main"].isin(IMPORTANT_WEATHER_MAIN), "other"
    )
    data["weather_description"] = desc_lower.where(
        desc_lower.isin(IMPORTANT_WEATHER_DESC), "other"
    )

    data["is_raining"] = (rain_1h > 0).astype(int)
    data["is_snowing"] = (snow_1h > 0).astype(int)
    data["is_severe_weather"] = desc_lower.apply(
        lambda x: int(any(key in x for key in SEVERE_KEYWORDS))
    )
    data["rush_x_severe"] = (
        data["is_morning_rush"] * data["is_severe_weather"]
        + data["is_evening_rush"] * data["is_severe_weather"]
    )

    data = _add_cubic_metrics(data, CUBIC_BASE_COLS)

    categorical_cols = [
        "weather_main",
        "weather_description",
        "holiday",
        "precip_intensity_bin",
        "wind_quadrant",
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")

    data["dayofweek"] = data["dayofweek_numeric"].astype(str)
    data["month"] = data["month_numeric"].astype(str)
    data["weather_final"] = data["weather_final"].astype("category")

    final_feature_cols = [
        "hour",
        "dayofweek",
        "year",
        "dayofyear",
        "weather_final",
        "temp",
        "clouds_all",
    ]

    missing_final = [col for col in final_feature_cols if col not in data.columns]
    if missing_final:
        raise KeyError(f"Missing required feature column(s): {missing_final}")

    return data[final_feature_cols].copy()
