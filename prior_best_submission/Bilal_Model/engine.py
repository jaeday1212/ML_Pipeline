from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- Setup Logging ---
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
LOGGER = logging.getLogger("hgb-lean-ensemble")
LOGGER.setLevel(logging.INFO)

# --- Artifact paths ---
ARTIFACTS = Path.cwd() / "artifacts_ensemble_LEAN"
ARTIFACTS.mkdir(exist_ok=True)
SUBMISSION_PATH = ARTIFACTS / "submission_lean_clean.csv"

# --- Parameters (Best-in-Class) ---
BASE_PARAMS = dict(
    learning_rate=0.03,
    max_iter=6000,
    max_depth=18,
    max_leaf_nodes=84,
    min_samples_leaf=15,
    max_bins=160,
    l2_regularization=0.10,
    early_stopping=False,
    scoring="neg_root_mean_squared_error",
)

# --- Ensemble controls ---

SEED_LIST = [2050, 3050, 4050, 5050, 6050, 7050]
N_FOLDS = 5

ONEHOT_KWARGS = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
    ONEHOT_KWARGS["sparse_output"] = False
else:
    ONEHOT_KWARGS["sparse"] = False


def compute_sample_weights(y: pd.Series, artifacts_dir: Path) -> np.ndarray:
    """Approximate influence weights from prior OOF residuals."""
    oof_path = artifacts_dir / "oof_preds.npy"
    if oof_path.exists():
        try:
            prev_preds = np.load(oof_path)
            if prev_preds.shape[0] == len(y):
                residuals = y.values - prev_preds
                mad = np.median(np.abs(residuals)) + 1e-6
                weights = 1.0 / (1.0 + (np.abs(residuals) / mad))
                weights = np.clip(weights, 0.1, 1.0)
                LOGGER.info("Loaded influence-style weights.")
                return weights
        except Exception:
            pass
    LOGGER.info("Falling back to uniform weights.")
    return np.ones(len(y), dtype=float)


def compute_hour_of_week_bias(
    df: pd.DataFrame, y: pd.Series, artifacts_dir: Path
) -> pd.Series:
    """Compute a small, damped corrective feature per hour-of-week from prior OOF preds.

    Returns a pandas Series indexed by hour_of_week (0..167) giving a small numeric
    correction (scaled) that can be added as a feature. If no prior OOF exists,
    returns zeros.
    """
    oof_path = artifacts_dir / "oof_preds.npy"
    hour = (df["dayofweek"].astype(int) * 24 + df["hour"].astype(int)).astype(int)

    if oof_path.exists():
        try:
            prev_preds = np.load(oof_path)
            if prev_preds.shape[0] == len(y):
                # Work in original scale for residuals (more interpretable)
                true_orig = np.expm1(y.values)
                pred_orig = np.expm1(prev_preds)
                residuals = true_orig - pred_orig

                # group mean residual by hour_of_week
                df_tmp = pd.DataFrame({"hour_of_week": hour, "residual": residuals})
                mean_resid = df_tmp.groupby("hour_of_week").residual.mean()

                # damp and scale to keep feature small relative to model inputs
                scale = np.median(true_orig) + 1.0
                bias_feature = -0.5 * mean_resid / scale
                LOGGER.info("Computed hour-of-week bias feature from prior OOF.")
                return bias_feature.reindex(range(24 * 7), fill_value=0.0)
        except Exception:
            pass

    LOGGER.info("No prior OOF for hour-of-week bias; using zeros.")
    return pd.Series(0.0, index=range(24 * 7))


def compute_tail_probability(df: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Estimate per-hour probability of being in the high-traffic tail.

    We compute a threshold (e.g., 99th percentile) on observed traffic and
    return for each hour_of_week the proportion of rows exceeding that threshold.
    This can be used as a feature to downweight / trim tail influence.
    """
    hour = (df["dayofweek"].astype(int) * 24 + df["hour"].astype(int)).astype(int)
    traffic_orig = np.expm1(y.values)

    try:
        threshold = np.percentile(traffic_orig, 99)
    except Exception:
        threshold = traffic_orig.max()

    df_tmp = pd.DataFrame(
        {"hour_of_week": hour, "is_tail": (traffic_orig > threshold).astype(int)}
    )
    prob = df_tmp.groupby("hour_of_week").is_tail.mean()
    return prob.reindex(range(24 * 7), fill_value=0.0)


def build_pipeline(features: pd.DataFrame, seed: int) -> Pipeline:
    model_params = BASE_PARAMS.copy()
    model_params["random_state"] = seed

    # Auto-detect categoricals
    categorical_cols = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_cols = [col for col in features.columns if col not in categorical_cols]

    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(**ONEHOT_KWARGS), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )
    model = HistGradientBoostingRegressor(**model_params)
    return Pipeline([("preprocessor", transformer), ("model", model)])


def create_weather_final(df: pd.DataFrame) -> np.ndarray:
    """Re-creates the validated 8-level weather feature from raw descriptions."""
    # Clean string
    desc = df["weather_description"].str.lower()

    # Logic matches our R script exactly
    conditions = [
        desc.isin(["sky is clear", "overcast clouds"]),  # Best
        desc.isin(["few clouds", "broken clouds", "scattered clouds", "haze"]),  # Cloudy/Hazy
        desc.isin(["mist", "fog"]),  # Low Viz
        desc.isin(
            [
                "light rain",
                "drizzle",
                "light intensity drizzle",
                "light rain and snow",
                "light intensity shower rain",
            ]
        ),  # Rain Light
        desc.isin(
            [
                "moderate rain",
                "heavy intensity rain",
                "freezing rain",
                "heavy intensity drizzle",
                "shower drizzle",
                "proximity shower rain",
            ]
        ),  # Rain Mod/Heavy
        desc.isin(["light snow", "light shower snow"]),  # Snow Light
        desc.isin(["snow", "heavy snow", "sleet", "shower snow"]),  # Snow Mod/Heavy
        desc.str.contains("thunderstorm"),  # Thunderstorm
    ]

    choices = [
        "Best_Conditions",
        "Cloudy_Hazy",
        "Low_Viz",
        "Rain_Light",
        "Rain_ModHeavy",
        "Snow_Light",
        "Snow_ModHeavy",
        "Thunderstorm",
    ]

    return np.select(conditions, choices, default="Other")


def main() -> None:
    LOGGER.info("--- Lean ensemble bootstrap ---")
    train_path = Path("TRAIN_CLEAN_V2.csv")
    kaggle_path = Path("KAGGLE_CLEAN_V2.csv")
    missing = [p.name for p in (train_path, kaggle_path) if not p.exists()]
    if missing:
        LOGGER.error("Missing prerequisite CSV(s): %s", ", ".join(missing))
        LOGGER.error("Run v14_step1_engineer.py from BEST_MODEL_261 to regenerate them.")
        return

    try:
        train = pd.read_csv(train_path)
        kaggle = pd.read_csv(kaggle_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Failed to load engineered datasets: %s", exc)
        return

    if not {"ID", "traffic_volume"}.issubset(kaggle.columns):
        LOGGER.error("Kaggle CSV missing required columns; found %s", list(kaggle.columns))
        return

    kaggle_ids = kaggle["ID"]
    LOGGER.info(
        "Loaded %d train rows x %d cols | %d kaggle rows.",
        len(train),
        train.shape[1],
        len(kaggle),
    )
    LOGGER.info(
        "Ensemble settings: %d folds x %d seeds (total %d fits per phase).",
        N_FOLDS,
        len(SEED_LIST),
        N_FOLDS * len(SEED_LIST),
    )

    # --- 1. CREATE WEATHER FINAL ---
    LOGGER.info("Engineering 'weather_final'...")
    train["weather_final"] = create_weather_final(train)
    kaggle["weather_final"] = create_weather_final(kaggle)

    # --- 2. THE CLEAN LIST ---
    # We select ONLY these columns. Everything else is dropped.
    keep_cols = [
        "hour",
        "dayofweek",
        "is_weekend",
        "year",
        "dayofyear",
        "month",
        "weather_final",
    ]

    X = train[keep_cols].copy()
    X_kaggle = kaggle[keep_cols].copy()

    # Ensure categoricals are strings
    for col in ["dayofweek", "month", "weather_final"]:
        X[col] = X[col].astype(str)
        X_kaggle[col] = X_kaggle[col].astype(str)

    # --- 3. THE LOG TARGET ---
    # We use the log column if it exists, or create it
    if "traffic_volume_log" in train.columns:
        y = train["traffic_volume_log"].astype(float)
    else:
        y = np.log1p(train["traffic_volume"])

    LOGGER.info(
        "Pruned feature set -> %d features: %s",
        len(keep_cols),
        keep_cols,
    )

    # --- BASE ENSEMBLE (same model as v14) ---
    sample_weights = compute_sample_weights(y, ARTIFACTS)

    oof_predictions_log = np.zeros(X.shape[0])
    prediction_counts = np.zeros(X.shape[0])
    kaggle_predictions_log = np.zeros(X_kaggle.shape[0])

    total_jobs = N_FOLDS * len(SEED_LIST)
    LOGGER.info("--- Starting Base Ensemble (%d fits) ---", total_jobs)
    jobs_done = 0

    for seed_idx, cv_seed in enumerate(SEED_LIST, start=1):
        LOGGER.info("Seed %d/%d (cv_seed=%d)", seed_idx, len(SEED_LIST), cv_seed)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            jobs_done += 1
            LOGGER.info(
                "Base fit %d/%d -> seed=%d fold=%d (%d train / %d val)",
                jobs_done,
                total_jobs,
                cv_seed,
                fold + 1,
                len(train_idx),
                len(val_idx),
            )
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            fold_weights = sample_weights[train_idx]

            pipeline = build_pipeline(X_train_fold, seed=cv_seed * 100 + fold)
            try:
                pipeline.fit(
                    X_train_fold,
                    y_train_fold,
                    model__sample_weight=fold_weights,
                )
            except Exception as exc:
                LOGGER.exception(
                    "Base fit failed for seed=%d fold=%d: %s", cv_seed, fold + 1, exc
                )
                raise

            val_preds_log = pipeline.predict(X_val_fold)
            oof_predictions_log[val_idx] += val_preds_log
            prediction_counts[val_idx] += 1

            kaggle_fold_preds_log = pipeline.predict(X_kaggle)
            kaggle_predictions_log += kaggle_fold_preds_log / (
                N_FOLDS * len(SEED_LIST)
            )

    if np.any(prediction_counts == 0):
        missing = int(prediction_counts.size - np.count_nonzero(prediction_counts))
        LOGGER.error("Found %d validation slots with zero predictions; aborting.", missing)
        return

    oof_predictions_log /= prediction_counts

    # Save base OOF so adjusted model can compute corrective features
    np.save(ARTIFACTS / "oof_preds_base.npy", oof_predictions_log)
    # also write to canonical name used by compute_sample_weights
    np.save(ARTIFACTS / "oof_preds.npy", oof_predictions_log)

    overall_rmse = np.sqrt(
        mean_squared_error(np.expm1(y), np.expm1(oof_predictions_log))
    )
    LOGGER.info("--- Base Training Complete ---")
    LOGGER.info("Base OOF RMSE (Honest Score): %.3f", overall_rmse)

    kaggle_predictions_base = np.expm1(kaggle_predictions_log)
    kaggle_predictions_base[kaggle_predictions_base < 0] = 0

    # Write base submission
    submission_base = pd.DataFrame(
        {"ID": kaggle_ids, "traffic_volume": kaggle_predictions_base}
    )
    submission_base.to_csv(SUBMISSION_PATH, index=False)
    LOGGER.info("Base submission written to %s", SUBMISSION_PATH.name)

    # --- ADJUSTED ENSEMBLE (bias-adjust + tail-trim features) ---
    LOGGER.info("Computing bias / tail features for adjusted model...")

    X_adj = X.copy()
    X_kaggle_adj = X_kaggle.copy()

    X_adj["hour_of_week"] = (
        train["dayofweek"].astype(int) * 24 + train["hour"].astype(int)
    ).astype(int)
    X_kaggle_adj["hour_of_week"] = (
        kaggle["dayofweek"].astype(int) * 24 + kaggle["hour"].astype(int)
    ).astype(int)

    # compute mapping features
    hw_bias_map = compute_hour_of_week_bias(train, y, ARTIFACTS)
    tail_prob_map = compute_tail_probability(train, y)

    # map to rows
    X_adj["hw_bias"] = X_adj["hour_of_week"].map(hw_bias_map).fillna(0.0)
    X_kaggle_adj["hw_bias"] = (
        X_kaggle_adj["hour_of_week"].map(hw_bias_map).fillna(0.0)
    )

    X_adj["tail_prob"] = X_adj["hour_of_week"].map(tail_prob_map).fillna(0.0)
    X_kaggle_adj["tail_prob"] = (
        X_kaggle_adj["hour_of_week"].map(tail_prob_map).fillna(0.0)
    )

    # Ensure categorical casting still holds
    for col in ["dayofweek", "month", "weather_final"]:
        X_adj[col] = X_adj[col].astype(str)
        X_kaggle_adj[col] = X_kaggle_adj[col].astype(str)

    # Recompute sample weights -- now compute_sample_weights will pick up base OOF we saved
    sample_weights_adj = compute_sample_weights(y, ARTIFACTS)

    oof_predictions_adj = np.zeros(X_adj.shape[0])
    prediction_counts_adj = np.zeros(X_adj.shape[0])
    kaggle_predictions_adj = np.zeros(X_kaggle_adj.shape[0])

    LOGGER.info("--- Starting Adjusted Ensemble (%d fits) ---", total_jobs)
    jobs_done = 0

    for seed_idx, cv_seed in enumerate(SEED_LIST, start=1):
        LOGGER.info("Adj seed %d/%d (cv_seed=%d)", seed_idx, len(SEED_LIST), cv_seed)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_adj, y)):
            jobs_done += 1
            LOGGER.info(
                "Adjusted fit %d/%d -> seed=%d fold=%d",
                jobs_done,
                total_jobs,
                cv_seed,
                fold + 1,
            )
            X_train_fold = X_adj.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X_adj.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # downweight rows that belong to the high-traffic tail
            tail_train = X_adj["tail_prob"].iloc[train_idx].values
            fold_weights = sample_weights_adj[train_idx] * (1.0 - 0.5 * tail_train)
            fold_weights = np.clip(fold_weights, 0.05, 1.0)

            pipeline = build_pipeline(
                X_train_fold, seed=cv_seed * 100 + fold + 9999
            )
            try:
                pipeline.fit(
                    X_train_fold,
                    y_train_fold,
                    model__sample_weight=fold_weights,
                )
            except Exception as exc:
                LOGGER.exception(
                    "Adjusted fit failed for seed=%d fold=%d: %s",
                    cv_seed,
                    fold + 1,
                    exc,
                )
                raise

            val_preds_log = pipeline.predict(X_val_fold)
            oof_predictions_adj[val_idx] += val_preds_log
            prediction_counts_adj[val_idx] += 1

            kaggle_fold_preds_log = pipeline.predict(X_kaggle_adj)
            kaggle_predictions_adj += kaggle_fold_preds_log / (
                N_FOLDS * len(SEED_LIST)
            )

    if np.any(prediction_counts_adj == 0):
        missing = int(
            prediction_counts_adj.size - np.count_nonzero(prediction_counts_adj)
        )
        LOGGER.error("Adjusted phase missing predictions for %d rows; aborting.", missing)
        return

    oof_predictions_adj /= prediction_counts_adj

    # Save adjusted OOF
    np.save(ARTIFACTS / "oof_preds_adjusted.npy", oof_predictions_adj)

    overall_rmse_adj = np.sqrt(
        mean_squared_error(np.expm1(y), np.expm1(oof_predictions_adj))
    )
    LOGGER.info("--- Adjusted Training Complete ---")
    LOGGER.info("Adjusted OOF RMSE (Honest Score): %.3f", overall_rmse_adj)

    kaggle_predictions_adj_orig = np.expm1(kaggle_predictions_adj)
    kaggle_predictions_adj_orig[kaggle_predictions_adj_orig < 0] = 0

    # Write adjusted submission
    submission_adj_path = ARTIFACTS / "submission_bias_adjusted.csv"
    submission_adj = pd.DataFrame(
        {"ID": kaggle_ids, "traffic_volume": kaggle_predictions_adj_orig}
    )
    submission_adj.to_csv(submission_adj_path, index=False)
    LOGGER.info("Adjusted submission written to %s", submission_adj_path.name)

    # Combine base and adjusted (lean bias toward adjusted predictions)
    combined = 0.1 * kaggle_predictions_base + 0.9 * kaggle_predictions_adj_orig
    combined_path = ARTIFACTS / "submission_lean_combined.csv"
    pd.DataFrame({"ID": kaggle_ids, "traffic_volume": combined}).to_csv(
        combined_path, index=False
    )
    LOGGER.info("Combined submission written to %s", combined_path.name)


if __name__ == "__main__":
    main()
