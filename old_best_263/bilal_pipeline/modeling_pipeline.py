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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("hgb-bagged-ensemble")

ARTIFACTS = Path.cwd() / "artifacts_ensemble_5fold_diverse"
ARTIFACTS.mkdir(exist_ok=True)

SUBMISSION_PATH = ARTIFACTS / "bagged_ensemble_5fold_diverse_submission.csv"
METRICS_PATH = ARTIFACTS / "bagged_ensemble_5fold_diverse_metrics.json"

BASE_PARAMS = dict(
    learning_rate=0.08,
    max_iter=1250,
    max_depth=16,
    max_leaf_nodes=36,
    min_samples_leaf=15,
    max_bins=160,
    l2_regularization=0.07,
    early_stopping=False,
)

N_FOLDS = 10
SEED_LIST = [2050, 3050]

ONEHOT_KWARGS = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
    ONEHOT_KWARGS["sparse_output"] = False
else:
    ONEHOT_KWARGS["sparse"] = False


def compute_sample_weights(y: pd.Series, artifacts_dir: Path) -> np.ndarray:
    oof_path = artifacts_dir / "oof_preds.npy"
    if oof_path.exists():
        try:
            prev_preds = np.load(oof_path)
            if prev_preds.shape[0] == len(y):
                residuals = y.values - prev_preds
                mad = np.median(np.abs(residuals)) + 1e-6
                weights = 1.0 / (1.0 + (np.abs(residuals) / mad))
                weights = np.clip(weights, 0.1, 1.0)
                LOGGER.info("Loaded influence-style weights from prior OOF residuals.")
                return weights
            LOGGER.warning("OOF predictions shape %s does not match y (len=%d).", prev_preds.shape, len(y))
        except Exception as exc:
            LOGGER.warning("Failed to compute weights from prior OOF predictions: %s", exc)

    LOGGER.info("Falling back to uniform weights (no prior OOF residuals).")
    return np.ones(len(y), dtype=float)


def build_pipeline(features: pd.DataFrame, seed: int) -> Pipeline:
    model_params = BASE_PARAMS.copy()
    model_params["random_state"] = seed

    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in features.columns if col not in categorical_cols]

    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(**ONEHOT_KWARGS), categorical_cols),
            ("numeric", "passthrough", numeric_cols),
        ]
    )

    model = HistGradientBoostingRegressor(**model_params)
    return Pipeline([("preprocessor", transformer), ("model", model)])


def main() -> None:
    LOGGER.info("Loading cleaned data from here.py outputs...")
    train = pd.read_csv("TRAIN_CLEAN_V2.csv")
    try:
        kaggle = pd.read_csv("KAGGLE_CLEAN_V2.csv")
        kaggle_ids = kaggle["ID"].copy()
    except FileNotFoundError as exc:
        LOGGER.error("Missing cleaned test file: %s", exc)
        LOGGER.error("Run python here.py first to regenerate the cleaned CSVs.")
        return

    drop_cols = ["traffic_volume", "ID", "traffic_volume_log"]
    X = train.drop(columns=drop_cols, errors="ignore")
    y = train["traffic_volume"].astype(float)
    X_kaggle = kaggle.drop(columns=drop_cols, errors="ignore")

    sample_weights = compute_sample_weights(y, ARTIFACTS)

    oof_predictions = np.zeros(X.shape[0])
    prediction_counts = np.zeros(X.shape[0])
    kaggle_predictions = np.zeros(X_kaggle.shape[0])
    fold_metrics = []

    LOGGER.info("--- Starting %d-Fold x %d-Seed Ensemble ---", N_FOLDS, len(SEED_LIST))

    for seed_idx, cv_seed in enumerate(SEED_LIST, start=1):
        LOGGER.info("=== Seed %d/%d (random_state=%d) ===", seed_idx, len(SEED_LIST), cv_seed)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            LOGGER.info("--- Fold %d/%d (Seed=%d) ---", fold + 1, N_FOLDS, cv_seed)

            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            pipeline_seed = cv_seed * 100 + fold
            pipeline = build_pipeline(X_train_fold, seed=pipeline_seed)
            LOGGER.info("Training model on %d samples...", len(X_train_fold))
            fold_weights = sample_weights[train_idx]
            pipeline.fit(X_train_fold, y_train_fold, model__sample_weight=fold_weights)

            val_preds = pipeline.predict(X_val_fold)
            oof_predictions[val_idx] += val_preds
            prediction_counts[val_idx] += 1

            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
            fold_metrics.append({"seed": cv_seed, "fold": fold + 1, "rmse": fold_rmse})
            LOGGER.info("Seed %d Fold %d Validation RMSE: %.3f", cv_seed, fold + 1, fold_rmse)

            kaggle_fold_preds = pipeline.predict(X_kaggle)
            kaggle_predictions += kaggle_fold_preds / (N_FOLDS * len(SEED_LIST))

    prediction_counts[prediction_counts == 0] = 1
    oof_predictions /= prediction_counts

    np.save(ARTIFACTS / "oof_preds.npy", oof_predictions)
    np.save(ARTIFACTS / "y_true.npy", y.values)

    overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    LOGGER.info("--- Ensemble Training Complete ---")
    LOGGER.info("Overall OOF RMSE (honest score): %.3f", overall_rmse)

    metrics = {
        "overall_oof_rmse": overall_rmse,
        "base_model_params": BASE_PARAMS,
        "n_folds": N_FOLDS,
        "seeds": SEED_LIST,
        "fold_metrics": fold_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    submission = pd.DataFrame({"ID": kaggle_ids, "traffic_volume": kaggle_predictions})
    submission.to_csv(SUBMISSION_PATH, index=False)
    LOGGER.info("Diverse ensemble submission ready at %s", SUBMISSION_PATH.name)


if __name__ == "__main__":
    main()
