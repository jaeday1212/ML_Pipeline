""# Auto-generated from analysis.ipynb
# Each '# %% Cell n' marker matches the notebook's code cell order.
# %% Cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load Data
train_df = pd.read_csv('TRAIN.csv', parse_dates=['date_time'])
kaggle_df = pd.read_csv('KAGGLE.csv', parse_dates=['date_time'])

print("Train Shape:", train_df.shape)
print("Kaggle Shape:", kaggle_df.shape)
display(train_df.head())


# %% Cell 3
def extract_date_features(df):
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

train_df = extract_date_features(train_df)
kaggle_df = extract_date_features(kaggle_df)


# %% Cell 5
plt.figure(figsize=(10, 6))
sns.histplot(train_df['traffic_volume'], kde=True, bins=50)
plt.title('Distribution of Traffic Volume')
plt.show()

print("Skewness:", train_df['traffic_volume'].skew())
print("Kurtosis:", train_df['traffic_volume'].kurt())


# %% Cell 7
# Hourly Pattern
plt.figure(figsize=(12, 6))
sns.lineplot(data=train_df, x='hour', y='traffic_volume', hue='is_weekend', estimator='mean')
plt.title('Average Traffic Volume by Hour (Weekday vs Weekend)')
plt.xticks(range(0, 24))
plt.show()

# Day of Week Pattern
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='day_of_week', y='traffic_volume')
plt.title('Traffic Volume by Day of Week')
plt.show()


# %% Cell 9
plt.figure(figsize=(14, 6))
sns.boxplot(data=train_df, x='weather_main', y='traffic_volume')
plt.xticks(rotation=45)
plt.title('Traffic Volume vs Weather Main')
plt.show()

# Temperature vs Traffic
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='temp', y='traffic_volume', alpha=0.3)
plt.title('Traffic Volume vs Temperature')
plt.show()


# %% Cell 11
# Correlation Matrix
numeric_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day_of_week', 'month']
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# %% Cell 13
# Prepare Data
features = ['hour', 'day_of_week', 'month', 'year', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'is_weekend']
# Simple encoding for categorical weather if we wanted to include them, but let's stick to numeric/time for baseline
# Adding weather_main as categorical
train_df['weather_main_code'] = train_df['weather_main'].astype('category').cat.codes
kaggle_df['weather_main_code'] = kaggle_df['weather_main'].astype('category').cat.codes

features.append('weather_main_code')

X = train_df[features]
y = train_df['traffic_volume']

# Log Transform Target
y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train Model
model = HistGradientBoostingRegressor(random_state=42, categorical_features=[len(features)-1]) # weather_main_code is last
model.fit(X_train, y_train_log)

# Evaluate
y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)
y_val = np.expm1(y_val_log)

print("R2 Score:", r2_score(y_val, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))


# %% Cell 15
residuals = y_val - y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Traffic Volume')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Predicted Values')
plt.show()


# %% Cell 17
X_test = kaggle_df[features]
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({'ID': kaggle_df['ID'], 'traffic_volume': test_pred})
submission.to_csv('submission_analysis.csv', index=False)
print("Submission saved to submission_analysis.csv")


# %% Cell 18
# Sanity Check: Distribution of Predictions vs Training Data
plt.figure(figsize=(10, 6))
sns.kdeplot(y, label='Training Data (Actual)', fill=True, alpha=0.3)
sns.kdeplot(test_pred, label='Kaggle Predictions', fill=True, alpha=0.3)
plt.title('Distribution Comparison: Train Actuals vs Kaggle Predictions')
plt.xlabel('Traffic Volume')
plt.legend()
plt.show()


# %% Cell 19
from feature_engineering.transformers import engineer_features

df = engineer_features(train_df)
print(f"Engineered frame shape: {df.shape}")


# %% Cell 21
# Core dependencies for LightGBM training
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

DROP_FEATURES = [
    "ID",
    "year",
    "hour_sin",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "holiday",
]

BASE_FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "time_index",
    "is_weekend",
    "is_morning_rush",
    "is_evening_rush",
    "is_daylight",
    "is_raining",
    "is_snowing",
    "is_severe_weather",
    "is_holiday",
    "rush_x_severe",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
]

DERIVED_FEATURE_COLS = [
    "hour_of_week",
    "is_peak_hour",
    "weekend_peak",
    "hour_cos",
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS

MODEL_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    n_estimators=5000,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    reg_lambda=0.1,
)

def make_model() -> LGBMRegressor:
    """Return a configured LightGBM regressor instance."""
    return LGBMRegressor(**MODEL_PARAMS)


# %% Cell 22
def prepare_feature_matrix(df_in: pd.DataFrame, expected_cols=None):
    """Create feature matrix (with optional column ordering) and keep engineered extras."""
    if "time_index" not in df_in.columns:
        raise ValueError("Expected column 'time_index' is missing.")

    df_feat = df_in.copy()
    df_feat["hour_of_week"] = df_feat["day_of_week"] * 24 + df_feat["hour"]
    df_feat["is_peak_hour"] = (
        df_feat["hour"].between(6, 9) | df_feat["hour"].between(15, 19)
    ).astype(int)
    df_feat["weekend_peak"] = df_feat["is_weekend"] * df_feat["is_peak_hour"]

    drop_cols = [col for col in DROP_FEATURES if col in df_feat.columns]
    if drop_cols:
        df_feat = df_feat.drop(columns=drop_cols)

    available_cols = [col for col in ALL_FEATURE_COLS if col in df_feat.columns]
    if expected_cols is not None:
        missing = [col for col in expected_cols if col not in available_cols]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        ordered_cols = expected_cols
    else:
        ordered_cols = available_cols

    X_mat = df_feat[ordered_cols].to_numpy()
    return X_mat, ordered_cols, df_feat


df_sorted = df.sort_values("time_index").reset_index(drop=True)
y_raw = df_sorted["traffic_volume"].to_numpy()
y = np.log1p(y_raw)

X, FEATURE_COLS, df_features = prepare_feature_matrix(df_sorted)
print(f"Feature matrix shape: {X.shape}")
print(f"Using {len(FEATURE_COLS)} predictors: {FEATURE_COLS}")


# %% Cell 23
tscv = TimeSeriesSplit(n_splits=5)
cv_rmses = []

for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = make_model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=200, verbose=False)],
    )

    y_valid_pred_log = model.predict(X_valid)
    y_valid_pred = np.expm1(y_valid_pred_log)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_valid), y_valid_pred))
    cv_rmses.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:,.3f}")

print(f"Average RMSE: {np.mean(cv_rmses):,.3f} ± {np.std(cv_rmses):,.3f}")


# %% Cell 24
final_model = make_model()
final_model.fit(X, y)

train_pred_log = final_model.predict(X)
res_log = y - train_pred_log
res_raw = np.expm1(y) - np.expm1(train_pred_log)

print(f"Training RMSE (original scale): {np.sqrt(mean_squared_error(np.expm1(y), np.expm1(train_pred_log))):,.3f}")
print(f"Residual log mean: {res_log.mean():.4f}, std: {res_log.std():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=train_pred_log, y=res_log, alpha=0.3, s=12, ax=ax)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_title("Residuals vs Predicted (log space)")
ax.set_xlabel("Predicted log1p traffic volume")
ax.set_ylabel("Residual (log space)")
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(res_log, bins=60, kde=True, color="teal")
plt.title("Residual Distribution (log space)")
plt.xlabel("Residual (log space)")
plt.show()


# %% Cell 25
def predict_traffic(df_new: pd.DataFrame, model: LGBMRegressor, feature_cols=FEATURE_COLS) -> np.ndarray:
    """Predict hourly traffic volume for a new, engineered dataframe."""
    df_clean = df_new.sort_values("time_index").reset_index(drop=True)
    X_new, _, _ = prepare_feature_matrix(df_clean, expected_cols=feature_cols)
    preds_log = model.predict(X_new)
    preds = np.expm1(preds_log)
    return np.clip(preds, a_min=0, a_max=None)


# Example usage (commented out):
# future_preds = predict_traffic(df_future, final_model)
# print(future_preds[:5])


# %% Cell 26
from pathlib import Path

kaggle_eng = engineer_features(kaggle_df)
submission_preds = predict_traffic(kaggle_eng, final_model)

submission_df = pd.DataFrame({
    "ID": kaggle_eng["ID"],
    "traffic_volume": submission_preds
})

submission_path = Path("lightgbm_submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Saved LightGBM submission to {submission_path.resolve()}")
submission_df.head()


# %% Cell 27
from pathlib import Path

kaggle_eng_v2 = engineer_features(kaggle_df)
submission_preds_v2 = predict_traffic(kaggle_eng_v2, final_model)

submission_df_v2 = pd.DataFrame({
    "ID": kaggle_eng_v2["ID"],
    "traffic_volume": submission_preds_v2
})

submission_path_v2 = Path("lightgbm_submission_v2.csv")
submission_df_v2.to_csv(submission_path_v2, index=False)

print(f"Saved LightGBM submission v2 to {submission_path_v2.resolve()}")
submission_df_v2.head()


# %% Cell 30
def ensure_lgbm_model_alignment():
    """Ensure the LightGBM model was trained with the current feature set."""
    global X, y, df_sorted, final_model, lgbm_model, FEATURE_COLS

    if 'FEATURE_COLS' not in globals() or 'X' not in globals():
        df_sorted = df.sort_values("time_index").reset_index(drop=True)
        y_raw = df_sorted["traffic_volume"].to_numpy()
        y = np.log1p(y_raw)
        X, FEATURE_COLS, _ = prepare_feature_matrix(df_sorted)

    if 'y' not in globals():
        y = np.log1p(df_sorted["traffic_volume"].to_numpy())

    if 'final_model' not in globals():
        final_model = make_model()
        final_model.fit(X, y)

    if 'lgbm_model' not in globals():
        lgbm_model = final_model

    expected_dim = len(FEATURE_COLS)
    trained_dim = getattr(lgbm_model, "n_features_in_", expected_dim)

    if trained_dim != expected_dim:
        print(
            f"Detected feature mismatch: model expects {trained_dim} features but pipeline has {expected_dim}. Retraining..."
        )
        lgbm_model = make_model()
        lgbm_model.fit(X, y)
        final_model = lgbm_model

    return lgbm_model


# %% Cell 31
if 'df_val' not in globals():
    split_idx = int(len(df_sorted) * 0.8)
    df_train = df_sorted.iloc[:split_idx].copy()
    df_val = df_sorted.iloc[split_idx:].copy()
else:
    if 'df_train' not in globals():
        df_train = df_sorted.copy()

y_train = df_train["traffic_volume"].to_numpy()

lgbm_model = ensure_lgbm_model_alignment()


# %% Cell 32
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

X_val, _, df_val_features = prepare_feature_matrix(df_val, expected_cols=FEATURE_COLS)
y_true = df_val["traffic_volume"].to_numpy()
y_true_log = np.log1p(y_true)

y_pred_log = lgbm_model.predict(X_val)
y_pred = np.expm1(y_pred_log)

residual = y_true - y_pred
residual_log = y_true_log - y_pred_log

cols_to_copy = ["traffic_volume"] + FEATURE_COLS
df_eval = df_val_features[cols_to_copy].copy()
df_eval["y_true"] = y_true
df_eval["y_pred"] = y_pred
df_eval["y_true_log"] = y_true_log
df_eval["y_pred_log"] = y_pred_log
df_eval["residual"] = residual
df_eval["residual_log"] = residual_log

print(df_eval[["y_true", "y_pred", "residual", "residual_log"]].describe().loc[["mean", "std", "min", "max"]])


# %% Cell 33
fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(y_train, fill=True, alpha=0.25, label="Train", ax=ax)
sns.kdeplot(df_eval["y_pred"], fill=True, alpha=0.25, label="LGBM pred", ax=ax)
ax.set_title("Traffic Volume Distribution: Train vs LightGBM Predictions")
ax.set_xlabel("Traffic volume")
ax.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df_eval["residual"], bins=60, color="steelblue", alpha=0.8)
plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.title("LightGBM residuals (actual - predicted)")
plt.xlabel("Residual (original scale)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(
    x=df_eval["y_pred_log"],
    y=df_eval["residual_log"],
    alpha=0.3,
    s=15,
)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Residuals vs Predicted (log space)")
plt.xlabel("Predicted log1p(traffic)")
plt.ylabel("Residual (log1p(actual) - log1p(pred))")
plt.show()


# %% Cell 35
if 'df_eval' not in globals():
    raise RuntimeError("df_eval not found. Run the LightGBM validation cell first.")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.boxplot(data=df_eval, x='hour', y='residual', ax=axes[0, 0], palette='Blues')
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('Residuals vs Hour of Day')
axes[0, 0].set_xlabel('Hour of day')
axes[0, 0].set_ylabel('Residual (actual - predicted)')

sns.boxplot(data=df_eval, x='day_of_week', y='residual', ax=axes[0, 1], palette='Oranges')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0, 1].set_title('Residuals vs Day of Week')
axes[0, 1].set_xlabel('Day of week (0=Mon)')
axes[0, 1].set_ylabel('Residual (actual - predicted)')

quantile_labels = [f'Q{i+1}' for i in range(5)]
df_eval['traffic_bin'] = pd.qcut(df_eval['y_true'], q=5, labels=quantile_labels, duplicates='drop')
sns.violinplot(data=df_eval, x='traffic_bin', y='residual', ax=axes[1, 0], palette='Purples')
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 0].set_title('Residuals by Traffic Volume Quantile')
axes[1, 0].set_xlabel('Traffic volume quantile (actual)')
axes[1, 0].set_ylabel('Residual (actual - predicted)')

sns.kdeplot(y_train, fill=True, alpha=0.3, label='Train', ax=axes[1, 1], color='steelblue')
sns.kdeplot(df_eval['y_pred'], fill=True, alpha=0.3, label='Test predictions', ax=axes[1, 1], color='darkorange')
axes[1, 1].set_title('Train vs Test Prediction Distribution')
axes[1, 1].set_xlabel('Traffic volume')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Clean up temporary column to avoid leaking into future analyses
df_eval.drop(columns=['traffic_bin'], inplace=True)


# %% Cell 37
if 'df_eval' not in globals():
    raise RuntimeError("df_eval not found. Run the validation diagnostics cell before plotting residuals.")

if 'hour_of_week' not in df_eval.columns:
    raise RuntimeError("'hour_of_week' column missing. Re-run feature engineering / validation prep.")

plot_sample = df_eval.sample(n=min(5000, len(df_eval)), random_state=42)
mean_residual = df_eval['residual'].mean()

fig, ax = plt.subplots(figsize=(18, 6))

sns.boxplot(
    data=df_eval,
    x='hour_of_week',
    y='residual',
    color='lightsteelblue',
    showfliers=False,
    ax=ax,
)

sns.scatterplot(
    data=plot_sample,
    x='hour_of_week',
    y='residual',
    color='navy',
    edgecolor=None,
    alpha=0.25,
    s=12,
    ax=ax,
)

ax.axhline(mean_residual, color='red', linestyle='--', linewidth=1.5, label=f"Mean residual ({mean_residual:.2f})")
ax.set_title('Residuals vs Hour of Week (scatter + box)')
ax.set_xlabel('Hour of week (0=Mon 00:00)')
ax.set_ylabel('Residual (actual - predicted)')
ax.set_xticks(np.arange(0, 168, 12))
ax.set_xticklabels([str(x) for x in np.arange(0, 168, 12)])
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()


# %% Cell 39
import shap

if 'lgbm_model' not in globals():
    lgbm_model = final_model

if 'X' not in globals() or 'X_val' not in globals():
    raise RuntimeError("Feature matrices X and X_val are missing. Re-run the LightGBM training/validation cells before computing SHAP values.")

X_train_df = pd.DataFrame(X, columns=FEATURE_COLS)
X_val_df = pd.DataFrame(X_val, columns=FEATURE_COLS)

background_size = min(2000, len(X_train_df))
background = shap.utils.sample(X_train_df, nsamples=background_size, random_state=42)

tree_explainer = shap.TreeExplainer(
    lgbm_model,
    data=background,
    feature_perturbation="interventional",
)

shap_values = tree_explainer.shap_values(X_val_df, check_additivity=False)
expected_value = tree_explainer.expected_value

if isinstance(shap_values, list):
    raise RuntimeError("Received multi-output SHAP values; expected regression with a single output.")

print(f"Computed SHAP values for {shap_values.shape[0]} validation rows and {shap_values.shape[1]} features.")


# %% Cell 40
if 'shap_values' not in globals():
    raise RuntimeError("Run the SHAP computation cell before plotting feature importance.")

importance_df = (
    pd.DataFrame({
        'feature': FEATURE_COLS,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    })
    .sort_values('mean_abs_shap', ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=importance_df,
    x='mean_abs_shap',
    y='feature',
    palette='viridis'
)
ax.set_title('Global SHAP Importance (mean |SHAP|)')
ax.set_xlabel('Mean |SHAP value|')
ax.set_ylabel('Feature')
plt.show()

shap.summary_plot(shap_values, X_val_df, plot_type='dot', max_display=20)


# %% Cell 41
dependence_features = ["hour_cos", "is_daylight", "hour", "is_morning_rush", "temp"]
missing_dependence = [feat for feat in dependence_features if feat not in FEATURE_COLS]
if missing_dependence:
    raise ValueError(f"Missing required dependence plot features: {missing_dependence}. Ensure feature engineering is up to date and rerun training.")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(dependence_features):
    shap.dependence_plot(
        feature,
        shap_values,
        X_val_df,
        ax=axes[idx],
        show=False,
    )
    axes[idx].set_title(f"SHAP Dependence: {feature}")

axes[-1].axis('off')
plt.tight_layout()
plt.show()


# %% Cell 42
interaction_feature = "hour"
interaction_with = "is_daylight"

missing_cols = [feat for feat in [interaction_feature, interaction_with] if feat not in FEATURE_COLS]
if missing_cols:
    raise ValueError(f"Missing required columns for interaction plot: {missing_cols}")

fig, ax = plt.subplots(figsize=(8, 6))
shap.dependence_plot(
    interaction_feature,
    shap_values,
    X_val_df,
    interaction_index=interaction_with,
    ax=ax,
    show=False,
)
ax.set_title("SHAP Interaction: hour × is_daylight")
plt.tight_layout()
plt.show()

