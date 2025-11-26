# Feature Plot Catalog

Use this checklist when deciding which engineered variables to keep, transform, or drop. All plots can be generated via quick pandas/seaborn snippets; many already exist in `feature_analysis/generate_feature_plots.py` or `models/plot_ensemble_validation.py`.

## 1. Target + Distribution Diagnostics
- Traffic volume histogram + KDE (linear + log scale) for skew + zero inflation.
- Empirical CDF of traffic volume to gauge percentile breakpoints for capping or quantile features.
- Rolling mean/median chart on ordered `date_time` to inspect non-stationarity.

## 2. Time Structure
- Hourly boxplot + violin, split by weekday/weekend (captures rush-hour asymmetry).
- Day-of-week + month heatmap (hour on x, day on y) with mean traffic to spot two-way interactions.
- Trend plot of `time_index` vs target plus LOWESS smoother for macro drift.
- Autocorrelation (ACF/PACF) on aggregated hourly counts to justify lag features.

## 3. Weather & Holiday Effects
- Boxplots of traffic vs `weather_main` and vs `is_severe_weather` flag.
- Scatter/hex bins of `temp` vs traffic (colored by daylight) to confirm nonlinearity.
- Rain/Snow vs traffic faceted by rush-hour indicator to check interaction strength.
- Holiday vs non-holiday comparison (box/strip) plus per-holiday type bar chart.

## 4. Engineered Signals
- Cyclical encoding scatter (e.g., `hour_sin` vs `hour_cos` colored by traffic) to verify circular coverage.
- Binary flag impact: grouped bar chart of mean traffic for each of `is_weekend`, `is_daylight`, etc.
- Interaction evaluation: `rush_x_severe` vs traffic to ensure the feature captures additive penalty.

## 5. Multivariate + Correlation
- Pearson heatmap of raw numerics (temp/rain/snow/clouds/traffic).
- Spearman heatmap of cyclical encodings (detect redundant transforms).
- Binary-flag phi-coefficient heatmap (treating dummies as numeric) to find collinearity.
- Target-vs-feature correlation bars (Pearson + Spearman) sorted by magnitude to prioritize modeling.

## 6. Model-Oriented Diagnostics
- Residual vs predicted scatter (already in `models/plot_ensemble_*`).
- Residuals by hour/day/month boxplots to spot systematic bias.
- Prediction distribution overlap (train OOF vs Kaggle submissions) to detect covariate shift.
- Feature importance from tree-based models (gain/SHAP) to validate handcrafted transforms.

## 7. Retention/Drop Decisions
For each candidate feature, review:
1. Distribution shape (heavy tails? constant?)
2. Direct association with target (correlation plots)
3. Stability across time/weather segments (facet plots)
4. Contribution inside models (importance/SHAP)
5. Multicollinearity with existing features (heatmaps above)

Document keep/drop notes beside each plot to build a defensible feature selection narrative.
