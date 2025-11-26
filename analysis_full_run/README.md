# analysis_full_run package

This folder is a self-contained snapshot of everything needed to execute `analysis.ipynb` end-to-end.

## Contents
- `analysis.ipynb` – primary notebook.
- `TRAIN.csv`, `KAGGLE.csv` – raw datasets loaded by the first cell.
- `prior_best_submission/` – contains prior leaderboard submissions used during the blend/stack steps.
- `artifacts/` – legacy HistGradientBoosting outputs that the stacking cell uses as a fallback for old-model OOF predictions.

## Usage
1. Open this folder (`analysis_full_run`) in VS Code or Jupyter so the working directory matches the notebook location.
2. Run the cells top-to-bottom. The pipeline cell will regenerate `artifacts_hgb_engineered_10fold/` and submission files locally.
3. The stacking cell automatically discovers the prior submission from `prior_best_submission/` and uses the artifacts fallback if older OOF `.npy` files are unavailable.
4. Plotting cells save PNGs to a freshly created `PLOTS/` directory.

Push this folder to GitHub to share the full, runnable workflow without referencing any paths outside the repo.
