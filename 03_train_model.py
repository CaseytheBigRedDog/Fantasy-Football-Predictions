"""
Step 3: Train baseline models with time-series-correct validation.

Validation strategy: train on 2010-2022, validate on 2023, test on 2024.
This mimics real deployment -- you'll always be predicting a future week
you haven't seen, never a randomly held-out week from the middle of a
season you partially trained on.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_parquet("model_data.parquet")

# ---------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------
feature_cols = [c for c in df.columns if c.endswith(("_r3", "_r5", "_seasontd"))]
feature_cols += [
    "games_played_prior", "team_implied_total", "spread_line", "total_line",
    "is_home", "rest_days", "def_pts_allowed_r5",
]

# One-hot encode position (QB/RB/WR/TE scoring profiles are very different)
df = pd.get_dummies(df, columns=["position"], prefix="pos")
feature_cols += [c for c in df.columns if c.startswith("pos_")]

feature_cols = list(dict.fromkeys(c for c in feature_cols if c in df.columns))  # dedupe, preserve order
print(f"Using {len(feature_cols)} features")

X = df[feature_cols].fillna(0)
y = df["target_fp"]

train_mask = df["season"] <= 2022
val_mask = df["season"] == 2023
test_mask = df["season"] == 2024

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train: {len(X_train):,} rows (2010-2022)")
print(f"Val:   {len(X_val):,} rows (2023)")
print(f"Test:  {len(X_test):,} rows (2024)")

# ---------------------------------------------------------------
# Baseline: "predict last-3-week average" -- the thing your model has to beat
# ---------------------------------------------------------------
naive_pred = df.loc[test_mask, "fantasy_points_ppr_r3"].fillna(df.loc[train_mask, "target_fp"].mean())
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
print(f"\n--- Naive baseline (3-week rolling avg) on 2024 test set ---")
print(f"MAE:  {naive_mae:.3f}")
print(f"RMSE: {naive_rmse:.3f}")

# ---------------------------------------------------------------
# Model 1: Ridge regression (linear baseline)
# ---------------------------------------------------------------
ridge = Ridge(alpha=10.0)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)
print(f"\n--- Ridge Regression on 2024 test set ---")
print(f"MAE:  {mean_absolute_error(y_test, pred_ridge):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_ridge)):.3f}")
print(f"R2:   {r2_score(y_test, pred_ridge):.3f}")

# ---------------------------------------------------------------
# Model 2: Gradient Boosting (sklearn)
# ---------------------------------------------------------------
gbr = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.8, random_state=42
)
gbr.fit(X_train, y_train)
pred_gbr = gbr.predict(X_test)
print(f"\n--- Gradient Boosting on 2024 test set ---")
print(f"MAE:  {mean_absolute_error(y_test, pred_gbr):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_gbr)):.3f}")
print(f"R2:   {r2_score(y_test, pred_gbr):.3f}")

# ---------------------------------------------------------------
# Model 3: XGBoost, tuned lightly using the 2023 validation set
# ---------------------------------------------------------------
xgb = XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, early_stopping_rounds=30, eval_metric="mae"
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
pred_xgb = xgb.predict(X_test)
print(f"\n--- XGBoost on 2024 test set ---")
print(f"MAE:  {mean_absolute_error(y_test, pred_xgb):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_xgb)):.3f}")
print(f"R2:   {r2_score(y_test, pred_xgb):.3f}")
print(f"Best iteration: {xgb.best_iteration}")

# ---------------------------------------------------------------
# Feature importance from XGBoost -- what's actually driving predictions
# ---------------------------------------------------------------
importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- Top 15 features (XGBoost) ---")
print(importances.head(15).to_string())

# Save the best model's predictions for error analysis
pos_cols = [c for c in df.columns if c.startswith("pos_")]
df["position"] = df[pos_cols].idxmax(axis=1).str.replace("pos_", "", regex=False)
results = df.loc[test_mask, ["season", "week", "player_display_name", "position", "target_fp"]].copy()
results["pred_xgb"] = pred_xgb
results["pred_naive"] = naive_pred.values
results["abs_error"] = (results["target_fp"] - results["pred_xgb"]).abs()
results.to_parquet("test_predictions.parquet", index=False)
print("\nSaved test_predictions.parquet for error analysis")
