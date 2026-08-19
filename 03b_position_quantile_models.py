"""
Step 3 (v2): Position-specific models + quantile regression.

Two upgrades from the v1 single pooled model:
1. Separate model per position (QB/RB/WR/TE) -- their scoring profiles and
   feature importances are different enough that pooling them costs accuracy.
2. Quantile regression instead of a single point estimate -- for each
   position we train three models (10th/50th/90th percentile) so predictions
   come as a FLOOR / MEDIAN / CEILING range, which is what you actually want
   for a start/sit decision, not a single number pretending to be certain.
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("model_data.parquet")

feature_cols = [c for c in df.columns if c.endswith(("_r3", "_r5", "_seasontd", "_trend"))]
feature_cols += [
    "games_played_prior", "team_implied_total", "spread_line", "total_line",
    "is_home", "rest_days", "def_pts_allowed_r5",
]
feature_cols = list(dict.fromkeys(c for c in feature_cols if c in df.columns))

X_all = df[feature_cols].fillna(0)
y_all = df["target_fp"]

train_mask = df["season"] <= 2022
val_mask = df["season"] == 2023
test_mask = df["season"] == 2024

QUANTILES = [0.1, 0.5, 0.9]  # floor, median, ceiling

results_all = []
models = {}

for pos in ["QB", "RB", "WR", "TE"]:
    pos_mask = df["position"] == pos
    print(f"\n{'='*50}\n{pos}\n{'='*50}")

    X_train = X_all[pos_mask & train_mask]
    y_train = y_all[pos_mask & train_mask]
    X_val = X_all[pos_mask & val_mask]
    y_val = y_all[pos_mask & val_mask]
    X_test = X_all[pos_mask & test_mask]
    y_test = y_all[pos_mask & test_mask]

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    pos_preds = {}
    for q in QUANTILES:
        model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, early_stopping_rounds=30,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pos_preds[q] = model.predict(X_test)
        models[(pos, q)] = model

    # Point accuracy: use the median (q=0.5) model as the point estimate
    mae = mean_absolute_error(y_test, pos_preds[0.5])
    print(f"MAE (median model): {mae:.3f}")

    # Interval calibration check: what fraction of actual outcomes fell
    # inside the predicted [floor, ceiling] range? Should land near 80%
    # since we used the 10th/90th percentiles.
    within_interval = (
        (y_test.values >= pos_preds[0.1]) & (y_test.values <= pos_preds[0.9])
    ).mean()
    print(f"Actual results inside predicted 10-90 range: {within_interval:.1%} (target: ~80%)")

    avg_width = (pos_preds[0.9] - pos_preds[0.1]).mean()
    print(f"Average interval width: {avg_width:.1f} points")

    pos_results = df.loc[pos_mask & test_mask, ["season", "week", "player_display_name", "position", "target_fp"]].copy()
    pos_results["pred_floor"] = pos_preds[0.1]
    pos_results["pred_median"] = pos_preds[0.5]
    pos_results["pred_ceiling"] = pos_preds[0.9]
    results_all.append(pos_results)

all_results = pd.concat(results_all, ignore_index=True)
all_results.to_parquet("test_predictions_v2.parquet", index=False)

print(f"\n{'='*50}")
print("Overall MAE across all positions (median model):",
      round(mean_absolute_error(all_results["target_fp"], all_results["pred_median"]), 3))
print("Saved test_predictions_v2.parquet")
