"""
predict_week.py

Projects the next unplayed NFL week: for every likely player, a FLOOR, MEDIAN
and CEILING of PPR fantasy points (10th / 50th / 90th percentile).

Run after refresh_data.py (which runs 02_features.py and creates
upcoming_data.parquet):

    python predict_week.py

What it does:
  1. Backtest: trains on seasons before the last full season and scores the
     last full season, so you see how accurate THIS script's models are.
  2. Final models: retrains on every finished game and predicts the upcoming week.
  3. Saves predictions_<season>_week<week>.csv and prints the top players.

What it does NOT know: injuries, inactives, depth-chart news, or weather.
A player who got hurt last week still gets a projection based on his form.
Check injury reports before acting on any number.
"""
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

POSITIONS = ["QB", "RB", "WR", "TE"]
QUANTILES = [0.1, 0.5, 0.9]  # floor, median, ceiling

train = pd.read_parquet("model_data.parquet")
up = pd.read_parquet("upcoming_data.parquet")

if up.empty:
    print("No upcoming games found, so there is nothing to predict.")
    print("(Run refresh_data.py first, or the season may be over.)")
    sys.exit(0)

season, week = int(up["season"].iloc[0]), int(up["week"].iloc[0])

# Same features the position models in 03b_position_quantile_models.py use
feature_cols = [c for c in train.columns if c.endswith(("_r3", "_r5", "_seasontd", "_trend"))]
feature_cols += [
    "games_played_prior", "team_implied_total", "spread_line", "total_line",
    "is_home", "rest_days", "def_pts_allowed_r5",
]
feature_cols = list(dict.fromkeys(c for c in feature_cols if c in train.columns))


def fit_quantile_model(X, y, alpha):
    model = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=250, learning_rate=0.04, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42,
    )
    model.fit(X, y)
    return model


def predict_range(models, X):
    """Floor / median / ceiling, sorted so floor <= median <= ceiling."""
    raw = np.column_stack([m.predict(X) for m in models])
    return np.sort(raw, axis=1)


# ---------------------------------------------------------------
# 1. Backtest on the last full season
# ---------------------------------------------------------------
holdout = int(train["season"].max()) - 1
print(f"Backtest: train on seasons before {holdout}, score on {holdout}\n")
print(f"{'Pos':<4} {'MAE':>6} {'In 10-90 range':>16} {'Rows':>7}")
all_err = []
for pos in POSITIONS:
    tr = train[(train["position"] == pos) & (train["season"] < holdout)]
    te = train[(train["position"] == pos) & (train["season"] == holdout)]
    if te.empty:
        continue
    models = [fit_quantile_model(tr[feature_cols].fillna(0), tr["target_fp"], q) for q in QUANTILES]
    rng = predict_range(models, te[feature_cols].fillna(0))
    err = (te["target_fp"].values - rng[:, 1])
    inside = ((te["target_fp"].values >= rng[:, 0]) & (te["target_fp"].values <= rng[:, 2])).mean()
    all_err.extend(np.abs(err))
    print(f"{pos:<4} {np.abs(err).mean():>6.3f} {inside:>15.1%} {len(te):>7,}")
print(f"{'All':<4} {np.mean(all_err):>6.3f}\n")

# ---------------------------------------------------------------
# 2. Final models: train on everything finished, predict the upcoming week
# ---------------------------------------------------------------
print(f"Predicting {season} week {week} ...")
out_frames = []
for pos in POSITIONS:
    tr = train[train["position"] == pos]
    fut = up[up["position"] == pos]
    if fut.empty:
        continue
    models = [fit_quantile_model(tr[feature_cols].fillna(0), tr["target_fp"], q) for q in QUANTILES]
    rng = predict_range(models, fut[feature_cols].fillna(0))
    frame = fut[["player_display_name", "position", "recent_team", "opponent_team",
                 "is_home", "spread_line", "total_line", "games_played_prior"]].copy()
    frame["floor"], frame["median"], frame["ceiling"] = rng[:, 0], rng[:, 1], rng[:, 2]
    out_frames.append(frame)

preds = pd.concat(out_frames, ignore_index=True)
preds = preds.rename(columns={
    "player_display_name": "player", "recent_team": "team", "opponent_team": "opponent",
})
preds[["floor", "median", "ceiling"]] = preds[["floor", "median", "ceiling"]].round(1)
preds = preds.sort_values("median", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------
# 3. Save and show
# ---------------------------------------------------------------
path = f"predictions_{season}_week{week}.csv"
preds.to_csv(path, index=False)

missing_lines = int(preds["spread_line"].isna().sum())
if missing_lines:
    print(f"\nWARNING: {missing_lines} players have no Vegas line yet, so their "
          f"projections are less reliable. Re-run closer to game day.")

few_games = int((preds["games_played_prior"] < 3).sum())
if few_games:
    print(f"Note: {few_games} players have fewer than 3 games of history (rookies "
          f"or new to the data); treat their ranges as rough.")

for pos in POSITIONS:
    top = preds[preds["position"] == pos].head(10)
    if top.empty:
        continue
    print(f"\n--- Top {pos} ---")
    for _, r in top.iterrows():
        where = "vs" if r["is_home"] == 1 else "@"
        print(f"{r['player']:<24} {r['team']:>3} {where} {r['opponent']:<3}  "
              f"{r['floor']:>5.1f} / {r['median']:>5.1f} / {r['ceiling']:>5.1f}")

print(f"\nSaved {path} ({len(preds):,} players). Columns: floor / median / ceiling.")
print("Reminder: injuries and inactives are NOT included -- check them before you decide.")
