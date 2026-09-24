"""
predict_week.py

Projects the next unplayed NFL week: for every likely player, a FLOOR, MEDIAN
and CEILING of PPR fantasy points (10th / 50th / 90th percentile), plus an
EXPECTED value (the average outcome). The median is the typical game; the
expected value is the same kind of number ESPN and most sites show, and it runs
higher than the median because a few huge games pull the average up. Compare
"expected" with other sites' projections, and the range with your risk appetite.

Run after refresh_data.py (which runs 02_features.py and creates
upcoming_data.parquet):

    python predict_week.py

What it does:
  1. Backtest: trains on seasons before the last full season and scores the
     last full season, so you see how accurate THIS script's models are.
  2. Final models: retrains on every finished game and predicts the upcoming week.
  3. Saves predictions_<season>_week<week>.csv and prints the top players.

How the floor / median / ceiling are built is set by RANGE_METHOD in
range_calibration.py ("quantile" = direct percentile models, "residual" = built
around the expected model, "hybrid" = residual ranges for the top projections and
direct models for the rest). The residual and hybrid methods take a few extra
minutes because they re-create past seasons' out-of-sample projections.

Injuries and depth charts (from download_injuries.py):
  * Once a team publishes an OFFICIAL designation (Out / Doubtful / Questionable, usually
    Friday for Sunday games), the player's projection is adjusted by the historical chance
    that players with that designation actually play. Out and Doubtful players drop to ~0.
  * Practice-only flags (no designation yet) are shown as "Practice: DNP / limited" but do
    NOT change any numbers, because early-week practice status is not reliable.
  * Re-running later in the week is safe: a team whose game has already kicked off keeps
    its earlier projections, so nothing is changed after a game starts.
  * Late scratches, weather, and coaching decisions are still unknown.
"""
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import availability as av
import kickoff
import range_calibration as rc

POSITIONS = rc.POSITIONS
QUANTILES = list(rc.QUANTILES)   # floor, median, ceiling
method = rc.RANGE_METHOD

train = pd.read_parquet("model_data.parquet")
up = pd.read_parquet("upcoming_data.parquet")

if up.empty:
    print("No upcoming games found, so there is nothing to predict.")
    print("(Run refresh_data.py first, or the season may be over.)")
    sys.exit(0)

season, week = int(up["season"].iloc[0]), int(up["week"].iloc[0])

# Same features the position models in 03b_position_quantile_models.py use
feature_cols = rc.get_feature_cols(train)


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
describe = {"quantile": "direct percentile models",
            "residual": "built around the expected model",
            "hybrid": "direct percentile models, residual ranges for the top projections"}
print(f"Range method: {method} ({describe[method]})")
print(f"Backtest: train on seasons before {holdout}, score on {holdout}\n")

# Out-of-sample 'expected' values. The residual method needs several past seasons to learn
# typical spreads; the quantile method only needs the backtest season itself.
first = rc.FIRST_CALIBRATION_SEASON if method in ("residual", "hybrid") else holdout
pool_all = rc.oos_expected(train, feature_cols, first, holdout)
pool_cal = pool_all[pool_all["season"] < holdout]
hold = pool_all[pool_all["season"] == holdout]

print(f"{'Pos':<4} {'MAE(median)':>12} {'In 10-90 range':>15} {'MAE(expected)':>14} {'Bias(exp)':>10} {'Rows':>7}")
all_err, all_exp_err, bt_rows = [], [], []
for pos in POSITIONS:
    te = train[(train["position"] == pos) & (train["season"] == holdout)]
    h = hold[hold["pos"] == pos]
    if te.empty or len(h) != len(te):
        continue
    exp_pred = h["expected"].to_numpy()
    if method == "residual":
        rng = rc.residual_range(exp_pred, pos, pool_cal)
    else:
        tr = train[(train["position"] == pos) & (train["season"] < holdout)]
        models = [fit_quantile_model(tr[feature_cols].fillna(0), tr["target_fp"], q) for q in QUANTILES]
        rng = np.clip(predict_range(models, te[feature_cols].fillna(0)), 0, None)
        if method == "hybrid":
            rng = rc.hybrid_range(exp_pred, pos, pool_cal, rng)
    actual = te["target_fp"].to_numpy()
    err = actual - rng[:, 1]
    inside = ((actual >= rng[:, 0]) & (actual <= rng[:, 2])).mean()
    all_err.extend(np.abs(err))
    exp_err = actual - exp_pred
    all_exp_err.extend(exp_err)
    bt_rows.append(pd.DataFrame({
        "pos": pos, "actual": actual, "expected": exp_pred,
        "floor": rng[:, 0], "median": rng[:, 1], "ceiling": rng[:, 2]}))
    print(f"{pos:<4} {np.abs(err).mean():>12.3f} {inside:>15.1%} {np.abs(exp_err).mean():>14.3f} "
          f"{exp_err.mean():>+10.2f} {len(te):>7,}")
print(f"{'All':<4} {np.mean(all_err):>12.3f} {'':>15} {np.abs(all_exp_err).mean():>14.3f} "
      f"{np.mean(all_exp_err):>+10.2f}")
print("Bias(exp) = actual minus expected; near 0 means the expected column is well calibrated.\n")

# Calibration by size of projection: are the BIG projections (the stars) as trustworthy as the rest?
bt = pd.concat(bt_rows, ignore_index=True)
bt["pct"] = bt.groupby("pos")["expected"].rank(pct=True)
print(f"Calibration by size of projection ({holdout} backtest; players ranked within position)")
print(f"{'Tier':<16}{'N':>6}{'Expected':>10}{'Actual':>8}{'Bias':>8}{'Below floor':>13}{'Below median':>14}{'Above ceiling':>15}")
for lo, hi, label in [(0, .5, "bottom half"), (.5, .8, "50th-80th pct"), (.8, .9, "80th-90th pct"),
                      (.9, .97, "90th-97th pct"), (.97, 1.01, "top 3%")]:
    g = bt[(bt["pct"] > lo) & (bt["pct"] <= hi)]
    if len(g) < 30:
        continue
    b = (g["actual"] - g["expected"])
    m = 1.96 * b.std(ddof=1) / np.sqrt(len(g))
    print(f"{label:<16}{len(g):>6,}{g['expected'].mean():>10.1f}{g['actual'].mean():>8.1f}"
          f"{b.mean():>+8.1f}{(g['actual'] < g['floor']).mean():>13.0%}{(g['actual'] <= g['median']).mean():>14.0%}"
          f"{(g['actual'] > g['ceiling']).mean():>15.0%}   (+/-{m:.1f})")
print("Bias should be near 0 in every tier. 'Below floor' should be about 10%, 'Below median' "
      "about 50% and 'Above ceiling' about 10%.\n")

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
    mean_model = rc.fit_mean_model(tr[feature_cols].fillna(0), tr["target_fp"])
    expected = np.maximum(mean_model.predict(fut[feature_cols].fillna(0)), 0)
    if method == "residual":
        rng = rc.residual_range(expected, pos, pool_all)
    else:
        models = [fit_quantile_model(tr[feature_cols].fillna(0), tr["target_fp"], q) for q in QUANTILES]
        rng = np.clip(predict_range(models, fut[feature_cols].fillna(0)), 0, None)
        if method == "hybrid":
            rng = rc.hybrid_range(expected, pos, pool_all, rng)
    frame = fut[["player_id", "player_display_name", "position", "recent_team", "opponent_team",
                 "is_home", "spread_line", "total_line", "games_played_prior"]].copy()
    frame["floor"], frame["median"], frame["ceiling"] = rng[:, 0], rng[:, 1], rng[:, 2]
    frame["expected"] = expected
    out_frames.append(frame)

preds = pd.concat(out_frames, ignore_index=True)
preds = preds.rename(columns={
    "player_display_name": "player", "recent_team": "team", "opponent_team": "opponent",
})

# ---------------------------------------------------------------
# 2b. Injuries and depth chart
# ---------------------------------------------------------------
preds["expected_if_active"] = preds["expected"]
preds["p_play"] = 1.0
preds["status"] = ""
preds["depth"] = ""

inj = av.load_injuries()
if inj.empty or not os.path.exists("stats_clean.parquet"):
    print("Injuries: no injury data found (run download_injuries.py), so projections ignore injuries.")
else:
    stats_hist = pd.read_parquet("stats_clean.parquet")[["player_id", "season", "week"]]
    rates = av.fit_play_rates(inj, stats_hist, season, week)
    flags = av.week_flags(inj, season, week, rates)
    if flags.empty:
        print(f"Injuries: no injury reports for {season} week {week} yet (they usually appear Wednesday "
              f"to Friday), so projections ignore injuries.")
    else:
        f = preds[["player_id"]].merge(flags, left_on="player_id", right_on="gsis_id", how="left")
        official = f["official"].fillna(False).to_numpy(dtype=bool)
        preds["status"] = f["status"].fillna("").to_numpy()
        preds["p_play"] = np.where(official, f["p_play"].fillna(1.0).to_numpy(), 1.0)
        for i in np.where(preds["p_play"].to_numpy() < 1.0)[0]:
            p_i = float(preds.at[i, "p_play"])
            preds.loc[i, ["floor", "median", "ceiling"]] = rc.mixture_range(
                preds.at[i, "expected_if_active"], preds.at[i, "position"], pool_all, p_i)
            preds.at[i, "expected"] = p_i * preds.at[i, "expected_if_active"]
        rep_col = f["report"].fillna("none").to_numpy()
        n_out = int((preds["p_play"] <= 0.02).sum())
        n_q = int((rep_col == "questionable").sum())
        n_prov = int(((~official) & (preds["status"].to_numpy() != "")).sum())
        lost = float((preds["expected_if_active"] - preds["expected"]).sum())
        print(f"Injuries (reports for week {week}): {int(f['gsis_id'].notna().sum())} of {len(preds)} projected "
              f"players are on the report.")
        print(f"  Official designations applied: {n_out} out/doubtful, {n_q} questionable "
              f"({lost:,.0f} projected points removed in total).")
        print(f"  Practice-only flags shown but not applied (provisional): {n_prov}")
        tbl = rates["rp"][(rates["rp"]["report"] == "questionable") & (rates["rp"]["n"] >= av.MIN_N)]
        if len(tbl):
            print("  Chance a Questionable player plays, from past seasons: "
                  + ", ".join(f"{r.practice} practice {r.p:.0%}" for r in tbl.itertuples()))

# ---------------------------------------------------------------
# 2c. Freeze projections for games that have already kicked off
# ---------------------------------------------------------------
path = f"predictions_{season}_week{week}.csv"
if os.path.exists("games_clean.parquet"):
    frozen = kickoff.frozen_teams(pd.read_parquet("games_clean.parquet"), season, week)
    if frozen:
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=preds.columns)
        kept = old[old["team"].isin(frozen)]
        preds = preds[~preds["team"].isin(frozen)]
        if len(kept):
            preds = pd.concat([preds, kept], ignore_index=True)
            print(f"Freeze: kept {len(kept)} earlier projections for teams whose games have already "
                  f"kicked off ({', '.join(sorted(frozen))}); they are unchanged.")
        else:
            print(f"Freeze: games for {', '.join(sorted(frozen))} have already kicked off and no earlier "
                  f"projections are on file, so those teams are left out.")
        preds["status"] = preds["status"].fillna("")
        preds["depth"] = preds["depth"].fillna("")
        preds["p_play"] = preds["p_play"].fillna(1.0)
        preds["expected_if_active"] = preds["expected_if_active"].fillna(preds["expected"])

# Depth-chart labels for every row (including any kept from an earlier file)
depth = av.load_depth_labels()
if len(depth):
    preds["depth"] = (preds[["player_id"]].merge(depth, left_on="player_id", right_on="gsis_id", how="left")
                      ["depth"].fillna("").to_numpy())

num = ["floor", "median", "ceiling", "expected", "expected_if_active"]
preds[num] = preds[num].round(1)
preds["p_play"] = preds["p_play"].round(2)
preds = preds.sort_values("expected", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------
# 3. Save and show
# ---------------------------------------------------------------
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
              f"{r['floor']:>5.1f} / {r['median']:>5.1f} / {r['ceiling']:>5.1f}   expected {r['expected']:>5.1f}"
              + (f"   [{r['status']}]" if r["status"] else ""))

print(f"\nSaved {path} ({len(preds):,} players). Columns: floor / median / ceiling / expected.")
print("Sorted by 'expected' (the average outcome). Compare it with ESPN-style projections.")
print("Reminder: only OFFICIAL designations are applied. Late scratches and news after the last "
      "injury update are not included.")
