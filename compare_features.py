"""
compare_features.py

STEP 2: does adding the newer features (recency-weighted averages, current-season
averages, role-change signals) actually help?

Trains the main XGBoost model twice on the SAME split (train through 2023,
tune on 2024, test on 2025): once with the current features, once with the
current features plus the new ones. Then compares:
  * overall accuracy (error per player-week, with a margin of error)
  * whether the model still runs low for heavily used players

Run (after 02_features.py has been rerun to create the new columns):
    python compare_features.py

Nothing in your real models changes until you run adopt_new_features.py.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from new_features import new_feature_cols

TRAIN_END, VAL_SEASON, TEST_SEASON = 2023, 2024, 2025

df = pd.read_parquet("model_data.parquet")
new_cols = new_feature_cols(df)
if not new_cols:
    raise SystemExit("model_data.parquet has no new features yet. Run: python 02_features.py")

position = df["position"].copy()
df = pd.get_dummies(df, columns=["position"], prefix="pos")

base_cols = [c for c in df.columns if c.endswith(("_r3", "_r5", "_seasontd"))]
base_cols += ["games_played_prior", "team_implied_total", "spread_line", "total_line",
              "is_home", "rest_days", "def_pts_allowed_r5"]
base_cols += [c for c in df.columns if c.startswith("pos_")]
base_cols = list(dict.fromkeys(c for c in base_cols if c in df.columns))
new_all = list(dict.fromkeys(base_cols + new_cols))

train = df["season"] <= TRAIN_END
val = df["season"] == VAL_SEASON
test = df["season"] == TEST_SEASON
y = df["target_fp"]

print(f"Base features: {len(base_cols)} | with new features: {len(new_all)} "
      f"(+{len(new_all) - len(base_cols)})")
print(f"Train {train.sum():,} | tune {val.sum():,} | test {test.sum():,} player-weeks\n")


def fit_predict(cols):
    X = df[cols].fillna(0)
    model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, early_stopping_rounds=30, eval_metric="mae",
    )
    model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
    return model.predict(X[test])


print("Training the current model ...")
pred_base = fit_predict(base_cols)
print("Training with the new features ...\n")
pred_new = fit_predict(new_all)

t = pd.DataFrame({
    "actual": y[test].values, "pos": position[test].values,
    "base": pred_base, "new": pred_new,
    "target_share_r5": df.loc[test, "target_share_r5"].values,
    "offense_pct_r5": df.loc[test, "offense_pct_r5"].values,
})
t["err_base"] = t["actual"] - t["base"]
t["err_new"] = t["actual"] - t["new"]

# ---------------- overall accuracy ----------------
mae_b, mae_n = t["err_base"].abs().mean(), t["err_new"].abs().mean()
diff = t["err_new"].abs() - t["err_base"].abs()          # negative = new is better
d_mean = diff.mean()
d_margin = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
print(f"{'':<28}{'Current':>10}{'With new':>10}")
print(f"{'Average error (MAE)':<28}{mae_b:>10.3f}{mae_n:>10.3f}")
print(f"{'RMSE':<28}{np.sqrt((t['err_base']**2).mean()):>10.3f}{np.sqrt((t['err_new']**2).mean()):>10.3f}")
print(f"{'Average miss (bias)':<28}{t['err_base'].mean():>+10.3f}{t['err_new'].mean():>+10.3f}")
print(f"\nChange in average error: {d_mean:+.3f} points (+/-{d_margin:.3f}); negative = new features better\n")

print("Average error by position")
print(f"  {'':<6}{'Current':>9}{'With new':>10}")
for p in ["QB", "RB", "WR", "TE"]:
    s = t[t["pos"] == p]
    print(f"  {p:<6}{s['err_base'].abs().mean():>9.3f}{s['err_new'].abs().mean():>10.3f}")

# ---------------- usage-tier bias ----------------
print("\nAverage miss (actual minus predicted; near 0 is best) by share of team targets")
print(f"  {'Tier':<10}{'N':>6}{'Current':>10}{'With new':>10}")
catch = t["pos"].isin(["WR", "TE", "RB"])
for lo, hi, label in [(0, .10, "0-10%"), (.10, .20, "10-20%"), (.20, .25, "20-25%"),
                      (.25, .30, "25-30%"), (.30, 1.01, "30%+")]:
    s = t[catch & (t["target_share_r5"] >= lo) & (t["target_share_r5"] < hi)]
    if len(s) >= 30:
        print(f"  {label:<10}{len(s):>6,}{s['err_base'].mean():>+10.2f}{s['err_new'].mean():>+10.2f}")

hu = t[catch & (t["target_share_r5"] >= 0.25) & (t["offense_pct_r5"] >= 0.60)]
hu_b, hu_n = hu["err_base"].mean(), hu["err_new"].mean()
if len(hu) >= 30:
    print(f"\nHigh-usage receivers (target share 25%+, snap share 60%+): {len(hu)} player-weeks")
    print(f"  average miss: current {hu_b:+.2f}  ->  with new features {hu_n:+.2f}")

# ---------------- verdict ----------------
print("\nVerdict")
better_overall = d_mean < -d_margin
worse_overall = d_mean > d_margin
hu_gain = (abs(hu_b) - abs(hu_n)) if len(hu) >= 30 else 0.0
if worse_overall:
    print("  The new features make overall accuracy WORSE. Do not adopt them.")
elif better_overall or hu_gain >= 0.5:
    if better_overall and hu_gain >= 0.5:
        msg = f"Overall accuracy is better and the high-usage miss shrinks by {hu_gain:.1f} points."
    elif better_overall:
        msg = "Overall accuracy is better."
    else:
        msg = f"The high-usage miss shrinks by {hu_gain:.1f} points without hurting overall accuracy."
    print(f"  {msg} Adoption looks worthwhile: run  python adopt_new_features.py")
else:
    print("  No clear difference either way. There's no strong reason to change the model; "
          "keep it as is.")
print("  (Caveat: this compares on the test season, so treat a small gain with caution; the "
      "live weekly scorecard is the real judge.)")
