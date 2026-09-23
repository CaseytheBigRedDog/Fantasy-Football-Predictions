"""
check_usage_bias.py

STEP 1: how widespread is the "model runs low for high-usage players" problem?

Takes the main model's 2025 test-season predictions and groups them by how much
a player was being used going in (share of team targets, snap share, recent
scoring). For each group it shows the average miss = actual minus predicted.
This model predicts AVERAGES, so a well-calibrated group should be near 0.

Run:  python check_usage_bias.py
"""
import numpy as np
import pandas as pd

preds = pd.read_parquet("test_predictions.parquet")   # from 03_train_model.py
feat = pd.read_parquet(
    "model_data.parquet",
    columns=["season", "week", "player_display_name", "position",
             "fantasy_points_ppr_r5", "target_share_r5", "offense_pct_r5"],
)
key = ["season", "week", "player_display_name", "position"]
df = preds.merge(feat.drop_duplicates(key), on=key, how="inner")
df["miss"] = df["target_fp"] - df["pred_xgb"]

print(f"{len(df):,} player-weeks (2025 test season). Overall average miss: {df['miss'].mean():+.2f} points\n")


def table(title, groups, note=""):
    """groups: list of (label, boolean mask). Prints N, predicted, actual, miss +/- margin."""
    print(title)
    print(f"  {'Group':<26}{'N':>6}{'Predicted':>11}{'Actual':>8}{'Miss':>8}   ")
    flagged = []
    for label, mask in groups:
        s = df[mask]
        if len(s) < 30:
            continue
        miss = s["miss"].mean()
        m = 1.96 * s["miss"].std(ddof=1) / np.sqrt(len(s))
        flag = ""
        if abs(miss) > m and abs(miss) >= 0.5:
            flag = "<-- runs LOW" if miss > 0 else "<-- runs HIGH"
            flagged.append(label)
        print(f"  {label:<26}{len(s):>6,}{s['pred_xgb'].mean():>11.1f}{s['target_fp'].mean():>8.1f}"
              f"{miss:>+8.1f} (+/-{m:.1f}) {flag}")
    if note:
        print(f"  {note}")
    print()
    return flagged


catchers = df["position"].isin(["WR", "TE", "RB"])
flags = []

ts_bins = [(0, .10), (.10, .15), (.15, .20), (.20, .25), (.25, .30), (.30, 1.01)]
flags += table("By share of team targets (WR / TE / RB)", [
    (f"{lo:.0%}-{min(hi, 1):.0%}" if hi <= 1 else f"{lo:.0%}+",
     catchers & (df["target_share_r5"] >= lo) & (df["target_share_r5"] < hi))
    for lo, hi in ts_bins])

sn_bins = [(0, .4), (.4, .6), (.6, .8), (.8, 1.01)]
flags += table("By snap share (WR / TE / RB)", [
    (f"{lo:.0%}-{min(hi, 1):.0%}", catchers & (df["offense_pct_r5"] >= lo) & (df["offense_pct_r5"] < hi))
    for lo, hi in sn_bins])

pt_bins = [(-99, 5), (5, 10), (10, 15), (15, 20), (20, 99)]
flags += table("By last-5-game scoring (all positions)", [
    (f"{lo} to {hi} pts" if lo > -99 and hi < 99 else (f"under {hi} pts" if lo <= -99 else f"{lo}+ pts"),
     (df["fantasy_points_ppr_r5"] >= lo) & (df["fantasy_points_ppr_r5"] < hi))
    for lo, hi in pt_bins])

flags += table("By position", [(p, df["position"] == p) for p in ["QB", "RB", "WR", "TE"]])

high = catchers & (df["target_share_r5"] >= 0.25) & (df["offense_pct_r5"] >= 0.60)
flags += table("The Parker Washington type: target share 25%+ and snap share 60%+", [("High-usage receivers", high)])

print("Summary")
if flags:
    print("  Groups where the model is off by more than chance (and at least half a point):")
    for f in flags:
        print(f"    - {f}")
    print("  If the flagged groups are mostly the top usage tiers, the problem is narrow: "
          "the model under-rates heavily used players.")
    print("  If almost every row is flagged, the problem is broad.")
else:
    print("  No group is off by more than chance. The model looks well calibrated by usage.")
print("\n'(+/-)' is the 95% margin of error; a miss smaller than it is noise.")
