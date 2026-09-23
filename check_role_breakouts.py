"""
check_role_breakouts.py

Follow-up to check_breakouts.py. Among pass-catchers (WR / TE / RB) whose recent
scoring is well above their career average, does it matter whether their
share of the team's targets ALSO jumped?

  * "Role jump":  target share over the last 5 games is at least 5 percentage
                  points above his career target share (a real change in role)
  * "No role jump": scoring is up, but his share of the offense hasn't changed
                  (more likely touchdowns, a big play or two, or a friendly matchup)

If role-based breakouts score MORE than the model predicts (compared with the
no-role-change group), the model is not giving real role changes enough credit.

Run:  python check_role_breakouts.py
"""
import numpy as np
import pandas as pd

ROLE_JUMP = 0.05   # 5 percentage points of team target share

ours = pd.read_parquet("test_predictions_v2.parquet")  # 2025 test set, from 03b
feat = pd.read_parquet(
    "model_data.parquet",
    columns=["season", "week", "player_display_name", "position",
             "fantasy_points_ppr_r5", "fantasy_points_ppr_seasontd", "games_played_prior",
             "target_share_r5", "target_share_seasontd"],
)
key = ["season", "week", "player_display_name", "position"]
df = ours.merge(feat.drop_duplicates(key), on=key, how="inner")

# Veterans who catch passes (quarterbacks don't have a target share)
df = df[(df["games_played_prior"] >= 16) & (df["fantasy_points_ppr_seasontd"] >= 4)
        & df["position"].isin(["WR", "TE", "RB"])]
df = df.dropna(subset=["target_share_r5", "target_share_seasontd"]).copy()

df["ratio"] = df["fantasy_points_ppr_r5"] / df["fantasy_points_ppr_seasontd"]
df["ts_change"] = df["target_share_r5"] - df["target_share_seasontd"]
df["role_jump"] = df["ts_change"] >= ROLE_JUMP
df["miss"] = df["target_fp"] - df["pred_median"]   # actual minus predicted

print(f"Pass-catching veterans analyzed: {len(df):,} player-weeks (2025 test season)")
print(f"Overall average miss: {df['miss'].mean():+.1f} points "
      f"(a small positive number is normal; see the last check)\n")


def margin(s):
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else float("nan")


def show(label, s):
    print(f"{label:<34}{len(s):>5,}{s['fantasy_points_ppr_r5'].mean():>8.1f}"
          f"{s['ts_change'].mean() * 100:>+10.1f}{s['pred_median'].mean():>11.1f}"
          f"{s['target_fp'].mean():>8.1f}{s['miss'].mean():>+7.1f} (+/-{margin(s['miss']):.1f})")


print(f"{'Group':<34}{'N':>5}{'Last 5':>8}{'TgtShare':>10}{'Predicted':>11}{'Actual':>8}{'Miss':>7}")
print(f"{'':<34}{'':>5}{'pts':>8}{'chg (pts)':>10}")

for title, mask in [("Breakouts (last 5 >= 1.5x career scoring)", df["ratio"] >= 1.5),
                    ("Hot streaks (last 5 >= 1.2x career scoring)", df["ratio"] >= 1.2)]:
    sub = df[mask]
    role, norole = sub[sub["role_jump"]], sub[~sub["role_jump"]]
    print(f"\n{title}")
    if len(role) < 30 or len(norole) < 30:
        print(f"  Too few players to split (role jump: {len(role)}, no role jump: {len(norole)}).")
        continue
    show("  Role jump (target share up)", role)
    show("  No role jump", norole)
    diff = role["miss"].mean() - norole["miss"].mean()
    err = np.sqrt(margin(role["miss"]) ** 2 + margin(norole["miss"]) ** 2)
    if diff > err and diff > 0.5:
        verdict = (f"Role-based players beat the model by {diff:+.1f} more points than the "
                   f"others: the model under-credits real role changes.")
    elif diff < -err and diff < -0.5:
        verdict = (f"Role-based players did {diff:+.1f} points WORSE than the others relative to "
                   f"the model: it already over-credits them.")
    else:
        verdict = (f"Difference {diff:+.1f} points is within the margin of error: no detectable "
                   f"gap, so the model already handles role changes about right.")
    print(f"  -> {verdict}")

print("\n'Miss' is actual minus predicted; (+/-) is the 95% margin of error, so differences "
      "smaller than that are noise.")
print("Predicted vs. Actual also shows whether the model already gives role-jump players a higher "
      "projection than the rest.")
