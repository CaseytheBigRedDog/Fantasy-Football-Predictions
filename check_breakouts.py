"""
check_breakouts.py

Does the model react properly when a player's recent form jumps above (or
falls below) his career average?

For veterans in the 2025 test season, groups players by how their last-5-game
scoring compares to their career average, then compares the model's average
prediction with what actually happened. If the model is too slow to react to
role changes, the "breakout" group will score MORE than predicted.

Run:  python check_breakouts.py
"""
import pandas as pd

ours = pd.read_parquet("test_predictions_v2.parquet")  # 2025 test set, from 03b
feat = pd.read_parquet(
    "model_data.parquet",
    columns=["season", "week", "player_display_name", "position",
             "fantasy_points_ppr_r5", "fantasy_points_ppr_seasontd", "games_played_prior"],
)
key = ["season", "week", "player_display_name", "position"]
df = ours.merge(feat.drop_duplicates(key), on=key, how="inner")

# Veterans with a real history, so "career average" means something
df = df[(df["games_played_prior"] >= 16) & (df["fantasy_points_ppr_seasontd"] >= 4)].copy()
df["ratio"] = df["fantasy_points_ppr_r5"] / df["fantasy_points_ppr_seasontd"]

bins = [0, 0.8, 1.2, 1.5, 100]
labels = ["Cooling (last 5 below 0.8x career)", "Steady (0.8x-1.2x)",
          "Rising (1.2x-1.5x)", "Breakout (1.5x+ career)"]
df["group"] = pd.cut(df["ratio"], bins=bins, labels=labels, right=False)

print(f"Veteran player-weeks analyzed: {len(df):,} (2025 test season)\n")
print(f"{'Group':<36}{'N':>6}{'Career':>8}{'Last 5':>8}{'Predicted':>11}{'Actual':>8}{'Bias':>7}")
rows = {}
for g in labels:
    s = df[df["group"] == g]
    if s.empty:
        continue
    bias = (s["target_fp"] - s["pred_median"]).mean()
    rows[g] = (len(s), bias)
    print(f"{g:<36}{len(s):>6,}{s['fantasy_points_ppr_seasontd'].mean():>8.1f}"
          f"{s['fantasy_points_ppr_r5'].mean():>8.1f}{s['pred_median'].mean():>11.1f}"
          f"{s['target_fp'].mean():>8.1f}{bias:>+7.1f}")

overall = (df["target_fp"] - df["pred_median"]).mean()
print(f"\nAll veterans: average bias {overall:+.1f} points (actual minus predicted).")
print("Bias = how many points per game the model is off on average. "
      "Near 0 means well calibrated; positive means the model predicts too low.\n")

b = rows.get("Breakout (1.5x+ career)")
if b is None:
    print("No breakout players in this sample.")
else:
    n, bias = b
    rel = bias - overall
    if n < 100:
        print(f"Only {n} breakout player-weeks, so treat this as a rough hint.")
    if rel > 1.0:
        print(f"Breakout players scored {bias:+.1f} points MORE than predicted "
              f"({rel:+.1f} vs. veterans overall): the model is too slow to react to role changes.")
    elif rel < -1.0:
        print(f"Breakout players scored {bias:+.1f} points vs. prediction "
              f"({rel:+.1f} vs. veterans overall): the model over-reacts to hot streaks.")
    else:
        print(f"Breakout players' bias ({bias:+.1f}) is close to everyone else's "
              f"({overall:+.1f}): the model's caution about hot streaks is justified.")
