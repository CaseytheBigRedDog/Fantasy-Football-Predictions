"""
find_comps.py

Finds past players who went into a game with a similar profile to an upcoming
player (same position, similar recent scoring, share of team targets and snap
share) and shows what they ACTUALLY scored in that game.

This is an independent check on a projection: if players like this really
averaged 15 points, a projection of 10.6 is probably too low; if they averaged
11, it's probably fine.

Usage (after refresh_data.py and predict_week.py):
    python find_comps.py "Parker Washington"
"""
import os
import sys

import numpy as np
import pandas as pd

name = " ".join(sys.argv[1:]).strip()
if not name:
    print('Usage: python find_comps.py "Player Name"')
    sys.exit(1)

up = pd.read_parquet("upcoming_data.parquet")
me = up[up["player_display_name"].str.lower() == name.lower()]
if me.empty:
    words = [w.lower() for w in name.split() if len(w) >= 4]
    lowered = up["player_display_name"].str.lower()
    hit = np.zeros(len(up), dtype=bool)
    for w in words:
        hit |= lowered.str.contains(w[:4], regex=False).to_numpy()   # first 4 letters, so small typos still match
    close = up.loc[hit, "player_display_name"].tolist()
    print(f"No upcoming-week row found for '{name}'.")
    if close:
        print("Did you mean: " + ", ".join(close[:8]))
    sys.exit(1)
me = me.iloc[0]

cols = ["player_display_name", "position", "season", "week", "games_played_prior",
        "fantasy_points_ppr_r5", "fantasy_points_ppr_seasontd", "target_share_r5",
        "offense_pct_r5", "target_fp"]
hist = pd.read_parquet("model_data.parquet", columns=cols)

pos = me["position"]
r5, career = me["fantasy_points_ppr_r5"], me["fantasy_points_ppr_seasontd"]
ts, snap = me["target_share_r5"], me["offense_pct_r5"]
season, week = int(me["season"]), int(me["week"])

# ---- his profile and our projection ----
print(f"{me['player_display_name']} ({pos}), going into {season} week {week}")
print(f"  last-5 average {r5:.1f} pts | career average {career:.1f} | target share {ts:.0%} | "
      f"snap share {snap:.0%} | {int(me['games_played_prior'])} prior games")
pred_path = f"predictions_{season}_week{week}.csv"
if os.path.exists(pred_path):
    p = pd.read_csv(pred_path)
    row = p[p["player"].str.lower() == name.lower()]
    if len(row):
        r = row.iloc[0]
        print(f"  Our projection: {r['floor']:.1f} floor / {r['median']:.1f} median / {r['ceiling']:.1f} ceiling")

# ---- find comparable past player-weeks ----
pts_tol = max(3.0, 0.2 * r5)
ts_tol, snap_tol = 0.06, 0.12
mask = (
    (hist["position"] == pos)
    & (hist["games_played_prior"] >= 16)
    & hist["fantasy_points_ppr_r5"].between(r5 - pts_tol, r5 + pts_tol)
    & hist["offense_pct_r5"].between(snap - snap_tol, snap + snap_tol)
    & (hist["player_display_name"] != me["player_display_name"])
)
crit = [f"last-5 scoring {r5 - pts_tol:.1f}-{r5 + pts_tol:.1f}",
        f"snap share {max(snap - snap_tol, 0):.0%}-{min(snap + snap_tol, 1):.0%}"]
if pos != "QB":
    mask &= hist["target_share_r5"].between(ts - ts_tol, ts + ts_tol)
    crit.insert(1, f"target share {max(ts - ts_tol, 0):.0%}-{ts + ts_tol:.0%}")
comps = hist[mask]


def describe(label, d):
    print(f"\n{label}")
    if len(d) < 15:
        print(f"  Only {len(d)} matching player-weeks; too few to say much.")
        return
    q = d["target_fp"].quantile([0.1, 0.5, 0.9])
    print(f"  {len(d):,} player-weeks from {d['player_display_name'].nunique():,} different players")
    print(f"  Points scored in that game:  average {d['target_fp'].mean():.1f} | median {q[0.5]:.1f} | "
          f"10th percentile {q[0.1]:.1f} | 90th percentile {q[0.9]:.1f}")


print(f"\nPast {pos}s ({', '.join(crit)}), veterans with 16+ games:")
describe("All comparable players:", comps)
describe(f"Only those whose career average was well below their recent form (career avg <= "
         f"65% of last-5 avg, like his {career / r5:.0%}):",
         comps[comps["fantasy_points_ppr_seasontd"] <= 0.65 * r5])

# ---- how did the model do on such players last season? ----
if os.path.exists("test_predictions_v2.parquet"):
    t = pd.read_parquet("test_predictions_v2.parquet")[
        ["season", "week", "player_display_name", "position", "pred_median"]]
    j = comps.merge(t, on=["season", "week", "player_display_name", "position"], how="inner")
    print("\nHow our model did on players like this in the 2025 test season:")
    if len(j) < 15:
        print(f"  Only {len(j)} matching player-weeks; too few to say much.")
    else:
        miss = j["target_fp"] - j["pred_median"]
        m = 1.96 * miss.std(ddof=1) / np.sqrt(len(j))
        print(f"  {len(j)} player-weeks: predicted median {j['pred_median'].mean():.1f}, "
              f"actual average {j['target_fp'].mean():.1f}, miss {miss.mean():+.1f} (+/-{m:.1f})")
        print("  (A miss near +1 to +1.5 is normal because average > median; "
              "much bigger means the model runs low for this type.)")

print("\nNote: several weeks from the same player are related, so treat these as "
      "approximate. Compare the AVERAGE to ESPN-style projections (which are averages)\n"
      "and the MEDIAN to our median.")
