"""
score_predictions.py

Grades a weekly predictions file (made by predict_week.py) against what
actually happened, and keeps a running accuracy log.

Usage (after refresh_data.py has pulled the finished week's stats):

    python score_predictions.py            # scores the latest week with finished games
    python score_predictions.py 2026 3     # or name the season and week

Suggested weekly routine (Tuesday, once Monday night is final):
    1. python refresh_data.py        (new stats, retrain)
    2. python score_predictions.py   (grade LAST week's predictions)
    3. python predict_week.py        (predict THIS coming week)

Outputs:
    scored_<season>_week<week>.csv   one row per predicted player, with the actual result
    accuracy_log.csv                 one row per week and position, rebuilt each run
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# How far to move FantasyPros' week labels. 0 = use the labels as 04_fantasypros_comparison.py
# does. Run check_fp_alignment.py; if it says a shift is needed, set it here (e.g. -1).
FP_WEEK_OFFSET = -1
FP_MIN_PLAYERS = 15   # skip a position if fewer players than this match

stats = pd.read_parquet("stats_clean.parquet")
games = pd.read_parquet("games_clean.parquet")


def final_teams(season, week):
    """Teams whose game that week has a final score."""
    g = games[(games["season"] == season) & (games["week"] == week) & games["home_score"].notna()]
    return set(g["home_team"]) | set(g["away_team"])


# ---------------------------------------------------------------
# 1. Pick which predictions file to score
# ---------------------------------------------------------------
if len(sys.argv) == 3:
    season, week = int(sys.argv[1]), int(sys.argv[2])
else:
    found = []
    for fn in glob.glob("predictions_*_week*.csv"):
        m = re.search(r"predictions_(\d{4})_week(\d+)\.csv$", fn)
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    season = week = None
    for s, w in sorted(found, reverse=True):
        if final_teams(s, w):  # newest file that has at least one finished game
            season, week = s, w
            break
    if season is None:
        print("No predictions file has finished games yet. Run refresh_data.py "
              "after the games are final, then try again.")
        sys.exit(0)

path = f"predictions_{season}_week{week}.csv"
if not os.path.exists(path):
    print(f"Could not find {path}.")
    sys.exit(1)

preds = pd.read_csv(path)
print(f"Scoring {path} ({len(preds):,} players)\n")

# ---------------------------------------------------------------
# 2. Attach player ids, actual points, and the naive baseline
# ---------------------------------------------------------------
# Only games BEFORE the predicted week count as history.
prior = stats[(stats["season"] < season) | ((stats["season"] == season) & (stats["week"] < week))]
prior = prior.sort_values(["season", "week"])

if "player_id" not in preds.columns:
    # Older files have no id column: match on name + position + team, using
    # each player's most recent game before the predicted week.
    key = ["player_display_name", "position", "recent_team"]
    lookup = (prior.drop_duplicates(key, keep="last")[key + ["player_id"]]
              .rename(columns={"player_display_name": "player", "recent_team": "team"}))
    preds = preds.merge(lookup, on=["player", "position", "team"], how="left")

actual = (stats[(stats["season"] == season) & (stats["week"] == week)]
          [["player_id", "fantasy_points_ppr"]]
          .rename(columns={"fantasy_points_ppr": "actual"})
          .drop_duplicates("player_id"))
preds = preds.merge(actual, on="player_id", how="left")

# Naive baseline: the player's average over his last 3 games (what the model must beat)
naive = (prior.groupby("player_id").tail(3).groupby("player_id")["fantasy_points_ppr"]
         .mean().rename("naive"))
preds = preds.merge(naive, left_on="player_id", right_index=True, how="left")

final = final_teams(season, week)
preds["status"] = np.where(
    ~preds["team"].isin(final), "pending",
    np.where(preds["actual"].notna(), "scored", "no_stats"),
)
preds.loc[preds["player_id"].isna(), "status"] = "unmatched"

counts = preds["status"].value_counts().to_dict()
if counts.get("pending"):
    print(f"Skipped {counts['pending']} players whose game isn't final yet.")
if counts.get("unmatched"):
    print(f"Skipped {counts['unmatched']} players that couldn't be matched to stats.")

scored = preds[preds["status"] == "scored"].copy()
if scored.empty:
    print("No finished games to score yet.")
    sys.exit(0)


# ---------------------------------------------------------------
# 3. Accuracy
# ---------------------------------------------------------------
def summarize(df):
    err = df["actual"] - df["median"]
    return {
        "n": len(df),
        "model_mae": err.abs().mean(),
        "naive_mae": (df["actual"] - df["naive"]).abs().mean(),
        "in_range": ((df["actual"] >= df["floor"]) & (df["actual"] <= df["ceiling"])).mean(),
        "above": (df["actual"] > df["ceiling"]).mean(),
        "below": (df["actual"] < df["floor"]).mean(),
        "rank_corr": df["median"].corr(df["actual"], method="spearman"),
    }


rows = {pos: summarize(scored[scored["position"] == pos])
        for pos in ["QB", "RB", "WR", "TE"] if (scored["position"] == pos).any()}
rows["ALL"] = summarize(scored)

print(f"{'Pos':<4} {'N':>5} {'Model MAE':>10} {'Naive MAE':>10} {'In range':>9} "
      f"{'Above':>7} {'Below':>7} {'Rank corr':>10}")
for pos, r in rows.items():
    rho = "n/a" if pd.isna(r["rank_corr"]) else f"{r['rank_corr']:.3f}"
    print(f"{pos:<4} {r['n']:>5} {r['model_mae']:>10.2f} {r['naive_mae']:>10.2f} "
          f"{r['in_range']:>9.1%} {r['above']:>7.1%} {r['below']:>7.1%} {rho:>10}")

overall = rows["ALL"]
verdict = "beat" if overall["model_mae"] < overall["naive_mae"] else "did not beat"
print(f"\nOverall the model {verdict} the naive last-3-games average "
      f"({overall['model_mae']:.2f} vs {overall['naive_mae']:.2f} points off, on average).")
print("For a well-calibrated model, about 80% should land in range and about "
      "10% each above and below.")
print("One week is a small sample and will bounce around; the trend over "
      "many weeks is what matters.")

# ---------------------------------------------------------------
# 4. Players with no recorded stats (inactive, injured, or a zero-stat game)
# ---------------------------------------------------------------
nostat = preds[preds["status"] == "no_stats"]
if len(nostat):
    both = preds[preds["status"].isin(["scored", "no_stats"])]
    mae_zero = (both["actual"].fillna(0) - both["median"]).abs().mean()
    print(f"\n{len(nostat)} predicted players have no recorded stats in a finished game "
          f"(inactive, injured, or no stats).")
    print(f"If they count as 0 points, the overall MAE is {mae_zero:.2f} instead of "
          f"{overall['model_mae']:.2f}.")
    print("Biggest of these (the model has no injury information):")
    for _, r in nostat.sort_values("median", ascending=False).head(8).iterrows():
        print(f"  {r['player']:<24} {r['position']} {r['team']:>3}  predicted {r['median']:>5.1f}")

# ---------------------------------------------------------------
# 5. Biggest surprises
# ---------------------------------------------------------------
scored["over_ceiling"] = scored["actual"] - scored["ceiling"]
scored["under_floor"] = scored["floor"] - scored["actual"]

def show_misses(title, df):
    print(title)
    if df.empty:
        print("  (none)")
    for _, r in df.head(5).iterrows():
        print(f"  {r['player']:<24} {r['position']} {r['team']:>3}  "
              f"{r['floor']:>5.1f} / {r['median']:>5.1f} / {r['ceiling']:>5.1f}  ->  {r['actual']:>5.1f}")


show_misses("\nBiggest breakouts (beat the ceiling by the most):",
            scored[scored["over_ceiling"] > 0].sort_values("over_ceiling", ascending=False))
show_misses("\nBiggest busts (fell below the floor by the most):",
            scored[scored["under_floor"] > 0].sort_values("under_floor", ascending=False))

# ---------------------------------------------------------------
# 5b. Compare with FantasyPros expert rankings (when the archive has that week)
# ---------------------------------------------------------------
def normalize_name(s):
    return (
        s.lower().replace(".", "").replace("'", "").replace(" jr", "")
        .replace(" sr", "").replace(" ii", "").replace(" iii", "")
        .replace(" iv", "").strip()
    )


def fantasypros_for_week(season, week):
    """FantasyPros rankings for one week, or (None, reason) if not available."""
    if not os.path.exists("fp_weekly_rankings.parquet"):
        return None, "fp_weekly_rankings.parquet not found"
    fp = pd.read_parquet("fp_weekly_rankings.parquet")
    fp = fp[["scrape_date", "position", "ecr", "mergename"]].sort_values("scrape_date")

    g = games.copy()
    g["gameday"] = pd.to_datetime(g["gameday"])
    week_starts = (
        g.groupby(["season", "week"])["gameday"].min()
        .reset_index().rename(columns={"gameday": "week_start"})
        .sort_values("week_start")
    )
    mapped = pd.merge_asof(
        fp, week_starts, left_on="scrape_date", right_on="week_start",
        direction="forward", tolerance=pd.Timedelta("6 days"),
    ).dropna(subset=["season", "week"])
    mapped["season"] = mapped["season"].astype(int)
    mapped["week"] = mapped["week"].astype(int) + FP_WEEK_OFFSET

    sel = mapped[(mapped["season"] == season) & (mapped["week"] == week)]
    if sel.empty:
        latest = fp["scrape_date"].max()
        return None, (f"the FantasyPros archive has no rankings for {season} week {week} "
                      f"yet (its latest scrape is {pd.Timestamp(latest).date()}). "
                      f"It updates weekly, so try again after the next refresh.")
    sel = sel[sel["scrape_date"] == sel["scrape_date"].max()].copy()
    sel["mergename"] = sel["mergename"].apply(lambda x: normalize_name(str(x)))
    return sel[["mergename", "position", "ecr"]].drop_duplicates(["mergename", "position"]), None


print("\nFantasyPros comparison")
fp_rows = {}
fpr, why = fantasypros_for_week(season, week)
if fpr is None:
    print(f"  Skipped: {why}")
else:
    scored["mergename"] = scored["player"].apply(normalize_name)
    both = scored.merge(fpr, on=["mergename", "position"], how="inner")
    print(f"  Matched {len(both)} of {len(scored)} scored players to FantasyPros rankings.")
    print(f"  Ranking correlation with actual results, same players for both "
          f"(higher is better):")
    print(f"  {'Pos':<4} {'N':>5} {'Our model':>10} {'FantasyPros':>12}")
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = both[both["position"] == pos]
        if len(sub) < FP_MIN_PLAYERS:
            continue
        our_rho = sub["median"].corr(sub["actual"], method="spearman")
        fp_rho = (-sub["ecr"]).corr(sub["actual"], method="spearman")
        fp_rows[pos] = {"fp_n": len(sub), "our_rho_matched": round(our_rho, 3), "fp_rho": round(fp_rho, 3)}
        print(f"  {pos:<4} {len(sub):>5} {our_rho:>10.3f} {fp_rho:>12.3f}")
    if not fp_rows:
        print(f"  (fewer than {FP_MIN_PLAYERS} matched players at every position)")
    else:
        print("  Fairness note: FantasyPros rankings are scraped later in the week and "
              "reflect injury news our model doesn't have,\n  so expect FantasyPros to look "
              "better in weeks with many late injuries.")

# ---------------------------------------------------------------
# 6. Save results and update the running log
# ---------------------------------------------------------------
out_cols = ["player", "position", "team", "opponent", "floor", "median", "ceiling",
            "actual", "naive", "status"]
out = preds[[c for c in out_cols if c in preds.columns]].copy()
out["error"] = (out["actual"] - out["median"]).round(2)
out.round(2).to_csv(f"scored_{season}_week{week}.csv", index=False)

new_log = pd.DataFrame([{"season": season, "week": week, "position": pos, **r, **fp_rows.get(pos, {})}
                        for pos, r in rows.items()]).round(3)
log_path = "accuracy_log.csv"
if os.path.exists(log_path):
    old = pd.read_csv(log_path)
    old = old[~((old["season"] == season) & (old["week"] == week))]
    new_log = pd.concat([old, new_log], ignore_index=True)
new_log.sort_values(["season", "week", "position"]).to_csv(log_path, index=False)

print(f"\nSaved scored_{season}_week{week}.csv and updated {log_path}.")
