"""
Step 4: Compare our model against FantasyPros expert consensus rankings (ECR).

Data source: dynastyprocess.com/data, which mirrors FantasyPros' public
rankings pages on a scheduled scrape (this is the same data source the
ffverse R packages use -- see https://github.com/dynastyprocess/data).
Coverage: weekly PPR positional rankings, 2019-2025.

IMPORTANT SCOPE NOTE: this script compares against FantasyPros' expert
CONSENSUS RANKINGS (a rank ordering: "who's the better play"), not raw
projected points. Our model predicts points, so we convert our predictions
to within-position ranks for an apples-to-apples comparison. See the bottom
of this script for why Yahoo/Sleeper/ESPN specifically aren't included.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------
# Load FantasyPros weekly rankings, map each scrape date to the NFL week
# it was published for (the Friday scrape ahead of that week's games)
# ---------------------------------------------------------------
fp = pd.read_parquet("fp_weekly_rankings.parquet")
fp = fp[["scrape_date", "player", "position", "ecr", "mergename"]].copy()
fp = fp.sort_values("scrape_date")

games = pd.read_parquet("games_clean.parquet")
games["gameday"] = pd.to_datetime(games["gameday"])
week_starts = (
    games.groupby(["season", "week"])["gameday"].min()
    .reset_index().rename(columns={"gameday": "week_start"})
    .sort_values("week_start")
)

# For each FP scrape date, find the NEXT upcoming game week (the week those
# rankings were published for) via merge_asof with direction='forward'
fp_mapped = pd.merge_asof(
    fp, week_starts, left_on="scrape_date", right_on="week_start",
    direction="forward", tolerance=pd.Timedelta("6 days"),
)
fp_mapped = fp_mapped.dropna(subset=["season", "week"])
fp_mapped["season"] = fp_mapped["season"].astype(int)
fp_mapped["week"] = fp_mapped["week"].astype(int)

# Within each (season, week, position), rank FP's ecr ascending (1 = best)
fp_mapped["fp_rank"] = fp_mapped.groupby(["season", "week", "position"])["ecr"].rank(method="first")

# ---------------------------------------------------------------
# Load our model's predictions and convert to within-position ranks too
# ---------------------------------------------------------------
ours = pd.read_parquet("test_predictions_v2.parquet")  # 2024 test set only
ours["our_rank"] = ours.groupby(["season", "week", "position"])["pred_median"].rank(
    method="first", ascending=False  # higher predicted points = better rank
)
ours["actual_rank"] = ours.groupby(["season", "week", "position"])["target_fp"].rank(
    method="first", ascending=False
)

# ---------------------------------------------------------------
# Join on player name (mergename in FP data is a cleaned/normalized name --
# build the same normalization on our side for a decent match rate)
# ---------------------------------------------------------------
def normalize_name(s):
    return (
        s.lower().replace(".", "").replace("'", "").replace(" jr", "")
        .replace(" sr", "").replace(" ii", "").replace(" iii", "")
        .replace(" iv", "").strip()
    )

ours["mergename"] = ours["player_display_name"].apply(normalize_name)
fp_mapped["mergename"] = fp_mapped["mergename"].apply(lambda x: normalize_name(str(x)))

merged = ours.merge(
    fp_mapped[["season", "week", "mergename", "fp_rank"]],
    on=["season", "week", "mergename"], how="inner",
)
print(f"Matched {len(merged):,} of {len(ours):,} of our 2024 test-set rows to FantasyPros rankings "
      f"({len(merged)/len(ours):.1%} match rate)")

# ---------------------------------------------------------------
# Compare: whose rank ordering better predicted the ACTUAL outcome?
# Spearman correlation between predicted rank and actual rank, by position.
# Higher (closer to 1.0) = better at ordering players correctly that week.
# ---------------------------------------------------------------
print(f"\n{'Position':<10}{'Our model (rho)':<20}{'FantasyPros (rho)':<20}{'N':<8}")
print("-" * 58)
summary_rows = []
for pos in ["QB", "RB", "WR", "TE"]:
    sub = merged[merged["position"] == pos].dropna(subset=["our_rank", "fp_rank", "actual_rank"])
    if len(sub) < 20:
        continue
    our_rho, _ = spearmanr(sub["our_rank"], sub["actual_rank"])
    fp_rho, _ = spearmanr(sub["fp_rank"], sub["actual_rank"])
    print(f"{pos:<10}{our_rho:<20.3f}{fp_rho:<20.3f}{len(sub):<8}")
    summary_rows.append({"position": pos, "our_rho": our_rho, "fp_rho": fp_rho, "n": len(sub)})

summary = pd.DataFrame(summary_rows)
summary.to_csv("fantasypros_comparison.csv", index=False)
print("\nSaved fantasypros_comparison.csv")

print("""
--------------------------------------------------------------------
NOTE ON YAHOO / SLEEPER / ESPN:
These platforms don't publish a free, bulk-downloadable historical archive
of their own weekly rankings or point projections -- unlike FantasyPros,
whose public rankings pages are scraped and mirrored by the open-source
ffverse/dynastyprocess project used above. The FantasyPros dataset does
include Yahoo/ESPN *roster ownership %* columns (player_owned_yahoo,
player_owned_espn), but that measures how many leagues have a player
rostered, not a points prediction -- a different thing.

If a comparison against current Yahoo/Sleeper/ESPN rankings specifically
would help, the more realistic version of that is a live, current-week
snapshot (e.g. checking this week's ESPN/Yahoo rankings against our
model's picks going forward), not a multi-season backtest, since there's
no free historical archive to backtest against for those three.
--------------------------------------------------------------------
""")
