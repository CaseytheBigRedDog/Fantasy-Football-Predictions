"""
check_fp_alignment.py

Checks whether the FantasyPros rankings are lined up with the right NFL week.

04_fantasypros_comparison.py labels each weekly scrape with the NEXT week whose
first game is within 6 days. The archive's scrapes all happen on Fridays, and
by Friday that week's Thursday game has been played, so the label might be one
week too late (a Friday scrape holding Week 2 rankings getting called Week 3).

The test: try the labels as they are (shift 0), one week earlier (-1) and one
week later (+1), and see which one makes the rankings agree best with what
actually happened. The right alignment should win clearly.

Run:  python check_fp_alignment.py
"""
import pandas as pd

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

# Same mapping 04_fantasypros_comparison.py uses
fp_mapped = pd.merge_asof(
    fp, week_starts, left_on="scrape_date", right_on="week_start",
    direction="forward", tolerance=pd.Timedelta("6 days"),
)
fp_mapped = fp_mapped.dropna(subset=["season", "week"])
fp_mapped["season"] = fp_mapped["season"].astype(int)
fp_mapped["week"] = fp_mapped["week"].astype(int)


def normalize_name(s):
    return (
        s.lower().replace(".", "").replace("'", "").replace(" jr", "")
        .replace(" sr", "").replace(" ii", "").replace(" iii", "")
        .replace(" iv", "").strip()
    )


fp_mapped["mergename"] = fp_mapped["mergename"].apply(lambda x: normalize_name(str(x)))

ours = pd.read_parquet("test_predictions_v2.parquet")  # the test season
ours["actual_rank"] = ours.groupby(["season", "week", "position"])["target_fp"].rank(
    method="first", ascending=False
)
ours["mergename"] = ours["player_display_name"].apply(normalize_name)

print("Correlation between FantasyPros rank and actual results, by week shift")
print("(higher is better; the right alignment should stand out)\n")
print(f"{'Shift':<11}{'QB':>7}{'RB':>7}{'WR':>7}{'TE':>7}{'Avg':>8}{'Matched rows':>15}")

results = {}
for shift in (-1, 0, 1):
    f = fp_mapped.copy()
    f["week"] = f["week"] + shift
    f = f[f["week"] >= 1]
    f["fp_rank"] = f.groupby(["season", "week", "position"])["ecr"].rank(method="first")

    merged = ours.merge(
        f[["season", "week", "mergename", "fp_rank"]],
        on=["season", "week", "mergename"], how="inner",
    )
    rhos = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = merged[merged["position"] == pos].dropna(subset=["fp_rank", "actual_rank"])
        rhos[pos] = sub["fp_rank"].corr(sub["actual_rank"], method="spearman") if len(sub) >= 20 else float("nan")
    avg = pd.Series(rhos).mean()
    results[shift] = avg
    label = {-1: "-1", 0: "0 (as is)", 1: "+1"}[shift]
    print(f"{label:<11}{rhos['QB']:>7.3f}{rhos['RB']:>7.3f}{rhos['WR']:>7.3f}{rhos['TE']:>7.3f}"
          f"{avg:>8.3f}{len(merged):>15,}")

best = max(results, key=lambda k: results[k] if pd.notna(results[k]) else -9)
print()
if best == 0:
    print("Result: the labels are already lined up (shift 0 is best). No change needed.")
else:
    direction = "one week earlier" if best == -1 else "one week later"
    print(f"Result: shift {best:+d} is best -- each scrape's rankings belong {direction} "
          f"than the label says.")
    print(f"Fix: in score_predictions.py set FP_WEEK_OFFSET = {best}, and send me these results "
          f"so 04_fantasypros_comparison.py gets the same fix.")
print("If the top two shifts are close, the check is inconclusive; send me the table.")
