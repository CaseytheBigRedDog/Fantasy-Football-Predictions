"""
Step 2: Feature engineering

Key principle: every feature must be knowable BEFORE the game is played.
That means all player-form features use .shift(1) so a player's week-5
row only sees data through week 4. This is the #1 way people accidentally
leak the future into these models.

Changes in this version:
  * Rolling / season-to-date features are now computed PER PLAYER. The old
    version rolled across the whole table, so a player's first few games
    picked up the previous player's stats.
  * Builds "upcoming week" rows (one per likely player on a team that has an
    unplayed game) using the exact same feature code as the training rows,
    and saves them to upcoming_data.parquet for predict_week.py.
"""
import pandas as pd
import numpy as np

stats = pd.read_parquet("stats_clean.parquet")
games = pd.read_parquet("games_clean.parquet")

stats["is_upcoming"] = False

# ---------------------------------------------------------------
# 0. Upcoming week: one placeholder row per likely player
# ---------------------------------------------------------------
# The next week to predict = the earliest week that still has an unplayed
# game (no final score yet), in the latest season we have data for.
unplayed = games[games["home_score"].isna() & (games["season"] >= stats["season"].max())]
if len(unplayed):
    first = unplayed.sort_values(["season", "week"]).iloc[0]
    target_season, target_week = int(first["season"]), int(first["week"])
    todo = unplayed[(unplayed["season"] == target_season) & (unplayed["week"] == target_week)]

    # team -> opponent, for teams whose game hasn't been played yet
    sched = pd.concat([
        todo[["home_team", "away_team"]].rename(columns={"home_team": "recent_team", "away_team": "opponent_team"}),
        todo[["away_team", "home_team"]].rename(columns={"away_team": "recent_team", "home_team": "opponent_team"}),
    ], ignore_index=True)

    # Likely players = anyone who played in the last few weeks. Their team is
    # the team on their most recent row. (Off-season trades are not handled.)
    cur = stats[stats["season"] == target_season]
    if target_week > 1 and len(cur):
        recent = cur[cur["week"] >= target_week - 3]
    else:
        last_season = stats["season"].max()
        s = stats[stats["season"] == last_season]
        recent = s[s["week"] >= s["week"].max() - 3]
    id_cols = [c for c in ["player_id", "player_name", "player_display_name", "position", "recent_team"]
               if c in recent.columns]
    cand = recent.sort_values(["season", "week"]).groupby("player_id").tail(1)[id_cols]

    # skip anyone who already has a real row for that week (their game was played)
    already = stats.loc[(stats["season"] == target_season) & (stats["week"] == target_week), "player_id"]
    cand = cand[~cand["player_id"].isin(already)]

    upcoming = cand.merge(sched, on="recent_team", how="inner")
    upcoming["season"] = target_season
    upcoming["week"] = target_week
    upcoming["season_type"] = "REG"
    upcoming["is_upcoming"] = True
    stats = pd.concat([stats, upcoming], ignore_index=True)
    print(f"Upcoming week: {target_season} week {target_week} -> {len(upcoming):,} player rows")
else:
    print("No unplayed games found in the schedule -- no upcoming rows built.")

stats = stats.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

# ---------------------------------------------------------------
# 1. Rolling player form features (using only PAST games)
# ---------------------------------------------------------------
# Core volume/opportunity stats that predict fantasy points better than
# fantasy points themselves do (targets/carries are "stickier" week to week
# than scoring, which is noisier due to TD variance).
roll_cols = [
    "fantasy_points_ppr", "targets", "receptions", "receiving_yards",
    "carries", "rushing_yards", "attempts", "passing_yards",
    "target_share", "air_yards_share", "wopr",
    "passing_epa", "rushing_epa", "receiving_epa",
]
roll_cols = [c for c in roll_cols if c in stats.columns]

grp = stats.groupby("player_id")
pid = stats["player_id"]

for window in [3, 5]:
    for col in roll_cols:
        # shift(1) first so the current week is excluded, THEN roll over the
        # past -- separately inside each player's own history.
        shifted = grp[col].shift(1)
        stats[f"{col}_r{window}"] = (
            shifted.groupby(pid).rolling(window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

# Career-to-date average (through prior week) as a slower-moving signal
for col in roll_cols:
    shifted = grp[col].shift(1)
    stats[f"{col}_seasontd"] = (
        shifted.groupby(pid).expanding().mean().reset_index(level=0, drop=True)
    )

# Games of history available -- lets the model learn to trust rookies' rolling
# stats less (small sample) vs. veterans
stats["games_played_prior"] = grp.cumcount()

# ---------------------------------------------------------------
# 2. Opponent defense strength (fantasy points allowed to this position,
#    trailing, computed leak-free the same way)
# ---------------------------------------------------------------
def_allowed = (
    stats.groupby(["season", "week", "opponent_team", "position"])["fantasy_points_ppr"]
    .sum()
    .reset_index()
    .rename(columns={"fantasy_points_ppr": "pos_pts_allowed", "opponent_team": "defteam"})
)
# average per game allowed at that position, trailing 5 weeks, per team
def_allowed = def_allowed.sort_values(["defteam", "position", "season", "week"])
def_allowed["def_pts_allowed_r5"] = (
    def_allowed.groupby(["defteam", "position"])["pos_pts_allowed"]
    .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    .reset_index(level=[0, 1], drop=True)
)
def_allowed = def_allowed.rename(columns={"defteam": "opponent_team"})

stats = stats.merge(
    def_allowed[["season", "week", "opponent_team", "position", "def_pts_allowed_r5"]],
    on=["season", "week", "opponent_team", "position"],
    how="left",
)

# ---------------------------------------------------------------
# 3. Game context: Vegas lines, home/away, rest
# ---------------------------------------------------------------
home_games = games[["season", "week", "home_team", "away_team", "spread_line", "total_line", "home_rest", "away_rest", "roof", "surface"]].copy()

# Build a long (team-per-row) version of the schedule so we can merge on player's team
home_side = home_games.rename(columns={
    "home_team": "recent_team", "away_team": "opp_tmp",
    "home_rest": "rest_days",
})
home_side["is_home"] = 1
# nflverse convention: spread_line > 0 means the HOME team is favored by that many
# points. Verified empirically against known game lines before trusting this.
home_side["team_implied_total"] = (home_side["total_line"] / 2) + (home_side["spread_line"] / 2)

away_side = home_games.rename(columns={
    "away_team": "recent_team", "home_team": "opp_tmp",
    "away_rest": "rest_days",
})
away_side["is_home"] = 0
away_side["team_implied_total"] = (away_side["total_line"] / 2) - (away_side["spread_line"] / 2)

game_context = pd.concat([
    home_side[["season", "week", "recent_team", "spread_line", "total_line", "team_implied_total", "is_home", "rest_days", "roof", "surface"]],
    away_side[["season", "week", "recent_team", "spread_line", "total_line", "team_implied_total", "is_home", "rest_days", "roof", "surface"]],
], ignore_index=True)

stats = stats.merge(game_context, on=["season", "week", "recent_team"], how="left")

# team_implied_total: how many points Vegas expects THIS player's team to score.
# For the away side we flip the spread sign convention (spread_line is home - away in nflverse)
# double check sign empirically below in the sanity check script.

# ---------------------------------------------------------------
# 3b. Snap counts / snap share (from PFR via nflverse)
# ---------------------------------------------------------------
import glob

snap_files = sorted(glob.glob("snap_data/snap_counts_*.csv"))
snaps = pd.concat([pd.read_csv(f) for f in snap_files], ignore_index=True)
snaps = snaps[snaps["game_type"] == "REG"]

# Crosswalk PFR player id -> GSIS player id (the id player_stats.csv uses)
crosswalk = pd.read_csv("players_master.csv", usecols=["gsis_id", "pfr_id"])
crosswalk = crosswalk.dropna(subset=["gsis_id", "pfr_id"])

snaps = snaps.merge(
    crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left"
)
snaps = snaps.rename(columns={"gsis_id": "player_id"})
snaps = snaps.dropna(subset=["player_id"])

snaps = snaps[["player_id", "season", "week", "offense_snaps", "offense_pct"]].drop_duplicates(
    subset=["player_id", "season", "week"]
)

stats = stats.merge(snaps, on=["player_id", "season", "week"], how="left")

# Rolling snap share -- this is one of the stickiest, most predictive signals
# for role/opportunity, especially for detecting a player's role INCREASING
# (e.g. earning more snaps after a teammate injury) before points catch up.
stats = stats.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
grp2 = stats.groupby("player_id")
pid2 = stats["player_id"]
for window in [3, 5]:
    stats[f"offense_pct_r{window}"] = (
        grp2["offense_pct"].shift(1).groupby(pid2).rolling(window, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    stats[f"offense_snaps_r{window}"] = (
        grp2["offense_snaps"].shift(1).groupby(pid2).rolling(window, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

# Snap share trend: this week's rolling-3 minus rolling-5 -- a positive value
# means the player's role has been GROWING recently, a leading indicator
# rolling fantasy points alone won't catch as fast.
stats["offense_pct_trend"] = stats["offense_pct_r3"] - stats["offense_pct_r5"]

snap_match_rate = stats.loc[~stats["is_upcoming"], "offense_pct"].notna().mean()
print(f"Snap count match rate: {snap_match_rate:.1%}")

# ---------------------------------------------------------------
# 4. Target: raw fantasy points (standard scoring uses 'fantasy_points';
#    PPR uses 'fantasy_points_ppr'. We'll predict PPR since it's the most
#    common league format -- swap the column if the user plays standard.)
# ---------------------------------------------------------------
stats["target_fp"] = stats["fantasy_points_ppr"]

# Drop rows with no rolling history at all (first game of a player's career) --
# model can't do much with zero prior information anyway.
model_df = stats.dropna(subset=["fantasy_points_ppr_r3"]).copy()

# A small remaining sliver of rows (unmatched player IDs in the PFR<->GSIS
# crosswalk) will still have null snap features even in the 2013+ window --
# fill those specifically, not as a blanket early-era patch.
snap_feat_cols = [c for c in model_df.columns if "offense_pct" in c or "offense_snaps" in c]
model_df[snap_feat_cols] = model_df[snap_feat_cols].fillna(0)

# Split: finished games train the model; upcoming rows are what we predict.
upcoming_df = model_df[model_df["is_upcoming"]].copy()
train_df = model_df[~model_df["is_upcoming"]].drop(columns=["is_upcoming"])

print(f"Feature-engineered rows: {len(train_df):,}")
print(f"Columns: {train_df.shape[1]}")

train_df.to_parquet("model_data.parquet", index=False)
upcoming_df.to_parquet("upcoming_data.parquet", index=False)
print("Saved model_data.parquet")
print(f"Saved upcoming_data.parquet ({len(upcoming_df):,} rows)")
