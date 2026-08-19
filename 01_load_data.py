"""
Step 1: Load and clean raw data
- player_stats.csv: weekly player stats + fantasy points (nflverse)
- games.csv: schedules, results, Vegas lines (nflverse)
"""
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

# ---- Load weekly player stats ----
stats = pd.read_csv("player_stats.csv", low_memory=False)

# Keep regular season only, and only the four fantasy-relevant positions
stats = stats[stats["season_type"] == "REG"]
stats = stats[stats["position"].isin(["QB", "RB", "WR", "TE"])]

# Restrict to a reasonable modern era: passing rules, roster usage, and pace
# have shifted enough that pre-2010 data adds noise more than signal.
stats = stats[stats["season"] >= 2010]

print(f"player_stats rows after filtering: {len(stats):,}")
print(f"seasons: {stats['season'].min()}-{stats['season'].max()}")
print(f"positions: {stats['position'].value_counts().to_dict()}")

# ---- Load games / schedule / Vegas lines ----
games = pd.read_csv("games.csv", low_memory=False)
games = games[games["season"] >= 2010]
games = games[games["game_type"] == "REG"]

print(f"\ngames rows after filtering: {len(games):,}")

# Save intermediate cleaned files
stats.to_parquet("stats_clean.parquet", index=False)
games.to_parquet("games_clean.parquet", index=False)
print("\nSaved stats_clean.parquet and games_clean.parquet")
