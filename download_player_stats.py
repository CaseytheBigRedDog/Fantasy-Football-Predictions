"""
download_player_stats.py

Builds player_stats.csv from nflverse's per-season weekly player files.

Why this exists: the old combined player_stats.csv on nflverse stopped
updating after the 2024 season. Weekly stats now live in one file per season
(stats_player_week_<season>.parquet). This script downloads each season,
converts it to the same column layout the old file had (so 01_load_data.py
and 02_features.py keep working unchanged), and writes player_stats.csv.

Run on its own to test:   python download_player_stats.py
Or it is called by refresh_data.py.
"""
import os
import urllib.request
from datetime import datetime

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player"
FIRST_SEASON = 2013
DATA_DIR = "stats_data"
LEGACY_COLS_FILE = "legacy_columns.txt"

# Columns renamed in the new files (new name -> old name)
RENAMES = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
}

# The new files include rows for every player who was on a roster. Keep only
# rows where the player actually recorded something, like the old file did.
ACTIVITY_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
    "receiving_yards", "receiving_tds", "special_teams_tds",
    "fantasy_points", "fantasy_points_ppr",
]


def get_legacy_columns(out_csv):
    """The column list the rest of the pipeline was built on.

    Read once from your existing player_stats.csv (the old-format file) and
    remembered in legacy_columns.txt, so later runs keep the same layout.
    """
    if os.path.exists(LEGACY_COLS_FILE):
        with open(LEGACY_COLS_FILE) as f:
            return f.read().split()
    if os.path.exists(out_csv):
        cols = list(pd.read_csv(out_csv, nrows=0).columns)
        with open(LEGACY_COLS_FILE, "w") as f:
            f.write("\n".join(cols))
        return cols
    return None


def convert_season(df, legacy_cols):
    """Turn one season's new-format table into the old layout."""
    for new, old in RENAMES.items():
        if new in df.columns and old not in df.columns:
            df = df.rename(columns={new: old})

    # New files store sack yards as a negative number; the old file was positive.
    if "sack_yards_lost" in df.columns and "sack_yards" not in df.columns:
        df["sack_yards"] = -df["sack_yards_lost"]

    # 'dakota' is no longer published; keep the column so the layout is stable.
    if "dakota" not in df.columns:
        df["dakota"] = float("nan")

    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]

    cols = [c for c in ACTIVITY_COLS if c in df.columns]
    df = df[(df[cols].fillna(0) != 0).any(axis=1)]

    if legacy_cols is not None:
        missing = [c for c in legacy_cols if c not in df.columns]
        if missing:
            print(f"  Note: columns missing from new data, filled empty: {missing}")
        df = df.reindex(columns=legacy_cols)
    return df


def download_player_stats(current_year, first_season=FIRST_SEASON, out_csv="player_stats.csv"):
    os.makedirs(DATA_DIR, exist_ok=True)
    legacy_cols = get_legacy_columns(out_csv)
    if legacy_cols is None:
        print("  Warning: no existing player_stats.csv to copy the column layout "
              "from; keeping all columns from the new files.")

    frames = []
    for yr in range(first_season, current_year + 1):
        url = f"{BASE}/stats_player_week_{yr}.parquet"
        dest = os.path.join(DATA_DIR, f"stats_player_week_{yr}.parquet")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  [skip] {yr}: {e}")
            if not os.path.exists(dest):
                continue
            print(f"  using the copy already saved for {yr}")
        try:
            frames.append(convert_season(pd.read_parquet(dest), legacy_cols))
        except Exception as e:
            print(f"  [error] could not read {yr}: {e}")

    if not frames:
        print("  No player stats were downloaded; leaving player_stats.csv unchanged.")
        return False

    stats = pd.concat(frames, ignore_index=True)
    stats.to_csv(out_csv, index=False)

    print(f"  Saved {out_csv}: {len(stats):,} rows, seasons "
          f"{stats['season'].min()}-{stats['season'].max()}")
    latest = stats["season"].max()
    reg = stats[(stats["season"] == latest) & (stats["season_type"] == "REG")]
    if len(reg):
        print(f"  Latest season {latest}: weeks through {reg['week'].max()}")
    return True


if __name__ == "__main__":
    print("Downloading player stats...")
    download_player_stats(datetime.now().year)
