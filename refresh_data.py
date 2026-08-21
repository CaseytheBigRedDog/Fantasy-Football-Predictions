"""
refresh_data.py

Re-downloads all raw data from nflverse/dynastyprocess (same sources as the
original pipeline) and overwrites the local copies, then re-runs the full
modeling pipeline (01-04) so predictions reflect the latest available games.

Safe to run anytime: works fine mid-season (adds new completed weeks as they
happen) and does nothing harmful if run in the off-season (just re-pulls the
same data). Designed to be scheduled weekly via Windows Task Scheduler --
see SCHEDULING.md for setup instructions.
"""
import subprocess
import sys
import urllib.request
import pandas as pd
from datetime import datetime

CURRENT_YEAR = datetime.now().year

def download(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [skip] {url} -> {e}")
        return False

print(f"=== Refreshing data ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===\n")

# ---------------------------------------------------------------
# 1. Core files: player stats, schedules/Vegas lines, player ID crosswalk
# ---------------------------------------------------------------
print("Downloading core files...")
download(
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv",
    "player_stats.csv",
)
download(
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
    "games.csv",
)
download(
    "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv",
    "players_master.csv",
)

# ---------------------------------------------------------------
# 2. Snap counts: one file per season, 2013 through the current year.
#    Skips gracefully if a season's file doesn't exist yet (e.g. before
#    week 1 stats are posted).
# ---------------------------------------------------------------
import os
os.makedirs("snap_data", exist_ok=True)
print("\nDownloading snap counts by season...")
got_any_current_year = False
for yr in range(2013, CURRENT_YEAR + 1):
    ok = download(
        f"https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{yr}.csv",
        f"snap_data/snap_counts_{yr}.csv",
    )
    if ok and yr == CURRENT_YEAR:
        got_any_current_year = True

if not got_any_current_year:
    print(f"  Note: no snap count data found yet for {CURRENT_YEAR} "
          f"(normal before the season starts or early in week 1).")

# ---------------------------------------------------------------
# 3. FantasyPros weekly rankings archive (for the comparison script)
# ---------------------------------------------------------------
print("\nDownloading FantasyPros rankings archive...")
download(
    "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.parquet",
    "fp_ecr_all.parquet",
)
try:
    df = pd.read_parquet("fp_ecr_all.parquet")
    pages = {
        "QB": ["qb", "/nfl/rankings/qb.php"],
        "RB": ["ppr-rb", "/nfl/rankings/ppr-rb.php"],
        "WR": ["ppr-wr", "/nfl/rankings/ppr-wr.php"],
        "TE": ["ppr-te", "/nfl/rankings/ppr-te.php"],
    }
    frames = []
    for pos, pg in pages.items():
        sub = df[df["fp_page"].isin(pg)].copy()
        sub["position"] = pos
        frames.append(sub)
    fp = pd.concat(frames, ignore_index=True)
    fp["scrape_date"] = pd.to_datetime(fp["scrape_date"])
    fp = fp[fp.scrape_date.dt.year >= 2023]
    fp.to_parquet("fp_weekly_rankings.parquet", index=False)
    print(f"  Saved fp_weekly_rankings.parquet ({len(fp):,} rows)")
except Exception as e:
    print(f"  [warning] Could not process FantasyPros archive: {e}")

# ---------------------------------------------------------------
# 4. Re-run the full modeling pipeline with fresh data
# ---------------------------------------------------------------
print("\n=== Re-running pipeline ===\n")
for script in [
    "01_load_data.py", "02_features.py", "03_train_model.py",
    "03b_position_quantile_models.py", "04_fantasypros_comparison.py",
]:
    print(f"--- Running {script} ---")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"!!! {script} failed (exit code {result.returncode}) -- stopping.")
        sys.exit(1)

print("\n=== Refresh complete ===")
