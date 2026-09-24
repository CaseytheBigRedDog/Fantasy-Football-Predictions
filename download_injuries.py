"""
download_injuries.py

Downloads two things from nflverse into local folders:

  injury_data/injuries_<season>.csv       weekly injury reports, 2013 to now
  depth_data/depth_charts_<season>.csv    depth charts, 2025 to now (the data source and
                                          layout changed after the 2024 season, so older
                                          seasons are not compatible)

Run on its own to test:   python download_injuries.py
It is also called by refresh_data.py once apply_injury_download.py has hooked it in.

A missing file is skipped (with a message) rather than stopping anything, and an
older copy is kept if a new download fails.
"""
import os
import urllib.request
from datetime import datetime

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_INJURY_SEASON = 2013   # matches the rest of the pipeline
FIRST_DEPTH_SEASON = 2025


def _get(url, dest):
    """Download url to dest. Returns True if a usable file exists afterwards."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [skip] {os.path.basename(dest)}: {e}")
        return os.path.exists(dest)


def download_injury_data(current_year, first_injury=FIRST_INJURY_SEASON, first_depth=FIRST_DEPTH_SEASON):
    os.makedirs("injury_data", exist_ok=True)
    os.makedirs("depth_data", exist_ok=True)

    got = 0
    for yr in range(first_injury, current_year + 1):
        if _get(f"{BASE}/injuries/injuries_{yr}.csv", f"injury_data/injuries_{yr}.csv"):
            got += 1
    print(f"  Injury reports: {got} season file(s) in injury_data/")

    latest = f"injury_data/injuries_{current_year}.csv"
    if os.path.exists(latest):
        try:
            d = pd.read_csv(latest, usecols=["week", "game_type"])
            reg = d[d["game_type"] == "REG"]
            if len(reg):
                print(f"  {current_year} injury reports run through week {int(reg['week'].max())}")
        except Exception:
            pass

    got_depth = 0
    for yr in range(first_depth, current_year + 1):
        if _get(f"{BASE}/depth_charts/depth_charts_{yr}.csv", f"depth_data/depth_charts_{yr}.csv"):
            got_depth += 1
    print(f"  Depth charts: {got_depth} season file(s) in depth_data/")
    return got > 0


if __name__ == "__main__":
    print("Downloading injury reports and depth charts...")
    download_injury_data(datetime.now().year)
