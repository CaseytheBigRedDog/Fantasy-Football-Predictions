"""
top_players.py

Prints the top players at each position from the latest weekly predictions file.

Usage:
    python top_players.py              # top 20 at every position
    python top_players.py 10           # top 10 at every position
    python top_players.py 25 WR        # top 25 wide receivers only
    python top_players.py 20 all 2026 4   # a specific season and week

Columns: floor / median / ceiling / expected (the average outcome, comparable with
ESPN-style projections). Players are ranked by expected points.
Official injury designations are applied when the file has them; late scratches and news are not included.
"""
import glob
import os
import re
import sys

import pandas as pd

args = sys.argv[1:]
n = int(args[0]) if args and args[0].isdigit() else 20
position = args[1].upper() if len(args) > 1 and args[1].lower() != "all" else None
if position and position not in ("QB", "RB", "WR", "TE"):
    print("Position must be QB, RB, WR, TE or all.")
    sys.exit(1)

if len(args) >= 4:
    season, week = int(args[2]), int(args[3])
else:
    found = []
    for fn in glob.glob("predictions_*_week*.csv"):
        m = re.search(r"predictions_(\d{4})_week(\d+)\.csv$", fn)   # skips the _baseline copy
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    if not found:
        print("No predictions file found. Run predict_week.py first.")
        sys.exit(1)
    season, week = max(found)

path = f"predictions_{season}_week{week}.csv"
if not os.path.exists(path):
    print(f"Could not find {path}.")
    sys.exit(1)

df = pd.read_csv(path)
rank_col = "expected" if "expected" in df.columns else "median"
print(f"{path}  |  top {n} by {rank_col}  |  floor / median / ceiling / expected\n")

for pos in ([position] if position else ["QB", "RB", "WR", "TE"]):
    top = df[df["position"] == pos].sort_values(rank_col, ascending=False).head(n)
    if top.empty:
        continue
    print(f"--- Top {len(top)} {pos} ---")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        where = "vs" if r["is_home"] == 1 else "@"
        exp = f"   expected {r['expected']:>5.1f}" if "expected" in df.columns else ""
        status = r["status"] if "status" in df.columns and isinstance(r["status"], str) else ""
        depth = r["depth"] if "depth" in df.columns and isinstance(r["depth"], str) else ""
        print(f"{i:>2}. {r['player']:<24} {depth:<4} {r['team']:>3} {where} {r['opponent']:<3} "
              f"{r['floor']:>5.1f} / {r['median']:>5.1f} / {r['ceiling']:>5.1f}{exp}"
              + (f"   [{status}]" if status else ""))
    print()
