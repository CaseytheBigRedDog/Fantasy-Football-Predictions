"""
apply_injury_download.py

Adds the injury report / depth chart download to refresh_data.py, right after the
player stats download, so every refresh keeps them current.

Run once from your Fantasy-Football-Predictions folder:
    python apply_injury_download.py

It refuses to run twice. To undo: git checkout refresh_data.py
"""
import sys

name = "refresh_data.py"
with open(name, encoding="utf-8", newline="") as f:
    s = f.read()

if "download_injury_data" in s:
    print("refresh_data.py already downloads injury data. Nothing changed.")
    sys.exit(0)

anchor = "download_player_stats(CURRENT_YEAR)"
if s.count(anchor) != 1:
    print("Could not find the player stats download line in refresh_data.py; no changes made.")
    sys.exit(1)

nl = "\r\n" if "\r\n" in s else "\n"
addition = nl.join([
    anchor,
    "",
    "# Injury reports (2013+) and depth charts (2025+)",
    "from download_injuries import download_injury_data",
    "download_injury_data(CURRENT_YEAR)",
])
with open(name, "w", encoding="utf-8", newline="") as f:
    f.write(s.replace(anchor, addition, 1))
print("refresh_data.py: injury and depth chart download added.")
print("\nNext: python download_injuries.py   (or just run refresh_data.py)")
