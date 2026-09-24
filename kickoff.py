"""
kickoff.py

Which teams' games have already kicked off? predict_week.py uses this to freeze
those teams' projections when you re-run predictions later in the week (for example
after Friday's injury reports), so a projection is never changed after its game
has started. That keeps the "committed before kickoff" track record honest.

Kickoff times in the schedule are US Eastern time. Daylight saving is handled here
directly, so no extra packages are needed.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd


def _nth_sunday(year, month, n):
    first = datetime(year, month, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def eastern_now():
    """Current US Eastern time as a naive datetime."""
    utc = datetime.now(timezone.utc).replace(tzinfo=None)
    dst_start = _nth_sunday(utc.year, 3, 2) + timedelta(hours=7)    # 2:00 a.m. EST
    dst_end = _nth_sunday(utc.year, 11, 1) + timedelta(hours=6)     # 2:00 a.m. EDT
    return utc + timedelta(hours=-4 if dst_start <= utc < dst_end else -5)


def frozen_teams(games, season, week, now=None):
    """Set of team abbreviations whose game in (season, week) has already kicked off."""
    now = now or eastern_now()
    g = games[(games["season"] == season) & (games["week"] == week)]
    if g.empty or "gameday" not in g.columns:
        return set()
    teams = set()
    for _, r in g.iterrows():
        t = str(r["gametime"]) if "gametime" in g.columns and pd.notna(r.get("gametime")) else "13:00"
        try:
            kickoff = datetime.strptime(f"{str(r['gameday'])[:10]} {t[:5]}", "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if kickoff <= now:
            teams.add(r["home_team"])
            teams.add(r["away_team"])
    return teams
