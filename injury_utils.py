"""
injury_utils.py

Small shared helpers for reading nflverse injury reports.

The raw report says things like "Out", "Doubtful" or "Questionable" (game-day
designation) and "Did Not Participate In Practice" / "Limited Participation in
Practice" / "Full Participation in Practice" (practice status). These helpers turn
them into a few simple categories. Matching is by keyword, so small wording
changes in the source data don't break it.
"""
import pandas as pd

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def norm_report(value):
    """Game-day designation: out / doubtful / questionable / probable / none / other."""
    if pd.isna(value):
        return "none"
    t = str(value).lower()
    if "doubtful" in t:
        return "doubtful"
    if "questionable" in t:
        return "questionable"
    if "probable" in t:
        return "probable"
    if "out" in t or "injured reserve" in t or t.strip() in ("ir", "pup", "nfi", "suspended"):
        return "out"
    return "other"


def norm_practice(value):
    """Practice status: dnp / limited / full / none / other."""
    if pd.isna(value):
        return "none"
    t = str(value).lower()
    if "did not" in t or "dnp" in t or "out" in t:
        return "dnp"
    if "limited" in t:
        return "limited"
    if "full" in t:
        return "full"
    return "other"
