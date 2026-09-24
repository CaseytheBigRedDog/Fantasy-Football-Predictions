"""
inspect_injuries.py

STEP 1 of the injury / depth chart work: look at the real data before building on it.

Shows:
  1. what the injury files cover and which status words they actually use
  2. how predictive each designation has been (how often a player with that
     designation went on to record any stats, and how many points he scored if he did)
  3. YOUR current projections that carry an injury flag this week
  4. how the depth chart files are laid out

Run (after python download_injuries.py):
    python inspect_injuries.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from injury_utils import SKILL_POSITIONS, norm_practice, norm_report

files = sorted(glob.glob("injury_data/injuries_*.csv"))
if not files:
    raise SystemExit("No injury files found. Run:  python download_injuries.py")

inj = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
need = ["season", "week", "gsis_id", "position", "report_status", "practice_status"]
missing = [c for c in need if c not in inj.columns]
if missing:
    print("The injury files are missing expected columns:", missing)
    print("Columns found:", list(inj.columns))
    raise SystemExit(1)
if "game_type" not in inj.columns:
    inj["game_type"] = "REG"

skill = inj[inj["position"].isin(SKILL_POSITIONS)].copy()
skill["report"] = skill["report_status"].map(norm_report)
skill["practice"] = skill["practice_status"].map(norm_practice)
last_season = int(inj["season"].max())

# ---------------------------------------------------------------
print("=" * 70)
print("1. WHAT THE INJURY FILES COVER")
print("=" * 70)
print(f"{len(inj):,} report rows, seasons {int(inj['season'].min())}-{last_season}; "
      f"{len(skill):,} rows are QB/RB/WR/TE.")
cur = inj[(inj["season"] == last_season) & (inj["game_type"] == "REG")]
if len(cur):
    weeks = cur.groupby("week").size()
    print(f"{last_season} regular season: weeks with reports = "
          + ", ".join(f"{int(w)} ({n:,} rows)" for w, n in weeks.items()))
if "date_modified" in inj.columns:
    dm = pd.to_datetime(cur["date_modified"], errors="coerce")
    if dm.notna().any():
        print(f"Most recent update in the {last_season} file: {dm.max()}")

print("\nGame-day designation words in the data (QB/RB/WR/TE, all seasons):")
vc = skill["report_status"].fillna("(blank)").value_counts().head(8)
for raw, n in vc.items():
    print(f"  {raw!s:<28}{n:>8,}   -> treated as: {norm_report(None if raw == '(blank)' else raw)}")
print("\nPractice-status words:")
vc = skill["practice_status"].fillna("(blank)").value_counts().head(8)
for raw, n in vc.items():
    print(f"  {raw!s:<48}{n:>8,}   -> treated as: {norm_practice(None if raw == '(blank)' else raw)}")
others = skill[(skill["report"] == "other")]["report_status"].value_counts().head(5)
if len(others):
    print("\nDesignation words I did NOT recognize (tell me about these):")
    print(others.to_string())

# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("2. HOW PREDICTIVE HAVE THE DESIGNATIONS BEEN?")
print("=" * 70)
if not os.path.exists("stats_clean.parquet"):
    print("stats_clean.parquet not found (run refresh_data.py first); skipping.")
else:
    stats = pd.read_parquet("stats_clean.parquet")[["player_id", "season", "week", "fantasy_points_ppr"]]
    stats = stats.drop_duplicates(["player_id", "season", "week"])
    slot = lambda d: (d["season"] - 2013) * 20 + d["week"]      # a running week counter
    stats["idx"] = slot(stats)

    h = skill[(skill["game_type"] == "REG") & skill["gsis_id"].notna() & (skill["season"] >= last_season - 5)].copy()
    h["idx"] = slot(h)
    prev = pd.merge_asof(
        h.sort_values("idx"),
        stats.sort_values("idx")[["player_id", "idx"]].rename(columns={"idx": "prev_idx"}),
        left_on="idx", right_on="prev_idx", left_by="gsis_id", right_by="player_id",
        allow_exact_matches=False, direction="backward")
    # fantasy-relevant = recorded stats within roughly the previous 5 weeks
    h = prev[(prev["idx"] - prev["prev_idx"]) <= 5].copy()
    h = h.merge(stats[["player_id", "season", "week", "fantasy_points_ppr"]],
                left_on=["gsis_id", "season", "week"], right_on=["player_id", "season", "week"],
                how="left", suffixes=("", "_s"))
    h["played"] = h["fantasy_points_ppr"].notna()
    print(f"Players who recorded stats in the previous ~5 weeks, {last_season - 5}-{last_season}: "
          f"{len(h):,} player-weeks on an injury report.\n")

    def table(col, order, title):
        print(title)
        print(f"  {'':<14}{'Player-weeks':>13}{'Recorded stats':>16}{'Avg pts if they did':>21}")
        for k in order:
            g = h[h[col] == k]
            if len(g) >= 20:
                print(f"  {k:<14}{len(g):>13,}{g['played'].mean():>16.0%}"
                      f"{g.loc[g['played'], 'fantasy_points_ppr'].mean():>21.1f}")
        print()

    table("report", ["out", "doubtful", "questionable", "probable", "none"], "By game-day designation:")
    table("practice", ["dnp", "limited", "full", "none"], "By practice status:")
    q = h[h["report"].isin(["questionable", "doubtful"])]
    if len(q):
        print("Questionable / doubtful players, by practice status (share who recorded stats):")
        for rep in ["questionable", "doubtful"]:
            for pr in ["dnp", "limited", "full"]:
                g = q[(q["report"] == rep) & (q["practice"] == pr)]
                if len(g) >= 20:
                    print(f"  {rep:<13} + {pr:<8}{len(g):>7,} player-weeks   {g['played'].mean():>5.0%} recorded stats")
        print()

# ---------------------------------------------------------------
print("=" * 70)
print("3. YOUR CURRENT PROJECTIONS WITH AN INJURY FLAG")
print("=" * 70)
found = []
for fn in glob.glob("predictions_*_week*.csv"):
    m = re.search(r"predictions_(\d{4})_week(\d+)\.csv$", fn)
    if m:
        found.append(((int(m.group(1)), int(m.group(2))), fn))
if not found:
    print("No predictions file found.")
else:
    (S, W), path = max(found)
    preds = pd.read_csv(path)
    print(f"Projections: {path}")
    week_rows = inj[(inj["season"] == S) & (inj["week"] == W) & (inj["game_type"] == "REG")]
    if "player_id" not in preds.columns:
        print("This predictions file has no player_id column, so it can't be matched; rerun predict_week.py.")
    elif week_rows.empty:
        print(f"No injury reports for {S} week {W} yet. Designations usually appear Wednesday to Friday, "
              f"and the data updates about once a day.")
    else:
        week_rows = week_rows.assign(report=week_rows["report_status"].map(norm_report),
                                     practice=week_rows["practice_status"].map(norm_practice))
        cols = ["gsis_id", "report", "practice"] + (["report_primary_injury"] if "report_primary_injury" in week_rows.columns else [])
        j = preds.merge(week_rows.drop_duplicates("gsis_id")[cols], left_on="player_id", right_on="gsis_id", how="left")
        j["report"] = j["report"].fillna("none")
        j["practice"] = j["practice"].fillna("none")
        flagged = j[j["report"].isin(["out", "doubtful", "questionable"]) | j["practice"].isin(["dnp", "limited"])]
        ekey = "expected" if "expected" in j.columns else "median"
        print(f"{len(week_rows):,} players have a report for week {W}; {len(flagged)} of your "
              f"{len(preds)} projected players carry a flag.")
        risky = j[j["report"].isin(["out", "doubtful"])]
        if len(risky):
            print(f"{len(risky)} are OUT or DOUBTFUL, carrying {risky[ekey].sum():,.0f} projected points "
                  f"that will likely not happen.")
        print(f"\nFlagged players, highest projection first:")
        print(f"  {'Player':<24}{'Pos':<4}{'Team':<5}{'Proj':>6}  {'Designation':<13}{'Practice':<9}Injury")
        for _, r in flagged.sort_values(ekey, ascending=False).head(25).iterrows():
            inj_txt = str(r.get("report_primary_injury", "")) if pd.notna(r.get("report_primary_injury", np.nan)) else ""
            print(f"  {str(r['player']):<24}{r['position']:<4}{r['team']:<5}{r[ekey]:>6.1f}  "
                  f"{r['report']:<13}{r['practice']:<9}{inj_txt}")

# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("4. DEPTH CHART FILES")
print("=" * 70)
dfiles = sorted(glob.glob("depth_data/depth_charts_*.csv"))
if not dfiles:
    print("No depth chart files found. The download may have failed (look for '[skip]' lines when "
          "you ran download_injuries.py). Tell me what it said.")
else:
    for f in dfiles:
        d = pd.read_csv(f, low_memory=False)
        print(f"{os.path.basename(f)}: {len(d):,} rows")
    d = pd.read_csv(dfiles[-1], low_memory=False)
    print(f"\nColumns in the newest file: {list(d.columns)}")
    if "dt" in d.columns:
        dt = pd.to_datetime(d["dt"], errors="coerce")
        print(f"Snapshots (dt) run from {dt.min()} to {dt.max()}; {dt.nunique():,} distinct timestamps")
    for c in ["pos_grp", "pos_abb"]:
        if c in d.columns:
            print(f"\nMost common {c} values:")
            print(d[c].value_counts().head(12).to_string())
    if {"team", "pos_abb", "player_name"}.issubset(d.columns):
        if "dt" in d.columns:
            d = d[pd.to_datetime(d["dt"], errors="coerce") == pd.to_datetime(d["dt"], errors="coerce").max()]
        sample_team = d["team"].iloc[0]
        s = d[(d["team"] == sample_team) & (d["pos_abb"].isin(["WR", "RB", "TE", "QB"]))]
        keep = [c for c in ["player_name", "pos_abb", "pos_slot", "pos_rank"] if c in s.columns]
        print(f"\nSample: newest snapshot, team {sample_team}, offensive skill players")
        print(s[keep].head(14).to_string(index=False))
