"""
availability.py

Turns weekly injury reports and depth charts into three things for each player:

  p_play   the probability he plays, learned from how often players with the same
           designation (and practice status) actually recorded stats in past seasons
  status   a short label such as "Out" or "Questionable (limited) - hamstring"
  depth    his depth-chart slot, such as "WR2"

Only OFFICIAL game-day designations (Out / Doubtful / Questionable) change a
projection. Teams publish those late in the week (Friday for Sunday games). Before
that, the report only shows practice status, and the history can't tell us how
reliable an early-week practice status is (it stores each week's FINAL report). So
practice-only flags are labelled "provisional" and don't change any numbers.
"""
import glob

import numpy as np
import pandas as pd

from injury_utils import SKILL_POSITIONS, norm_practice, norm_report

MIN_N = 50          # smallest group whose historical rate is trusted
HISTORY_SEASONS = 6
PRACTICE_LABEL = {"dnp": "DNP", "limited": "limited", "full": "full"}


def load_injuries():
    """All injury reports for QB/RB/WR/TE (regular season), with simple status categories."""
    files = sorted(glob.glob("injury_data/injuries_*.csv"))
    if not files:
        return pd.DataFrame()
    inj = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    need = {"season", "week", "gsis_id", "position", "report_status", "practice_status"}
    if not need.issubset(inj.columns):
        return pd.DataFrame()
    if "game_type" not in inj.columns:
        inj["game_type"] = "REG"
    inj = inj[(inj["game_type"] == "REG") & inj["position"].isin(SKILL_POSITIONS) & inj["gsis_id"].notna()].copy()
    inj["report"] = inj["report_status"].map(norm_report)
    inj["practice"] = inj["practice_status"].map(norm_practice)
    # "Out (Definitely Will Not Play)" appears as a practice status
    inj["practice_out"] = inj["practice_status"].astype(str).str.lower().str.contains("out")
    inj["injury"] = inj["report_primary_injury"] if "report_primary_injury" in inj.columns else np.nan
    return inj


def _slot(df):
    """A running week counter, so week 1 of a season follows the last week of the one before."""
    return (df["season"] - 2013) * 20 + df["week"]


def history_table(inj, stats, season, week, n_seasons=HISTORY_SEASONS):
    """
    Past player-weeks on an injury report for fantasy-relevant players (those who recorded
    stats within roughly the previous 5 weeks), and whether each went on to record stats.
    Only weeks BEFORE (season, week) are used.
    """
    st = stats[["player_id", "season", "week"]].drop_duplicates()
    st = st.assign(idx=_slot(st))
    h = inj[(inj["season"] >= season - n_seasons)
            & ((inj["season"] < season) | ((inj["season"] == season) & (inj["week"] < week)))].copy()
    if h.empty:
        return h.assign(played=pd.Series(dtype=bool))
    h["idx"] = _slot(h)
    prev = pd.merge_asof(
        h.sort_values("idx"),
        st.sort_values("idx")[["player_id", "idx"]].rename(columns={"idx": "prev_idx"}),
        left_on="idx", right_on="prev_idx", left_by="gsis_id", right_by="player_id",
        allow_exact_matches=False, direction="backward")
    prev = prev[(prev["idx"] - prev["prev_idx"]) <= 5].copy()
    keys = pd.MultiIndex.from_frame(st[["player_id", "season", "week"]])
    prev["played"] = pd.MultiIndex.from_frame(prev[["gsis_id", "season", "week"]]).isin(keys)
    return prev


def fit_play_rates(inj, stats, season, week):
    """How often players with each designation went on to record stats (past weeks only)."""
    h = history_table(inj, stats, season, week)
    if h.empty:
        empty = pd.DataFrame(columns=["report", "practice", "n", "p"])
        return {"rp": empty, "r": empty[["report", "n", "p"]]}
    rp = h.groupby(["report", "practice"])["played"].agg(n="size", p="mean").reset_index()
    r = h.groupby("report")["played"].agg(n="size", p="mean").reset_index()
    return {"rp": rp, "r": r}


def play_probability(report, practice, practice_out, rates):
    """Probability the player records stats. Only official designations move it off 1.0."""
    if practice_out or report == "out":
        return 0.0
    if report not in ("doubtful", "questionable"):
        return 1.0
    rp, r = rates["rp"], rates["r"]
    row = rp[(rp["report"] == report) & (rp["practice"] == practice)]
    if len(row) and row["n"].iloc[0] >= MIN_N:
        p = row["p"].iloc[0]
    else:
        row = r[r["report"] == report]
        if len(row) and row["n"].iloc[0] >= MIN_N:
            p = row["p"].iloc[0]
        else:
            p = 0.60 if report == "questionable" else 0.05     # fallback if history is thin
    return float(min(max(p, 0.01), 1.0))


def status_text(report, practice, practice_out, injury):
    if practice_out or report == "out":
        label = "Out"
    elif report == "doubtful":
        label = "Doubtful"
    elif report == "questionable":
        label = "Questionable" + (f" ({PRACTICE_LABEL[practice]})" if practice in PRACTICE_LABEL else "")
    elif practice in ("dnp", "limited"):
        label = f"Practice: {PRACTICE_LABEL[practice]}"      # provisional, no official designation yet
    else:
        return ""
    if isinstance(injury, str) and injury.strip():
        label += f" - {injury.strip().lower()}"
    return label


def week_flags(inj, season, week, rates):
    """One row per player on that week's report: p_play, status, and whether it is official."""
    cur = inj[(inj["season"] == season) & (inj["week"] == week)].drop_duplicates("gsis_id", keep="last")
    if cur.empty:
        return pd.DataFrame(columns=["gsis_id", "report", "p_play", "status", "official"])
    out = pd.DataFrame({
        "gsis_id": cur["gsis_id"].to_numpy(),
        "report": cur["report"].to_numpy(),
        "p_play": [play_probability(r, p, o, rates) for r, p, o in zip(cur["report"], cur["practice"], cur["practice_out"])],
        "status": [status_text(r, p, o, i) for r, p, o, i in zip(cur["report"], cur["practice"], cur["practice_out"], cur["injury"])],
    })
    out["official"] = out["report"].isin(["out", "doubtful", "questionable"]) | cur["practice_out"].to_numpy()
    return out


def load_depth_labels():
    """Depth-chart slot labels ("WR1", "RB2", ...) from the newest depth chart snapshot."""
    files = sorted(glob.glob("depth_data/depth_charts_*.csv"))
    if not files:
        return pd.DataFrame(columns=["gsis_id", "depth"])
    want = {"dt", "gsis_id", "pos_abb", "pos_rank"}
    d = pd.read_csv(files[-1], usecols=lambda c: c in want, low_memory=False)
    if not want.issubset(d.columns):
        return pd.DataFrame(columns=["gsis_id", "depth"])
    d["dt"] = pd.to_datetime(d["dt"], errors="coerce", utc=True)
    d = d[(d["dt"] == d["dt"].max()) & d["pos_abb"].isin(SKILL_POSITIONS) & d["gsis_id"].notna()]
    d["pos_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    d = d.dropna(subset=["pos_rank"])
    best = d.sort_values("pos_rank").drop_duplicates("gsis_id", keep="first")
    best["depth"] = best["pos_abb"] + best["pos_rank"].astype(int).astype(str)
    return best[["gsis_id", "depth"]]
