"""
projections.py - loads YOUR model's weekly projections and looks players up.

This file has no web code in it. It only reads the newest predictions_<season>_week<N>.csv
that predict_week.py writes into the project folder, so everything the API and the assistant
say comes from your own numbers.
"""
import difflib
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

# The project folder is the parent of this api/ folder (override with the DATA_DIR setting).
ROOT = Path(os.environ.get("DATA_DIR") or Path(__file__).resolve().parent.parent)
FILE_PATTERN = re.compile(r"^predictions_(\d{4})_week(\d+)\.csv$")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

_cache = {"path": None, "mtime": None, "df": None, "season": None, "week": None}


def normalize(text):
    """Lowercase, drop accents and punctuation: "Ja'Marr Chase" -> "jamarr chase"."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = re.sub(r"['’.]", "", text.lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def last_name(name_key):
    parts = [p for p in name_key.split() if p not in SUFFIXES]
    return parts[-1] if parts else ""


def newest_file():
    """Return (path, season, week) of the most recent predictions file, or None."""
    best = None
    for path in ROOT.glob("predictions_*_week*.csv"):
        match = FILE_PATTERN.match(path.name)  # skips "..._baseline.csv" and similar
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            if best is None or key > best[0]:
                best = (key, path)
    if best is None:
        return None
    return best[1], best[0][0], best[0][1]


def load():
    """Return the projections DataFrame, reloading only if the file changed."""
    found = newest_file()
    if found is None:
        raise FileNotFoundError(
            f"No predictions_<season>_week<N>.csv found in {ROOT}. "
            "Run predict_week.py first, or set DATA_DIR to your project folder."
        )
    path, season, week = found
    mtime = path.stat().st_mtime
    if _cache["path"] == path and _cache["mtime"] == mtime:
        return _cache["df"]

    df = pd.read_csv(path)
    df["status"] = df["status"].fillna("").astype(str)
    df["depth"] = df["depth"].fillna("").astype(str)
    df["name_key"] = df["player"].map(normalize)
    df["last_key"] = df["name_key"].map(last_name)
    _cache.update(path=path, mtime=mtime, df=df, season=season, week=week)
    return df


def meta():
    df = load()
    return {
        "season": _cache["season"],
        "week": _cache["week"],
        "file": _cache["path"].name,
        "players": int(len(df)),
        "positions": sorted(df["position"].unique().tolist()),
        "teams": sorted(df["team"].unique().tolist()),
    }


def to_records(df):
    """DataFrame -> list of plain dicts (NaN becomes null), without our helper columns."""
    cols = [c for c in df.columns if c not in ("name_key", "last_key")]
    return json.loads(df[cols].to_json(orient="records"))


def top(position=None, team=None, n=10):
    df = load()
    if position:
        df = df[df["position"] == position.upper()]
    if team:
        df = df[df["team"] == team.upper()]
    return to_records(df.sort_values("expected", ascending=False).head(n))


def search(q=None, position=None, team=None, limit=500):
    df = load()
    if position:
        df = df[df["position"] == position.upper()]
    if team:
        df = df[df["team"] == team.upper()]
    if q:
        key = normalize(q)
        df = df[df["name_key"].str.contains(key, regex=False)]
    return to_records(df.sort_values("expected", ascending=False).head(limit))


def get_by_id(player_id):
    df = load()
    rows = df[df["player_id"] == player_id]
    return to_records(rows)[0] if len(rows) else None


def lookup(name, max_candidates=5):
    """
    Find players matching a typed name ("Gibbs", "jamarr chase", "Kenneth Walker").
    Returns a list of matching rows, best first. One row = confident match;
    several rows = ambiguous (e.g. two Smiths); empty = not found.
    """
    df = load()
    key = normalize(name)
    if not key:
        return []

    exact = df[df["name_key"] == key]
    if len(exact):
        return to_records(exact.head(max_candidates))

    # Every word typed must appear in the player's name ("walker" or "kenneth walker").
    words = [w for w in key.split() if w not in SUFFIXES]
    if words:
        mask = df["name_key"].map(lambda k: all(w in k.split() for w in words))
        hits = df[mask]
        if len(hits):
            return to_records(hits.sort_values("expected", ascending=False).head(max_candidates))

    # Last resort: spelling mistakes ("Jamar Chase", "Bijan Robinsen").
    close = difflib.get_close_matches(key, df["name_key"].tolist(), n=max_candidates, cutoff=0.75)
    if close:
        hits = df[df["name_key"].isin(close)]
        return to_records(hits.sort_values("expected", ascending=False).head(max_candidates))
    return []


def find_players_in_text(text):
    """
    Find players mentioned inside a sentence ("should I start Gibbs or Chase?").
    Returns (matches, ambiguous) where matches is a list of rows and ambiguous maps a typed
    last name to several possible players (so the assistant can ask which one you meant).
    """
    df = load()
    padded = f" {normalize(text)} "
    matches, ambiguous, used_last = [], {}, set()

    # Pass 1: full names ("ja'marr chase" / "kenneth walker iii" without the suffix).
    for row in df.itertuples():
        bare = " ".join(p for p in row.name_key.split() if p not in SUFFIXES)
        if f" {row.name_key} " in padded or f" {bare} " in padded:
            matches.append(row.player_id)
            used_last.add(row.last_key)

    # Pass 2: last names only ("Gibbs"), when we have not already matched that surname.
    words = set(padded.split())
    by_last = df.groupby("last_key")
    for last, group in by_last:
        if len(last) < 3 or last not in words or last in used_last:
            continue
        if len(group) == 1:
            matches.append(group.iloc[0]["player_id"])
        else:
            ambiguous[last] = to_records(group.sort_values("expected", ascending=False).head(5))

    ids = list(dict.fromkeys(matches))
    rows = to_records(df[df["player_id"].isin(ids)])
    order = {pid: i for i, pid in enumerate(ids)}
    rows.sort(key=lambda r: order[r["player_id"]])
    return rows, ambiguous
