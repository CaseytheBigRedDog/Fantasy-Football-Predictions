"""
new_features.py

One place that says whether the models use the newer features made in
02_features.py (recency-weighted averages, current-season averages and
role-change signals).

The switch below starts OFF, so nothing about your models changes until
adopt_new_features.py turns it on (only do that after compare_features.py
shows the new features help).
"""

USE_NEW_FEATURES = True


def new_feature_cols(df):
    """Every new feature column that exists in df (used to TEST them)."""
    cols = [c for c in df.columns if c.endswith(("_ewm", "_szn", "_change"))]
    cols += [c for c in ["season_games_prior", "pts_vs_career"] if c in df.columns]
    return list(dict.fromkeys(cols))


def extra_feature_cols(df):
    """The new features the models should use right now (none while switched off)."""
    return new_feature_cols(df) if USE_NEW_FEATURES else []
