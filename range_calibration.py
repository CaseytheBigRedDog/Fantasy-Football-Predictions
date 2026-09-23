"""
range_calibration.py

Shared pieces for the weekly predictions.

Two ways to build a player's FLOOR / MEDIAN / CEILING (10th / 50th / 90th percentile):

  "quantile"  Separate models trained directly to predict each percentile.
              (Fine for most players, but for the very top projections, where
              there is little data, the ceilings ran too low in backtests.)

  "residual"  Built around the well-calibrated "expected" (average) model.
              For a player projected at, say, 19 points, look at how far real
              results landed from the projection for OTHER players who were
              projected near 19 in past seasons (using only predictions made
              without seeing the result), and apply that spread.

  "hybrid"    The "quantile" ranges for almost everyone, but the "residual" ranges for
              the very top projections (those at or above the 97th percentile of
              past projections at that position), where the direct models have the
              least data.

RANGE_METHOD picks which one predict_week.py uses. It starts as "quantile" and is
switched by adopt_residual_ranges.py, only after compare_range_methods.py shows
the alternative is better.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from new_features import extra_feature_cols

RANGE_METHOD = "residual"

POSITIONS = ["QB", "RB", "WR", "TE"]
QUANTILES = (0.1, 0.5, 0.9)        # floor, median, ceiling
FIRST_CALIBRATION_SEASON = 2018    # earliest season used to learn typical spreads
NEIGHBORS = 300                    # how many similarly projected past players to compare with
HYBRID_CUTOFF = 0.97               # "hybrid" uses residual ranges at/above this percentile of past projections


def get_feature_cols(df):
    """The features the position models use (same recipe as 03b_position_quantile_models.py)."""
    cols = [c for c in df.columns if c.endswith(("_r3", "_r5", "_seasontd", "_trend"))]
    cols += [
        "games_played_prior", "team_implied_total", "spread_line", "total_line",
        "is_home", "rest_days", "def_pts_allowed_r5",
    ]
    cols += extra_feature_cols(df)
    return list(dict.fromkeys(c for c in cols if c in df.columns))


def fit_mean_model(X, y):
    """Predicts the AVERAGE outcome (what most sites call a projection)."""
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250, learning_rate=0.04, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42,
    )
    model.fit(X, y)
    return model


def oos_expected(train, feature_cols, first_season, last_season):
    """
    Out-of-sample 'expected' predictions for past seasons: for each season, train on the
    seasons BEFORE it and predict it. Returns one row per player-week with the
    prediction and what actually happened.
    """
    frames = []
    for season in range(first_season, last_season + 1):
        for pos in POSITIONS:
            tr = train[(train["position"] == pos) & (train["season"] < season)]
            te = train[(train["position"] == pos) & (train["season"] == season)]
            if te.empty or len(tr) < 500:
                continue
            model = fit_mean_model(tr[feature_cols].fillna(0), tr["target_fp"])
            frames.append(pd.DataFrame({
                "season": season, "pos": pos,
                "expected": np.maximum(model.predict(te[feature_cols].fillna(0)), 0),
                "actual": te["target_fp"].values,
            }))
    if not frames:
        return pd.DataFrame(columns=["season", "pos", "expected", "actual"])
    return pd.concat(frames, ignore_index=True)


def residual_range(expected, pos, pool, k=NEIGHBORS):
    """
    Floor / median / ceiling for players at one position, given their 'expected' values.
    `pool` is the out-of-sample history from oos_expected(). For each player we take the
    k past players (same position) whose projection was closest, look at how far their
    real results landed from their projections, and apply those percentiles.
    """
    p = pool[pool["pos"] == pos].sort_values("expected")
    exp_sorted = p["expected"].to_numpy()
    resid = (p["actual"] - p["expected"]).to_numpy()
    n = len(p)
    if n == 0:
        raise ValueError(f"No calibration history for {pos}.")
    k = min(k, n)
    out = np.empty((len(expected), 3))
    for i, e in enumerate(np.asarray(expected, dtype=float)):
        idx = np.searchsorted(exp_sorted, e)
        lo = min(max(idx - k // 2, 0), n - k)
        out[i] = e + np.quantile(resid[lo:lo + k], QUANTILES)
    return np.clip(out, 0, None)   # fantasy points below 0 are rare; show them as 0


def hybrid_range(expected, pos, pool, quantile_rng):
    """Residual ranges for the top projections, the direct-quantile ranges for everyone else."""
    expected = np.asarray(expected, dtype=float)
    resid_rng = residual_range(expected, pos, pool)
    cutoff = pool.loc[pool["pos"] == pos, "expected"].quantile(HYBRID_CUTOFF)
    use_residual = expected >= cutoff
    return np.where(use_residual[:, None], resid_rng, quantile_rng)
