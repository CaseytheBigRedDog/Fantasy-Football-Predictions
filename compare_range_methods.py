"""
compare_range_methods.py

Compares three ways of building the floor / median / ceiling ranges:

  CURRENT   direct percentile models (RANGE_METHOD = "quantile")
  RESIDUAL  built around the "expected" model for everyone
  HYBRID    residual ranges for the top ~3% of projections, CURRENT for the rest

Each test season is scored fairly: the residual method learns typical spreads only
from seasons BEFORE it (each past season predicted by a model that never saw it).

Usage:
    python compare_range_methods.py                  # the last full season
    python compare_range_methods.py 2023 2024 2025   # several seasons, pooled (recommended)

For a well-calibrated range, of all results: ~10% land below the floor, ~50% below the
median and ~10% above the ceiling. Run time: a few minutes, longer for more seasons.
Nothing changes in your predictions until you run adopt_residual_ranges.py.
"""
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import range_calibration as rc

train = pd.read_parquet("model_data.parquet")
feature_cols = rc.get_feature_cols(train)
latest_full = int(train["season"].max()) - 1
seasons = sorted(int(a) for a in sys.argv[1:]) or [latest_full]
if min(seasons) < rc.FIRST_CALIBRATION_SEASON + 2:
    raise SystemExit(f"Test seasons must be {rc.FIRST_CALIBRATION_SEASON + 2} or later "
                     f"(the residual method needs a few earlier seasons to learn from).")
print(f"Test season(s): {', '.join(map(str, seasons))}. Spreads are learned only from earlier seasons "
      f"(from {rc.FIRST_CALIBRATION_SEASON}).\n")

print("Building out-of-sample projections for past seasons (each season predicted by a model "
      "that never saw it) ...")
pool_all = rc.oos_expected(train, feature_cols, rc.FIRST_CALIBRATION_SEASON, max(seasons))


def fit_quantile_model(X, y, alpha):
    model = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=250, learning_rate=0.04, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42,
    )
    model.fit(X, y)
    return model


frames = []
for s in seasons:
    print(f"Scoring {s} ...")
    pool_cal = pool_all[pool_all["season"] < s]
    hold = pool_all[pool_all["season"] == s]
    for pos in rc.POSITIONS:
        te = train[(train["position"] == pos) & (train["season"] == s)]
        tr = train[(train["position"] == pos) & (train["season"] < s)]
        h = hold[hold["pos"] == pos]
        if te.empty or len(h) != len(te):
            continue
        Xte = te[feature_cols].fillna(0)
        q = np.clip(np.sort(np.column_stack([
            fit_quantile_model(tr[feature_cols].fillna(0), tr["target_fp"], a).predict(Xte)
            for a in rc.QUANTILES]), axis=1), 0, None)
        exp = h["expected"].to_numpy()
        r = rc.residual_range(exp, pos, pool_cal)
        hy = rc.hybrid_range(exp, pos, pool_cal, q)
        frames.append(pd.DataFrame({
            "season": s, "pos": pos, "actual": h["actual"].to_numpy(), "expected": exp,
            "cf": q[:, 0], "cm": q[:, 1], "cc": q[:, 2],       # current
            "rf": r[:, 0], "rm": r[:, 1], "rc": r[:, 2],       # residual
            "hf": hy[:, 0], "hm": hy[:, 1], "hc": hy[:, 2],    # hybrid
        }))
bt = pd.concat(frames, ignore_index=True)
bt["pct"] = bt.groupby(["season", "pos"])["expected"].rank(pct=True)
print()


def rates(g, p):
    a = g["actual"]
    return (a < g[p + "f"]).mean(), (a <= g[p + "m"]).mean(), (a > g[p + "c"]).mean()


def dev(g, p):
    f, m, c = rates(g, p)
    return np.mean([abs(f - .10), abs(m - .50), abs(c - .10)])


def pinball(g, p):
    total = 0.0
    for col, q in [("f", .1), ("m", .5), ("c", .9)]:
        d = g["actual"] - g[p + col]
        total += np.maximum(q * d, (q - 1) * d).mean()
    return total / 3


def row(label, g, n_label=None):
    c, r, h = rates(g, "c"), rates(g, "r"), rates(g, "h")
    print(f"{label:<15}{len(g):>6,}  {c[0]:>5.0%}{c[1]:>6.0%}{c[2]:>6.0%}   "
          f"{r[0]:>5.0%}{r[1]:>6.0%}{r[2]:>6.0%}   {h[0]:>5.0%}{h[1]:>6.0%}{h[2]:>6.0%}")


print("Share of results below the floor / below the median / above the ceiling (targets: 10% / 50% / 10%)\n")
print(f"{'':<21}{'CURRENT':^18}{'RESIDUAL':^18}{'HYBRID':^18}")
print(f"{'Tier':<15}{'N':>6}  {'floor med ceil':>17}  {'floor med ceil':>17}  {'floor med ceil':>17}")
for lo, hi, label in [(0, .5, "bottom half"), (.5, .8, "50th-80th pct"), (.8, .9, "80th-90th pct"),
                      (.9, .97, "90th-97th pct"), (.97, 1.01, "top 3%")]:
    g = bt[(bt["pct"] > lo) & (bt["pct"] <= hi)]
    if len(g) >= 30:
        row(label, g)
row("All players", bt)

if len(seasons) > 1:
    print("\nTop 3% of projections, season by season (does the pattern repeat?)")
    for s in seasons:
        g = bt[(bt["season"] == s) & (bt["pct"] > .97)]
        if len(g) >= 20:
            row(str(s), g)

print(f"\n{'':<34}{'Current':>10}{'Residual':>10}{'Hybrid':>10}")
w = lambda p: (bt[p + 'c'] - bt[p + 'f']).mean()
ins = lambda p: ((bt['actual'] >= bt[p + 'f']) & (bt['actual'] <= bt[p + 'c'])).mean()
mae = lambda p: (bt['actual'] - bt[p + 'm']).abs().mean()
print(f"{'Average width, floor to ceiling':<34}{w('c'):>10.1f}{w('r'):>10.1f}{w('h'):>10.1f}")
print(f"{'Results inside the range':<34}{ins('c'):>10.1%}{ins('r'):>10.1%}{ins('h'):>10.1%}")
print(f"{'Median: average error (MAE)':<34}{mae('c'):>10.3f}{mae('r'):>10.3f}{mae('h'):>10.3f}")
pc, pr, ph = pinball(bt, "c"), pinball(bt, "r"), pinball(bt, "h")
print(f"{'Overall range score (lower=better)':<34}{pc:>10.3f}{pr:>10.3f}{ph:>10.3f}")

# ---------------- verdict on the hybrid ----------------
top, rest = bt[bt["pct"] > 0.97], bt[bt["pct"] <= 0.97]
print("\nMiss from the targets (average of the three rates; 0 is perfect)")
print(f"  Top 3% of projections: current {dev(top, 'c'):.3f}  residual {dev(top, 'r'):.3f}  hybrid {dev(top, 'h'):.3f}")
print(f"  Everyone else:         current {dev(rest, 'c'):.3f}  residual {dev(rest, 'r'):.3f}  hybrid {dev(rest, 'h'):.3f}")

better_top = dev(top, "h") <= dev(top, "c") - 0.04
hurt_rest = dev(rest, "h") > dev(rest, "c") + 0.01
worse_score = ph > pc * 1.003
wins = sum(dev(bt[(bt["season"] == s) & (bt["pct"] > .97)], "h")
           < dev(bt[(bt["season"] == s) & (bt["pct"] > .97)], "c") for s in seasons)
consistent = wins >= (len(seasons) + 1) // 2 + (1 if len(seasons) >= 3 else 0)

print("\nVerdict on the HYBRID")
if better_top and not hurt_rest and not worse_score and (len(seasons) == 1 or consistent):
    extra = "" if len(seasons) > 1 else " (this is one season; running several is more convincing)"
    print(f"  It fixes the top-3% ranges without hurting the rest, in {wins} of {len(seasons)} "
          f"season(s).{extra}\n  Adoption looks worthwhile: run  python adopt_residual_ranges.py hybrid")
elif better_top and len(seasons) > 1 and not consistent:
    print(f"  It helped the top 3% overall but in only {wins} of {len(seasons)} seasons, so the "
          f"pattern is not reliable. Keep the current method.")
elif better_top:
    print("  It improves the top 3% but there is a cost elsewhere (worse range score or "
          "worse calibration for everyone else). Paste me this output and we'll decide.")
else:
    print("  No clear gain for the top 3%. Keep the current method.")

all_c, all_r = dev(bt, "c"), dev(bt, "r")
if all_r <= all_c - 0.02 and pr <= pc * 0.99:
    print("\nVerdict on the RESIDUAL method for everyone")
    print("  It is clearly better across the board (closer to the targets and a better overall "
          "range score).\n  Consider: python adopt_residual_ranges.py residual")
print("  (The live weekly scorecard is the real judge.)")
