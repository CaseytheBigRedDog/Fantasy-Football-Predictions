# Fantasy Football Weekly Points Predictor — Starter Pipeline

This is a working, tested pipeline that predicts weekly fantasy points (PPR)
for QB/RB/WR/TE using 2010–2024 NFL data. Run it top to bottom on real data
and it produces a model that beats a naive baseline out of the box.

## Data
- `player_stats.csv` — weekly player stats + fantasy points, from nflverse
  (updates nightly during the season)
- `games.csv` — schedules, results, Vegas lines, weather, from nflverse

Both pulled directly from:
`https://github.com/nflverse/nflverse-data/releases`
(release tags: `player_stats`, `schedules`)

To refresh with current-season data later, just re-download these two files
from the same URLs — no scraping needed, they're public CSVs.

## Pipeline (run in order)
```
python3 01_load_data.py      # loads + filters raw data -> stats_clean.parquet, games_clean.parquet
python3 02_features.py       # builds leak-free rolling features -> model_data.parquet
python3 03_train_model.py    # trains Ridge / GBM / XGBoost, evaluates on 2024 holdout
```
Requires: `pandas`, `pyarrow`, `scikit-learn`, `xgboost`

## Results on 2024 (fully held-out season)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive (3-wk rolling avg) | 5.15 | 7.00 | — |
| Ridge regression | 4.94 | 6.51 | 0.34 |
| Gradient Boosting | 4.90 | 6.46 | 0.35 |
| XGBoost | 4.89 | 6.44 | 0.36 |

MAE by position: TE 3.87, RB 4.84, WR 5.01, QB 6.22 — QBs have the highest
scoring variance (long TD strikes) and are hardest to nail down.

## Design decisions worth knowing about
1. **No data leakage**: every rolling/season-to-date feature is `.shift(1)`'d
   before rolling, so a player's week-N row only uses data through week N-1.
   This is the single most common mistake in these models — skip it and
   you'll get suspiciously great backtest numbers that fall apart in-season.
2. **Time-based split**: train on 2010–2022, validate on 2023 (used for
   XGBoost early stopping), test on 2024. Never randomly shuffle weeks —
   you'd be letting the model "see the future."
3. **Vegas lines matter**: `team_implied_total` (derived from spread + total)
   is one of the more useful context features — it's the market's read on
   game script, which correlates with volume/opportunity.
4. **Volume beats points**: target share, carries, and targets are more
   predictive and more stable week-to-week than raw fantasy point history,
   because touchdowns are noisier than opportunity.

## Natural next steps
- **Snap counts / routes run** (also on nflverse, release `snap_counts`) —
  usage rate rather than raw counts often generalizes better across role changes.
- **Injury report data** (release `injuries`) — questionable/out designations
  for teammates change target distribution.
- **Position-specific models** — a QB model and a WR/RB/TE model have very
  different feature importance profiles; splitting them usually helps.
- **Quantile regression / prediction intervals** — instead of a single point
  estimate, predict a range (floor/ceiling), which is often more useful for
  start/sit decisions than a single number.
- **Compare against FantasyPros consensus projections** as an external
  benchmark — if you can't beat the wisdom of the crowd, blend with it
  rather than replace it.
