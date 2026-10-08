## SQL Analysis

In addition to the Python model, this project uses SQL (SQLite) to engineer features, evaluate the model, and run data-quality checks. The queries live in the [`sql/`](sql/) folder and are run against a local database built from the project's CSV files.

All results below cover **2026 Weeks 3-4** and **players who played** (players with no stat line are excluded; see Limitations). With only two weeks of data, treat small differences as noise.

### What the SQL covers

| File | Purpose | Techniques |
|---|---|---|
| `01_rolling_features.sql` | Rolling average of each player's prior 3 games (the naive baseline feature) | Window functions, CTE, `ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING` to avoid data leakage |
| `03_model_vs_naive_expected.sql` | Model vs. naive baseline MAE by week and position | Multi-table joins, `UNION ALL`, aggregation |
| `04_bias_and_rmse.sql` | Pooled bias and RMSE by position for `expected`, `median`, and `naive` | Conditional math in aggregates |
| `05_accuracy_by_status.sql` | Accuracy by practice-participation status | `COALESCE`, grouped comparison |
| `06_prediction_coverage.sql` | How many predicted players have a scored result | `LEFT JOIN`, conditional counts |
| `07_missing_actuals.sql` | Why results are missing; separates "did not play" from real zeros | Data-quality profiling |
| `08_expected_vs_if_active.sql` | Compares the model's two point predictions | Side-by-side metrics |
| `09_p_play_calibration.sql` | Checks whether `p_play` behaves like a probability | Binning with `CASE`, calibration |
| `10_play_rate_by_depth.sql` | Play rate by depth-chart slot | Grouping, min/max diagnostics |
| `11_accuracy_by_depth_tier.sql` | Accuracy by depth-chart tier | Nested CTEs, `CASE` bucketing |

### Key findings

**1. The model beats the 3-game-average baseline at every position (RMSE, pooled Weeks 3-4).**

| Position | n | Model RMSE | Naive RMSE |
|---|---|---|---|
| QB | 62 | 6.34 | 8.11 |
| RB | 149 | 6.71 | 7.02 |
| TE | 121 | 5.74 | 6.32 |
| WR | 234 | 6.70 | 7.48 |

The RB margin is slim, and by-week results vary (for example, Week 4 RBs improved only 1.6% on MAE while Week 4 QBs improved 25.3%).

**2. A mean-based point prediction reduces bias compared with the median.** Fantasy scoring is right-skewed, so the median under-predicts. Average bias (actual minus prediction) was 0.3 to 1.0 points for `expected` versus 1.4 to 2.3 points for `median` (QB: 1.00 vs. 1.43). The median has slightly lower MAE in some cases, which is expected because the median minimizes absolute error. `expected` is used as the point prediction.

**3. The model's advantage is concentrated among starters.**

| Depth tier | n | Model MAE | Naive MAE |
|---|---|---|---|
| Slot 1 (starters) | 233 | 5.88 | 6.68 |
| Slot 2 | 156 | 4.46 | 4.84 |
| Slot 3 | 91 | 4.21 | 4.47 |
| Slot 4+ | 79 | 3.47 | 3.45 |

Starters are under-predicted by about 1.1 points on average, which is worth watching as more weeks come in.

**4. Players on a limited practice report are harder to predict.** MAE was 5.97 for "Practice: limited" players versus 4.77 for players with no practice flag, though the model still beat the baseline in both groups.

### Limitations

- **Small sample.** Two weeks of results. Position and tier differences, especially the smaller groups, may not hold up.
- **Only players who played are scored.** 209 of 775 predicted player-weeks (27%) have no stat line, and most of them are depth-chart backups. Results describe accuracy for players who played, not for everyone the model listed.
- **Selection effects in injury groups.** Only 22 of 60 "Practice: DNP" players have a scored result, so that group's accuracy is not reliable.
- **`p_play` does not yet work as a probability.** It is 1.0 for almost every player (774 of 775), while only about 73% of those players have a stat line. The model predicts points for players who play, but it does not yet predict who plays.
- **Some depth-chart data is missing.** Some players have an unknown depth slot.
- **Practice status is not a game-day designation.** The `status` column holds practice-participation labels, not official Questionable/Doubtful/Out designations.

### Next steps

- Build a real play probability using injury reports and depth-chart slot, and test it with the calibration query (`09_p_play_calibration.sql`).
- Fill in missing depth-chart data.
- Re-run all queries as more weeks accumulate, and report results pooled across weeks.
- Test whether the first-read-rate and red-zone features reduce the starter under-prediction.

### Reproducing the analysis

```
pip install pandas
mkdir data
Copy-Item *.csv data\
python load_to_sql.py
python run_query.py sql/04_bias_and_rmse.sql
```

`load_to_sql.py` builds a local `fantasy.db` from the CSV files in the `data/` folder. Both `fantasy.db` and `data/` are excluded from the repository via `.gitignore`, so you need to generate them locally.


