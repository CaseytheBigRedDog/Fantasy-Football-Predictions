# Fantasy Football Predictions

A Python pipeline that predicts weekly PPR fantasy football scores for QBs, RBs, WRs and TEs. For every player on a team with an upcoming game it produces:

- **Expected**: the average outcome (the same kind of number ESPN and other sites show)
- **Floor / median / ceiling**: the 10th, 50th and 90th percentile outcomes (a realistic bad game, the typical game, and a great game)

It then grades those predictions against what actually happened, week by week.

## Results

Trained on 2013-2023, tuned on 2024, tested on the full **2025 season** (about 5,200 player-weeks). Lower error is better.

| Model | Average error (points) | RMSE | R² |
|---|---|---|---|
| Naive baseline (player's last 3 games) | 5.207 | 7.095 | - |
| Ridge regression | 4.807 | 6.373 | 0.353 |
| Gradient boosting | 4.795 | 6.369 | 0.354 |
| **XGBoost** | **4.767** | **6.346** | **0.359** |

Position-specific models (overall average error 4.624, about 11% better than the naive baseline):

| Position | Average error | Actual results inside the predicted 10-90 range (target ~80%) |
|---|---|---|
| QB | 6.424 | 76.0% |
| RB | 4.704 | 82.3% |
| WR | 4.535 | 81.9% |
| TE | 3.642 | 81.4% |

**Versus FantasyPros expert consensus** (Spearman rank correlation with actual results, 2025, 4,104 matched player-weeks; higher is better):

| Position | This model | FantasyPros |
|---|---|---|
| QB | 0.412 | 0.475 |
| RB | 0.656 | 0.694 |
| WR | 0.568 | 0.598 |
| TE | 0.553 | 0.568 |

**In short:** a typical prediction misses by about 4.6 points per player-week, and the ranges are reasonably well calibrated. The model does **not** beat FantasyPros, which is ahead at every position, though the gap has narrowed as the model improved. One reason: FantasyPros rankings are scraped on Fridays and reflect injury news, while this model has no injury information at all.

**The expected column is well calibrated across projection sizes.** In the 2025 backtest, actual results landed within about a point of the expected value (at most 1.2 points off) in every tier from the bottom half of projections up to the top 3% (top 3%: 19.3 expected vs 19.5 actual).

## How it works

1. **Data** (all free): weekly player stats, schedules with Vegas lines, player ID crosswalk and snap counts from [nflverse](https://github.com/nflverse/nflverse-data); FantasyPros weekly expert rankings via [dynastyprocess](https://github.com/dynastyprocess/data).
2. **Features** use only information available *before* kickoff, computed separately for each player:
   - rolling 3- and 5-game averages of production and usage (targets, carries, yards, target share, air yards share, EPA, PPR points) and career-to-date averages
   - **recency-weighted averages** (recent games count more) and **current-season averages**, plus signals for how far recent form has moved from the career norm
   - snap share and its trend
   - opponent points allowed to the position
   - Vegas spread and total (and the team's implied score), home/away and rest days
3. **Models** are evaluated with time-ordered splits (never training on the future). An XGBoost model predicts the expected (average) score; per-position models produce the floor / median / ceiling.
4. **Ranges** are built one of three ways, chosen by `RANGE_METHOD` in `range_calibration.py`:
   - `quantile` (default): separate models trained directly for each percentile
   - `residual`: built around the expected model, using how far past results landed from similar past projections
   - `hybrid`: `quantile` for almost everyone, `residual` for the top ~3% of projections, where the direct models have the least data. `compare_range_methods.py` tests these against each other over several seasons; the top-3% ceilings ran low with the default method in the 2025 backtest (about 20% of those players beat their ceiling, versus a 10% target), and the residual method fixed that in the single-season test but was slightly worse just below it, so it is only adopted if a multi-season check confirms the gain
5. **Prediction** builds a row for each likely player on a team with an unplayed game, using exactly the same feature code as the training rows.

## Weekly routine

Run on Tuesday, after Monday night's game is final:

```
python refresh_data.py        # download new data, rebuild features, retrain, re-evaluate
python score_predictions.py   # grade last week's predictions
python predict_week.py        # project the coming week
```

Outputs: `predictions_<season>_week<N>.csv` (floor / median / ceiling / expected per player), `scored_<season>_week<N>.csv` (predictions next to actual results) and `accuracy_log.csv` (one row per week and position, including the comparison with FantasyPros when the archive has that week). Committing predictions before the games are played makes the history in this repository a real track record; live tracking starts with Week 3 of the 2026 season.

`run_refresh.bat` and `SCHEDULING.md` explain how to schedule the refresh with Windows Task Scheduler.

## Setup

```
pip install -r requirements.txt
```

Needs Python with pandas, numpy, scikit-learn, xgboost (2.0 or newer, for quantile regression), scipy and pyarrow.

## Files

**Core pipeline**

| File | Purpose |
|---|---|
| `refresh_data.py` | Downloads all raw data, then runs scripts 01-04 |
| `download_player_stats.py` | Builds `player_stats.csv` from nflverse's per-season weekly files |
| `01_load_data.py` | Cleans stats and schedules (regular season, 2013+, QB/RB/WR/TE) |
| `02_features.py` | Builds training features and the upcoming-week rows |
| `03_train_model.py` | Baseline, Ridge, gradient boosting and XGBoost; time-ordered evaluation |
| `03b_position_quantile_models.py` | Per-position floor / median / ceiling models |
| `04_fantasypros_comparison.py` | Compares rankings with FantasyPros |

**Weekly tools**

| File | Purpose |
|---|---|
| `predict_week.py` | Projects the next unplayed week; prints a backtest and a calibration table |
| `score_predictions.py` | Grades a week's predictions and updates the accuracy log |
| `range_calibration.py` | Shared code for the expected model and the range methods |
| `new_features.py` | Switch for the recency-weighted and current-season features (currently on) |

**Analysis and diagnostics** (read-only; they explain the model, they don't change it)

| File | Purpose |
|---|---|
| `check_usage_bias.py` | Does the model under- or over-rate players by usage level? |
| `check_breakouts.py`, `check_role_breakouts.py` | How does the model treat hot streaks and role changes? |
| `find_comps.py` | For any player, shows what similar past players scored next |
| `check_fp_alignment.py` | Verifies FantasyPros rankings are matched to the right week |
| `compare_features.py` | Tests the new features against the old ones |
| `compare_range_methods.py` | Tests the `residual` and `hybrid` range methods against the default `quantile` method (`python compare_range_methods.py 2023 2024 2025`) |

**One-time maintenance scripts**

| File | Purpose |
|---|---|
| `update_split.py`, `apply_fp_fix.py`, `adopt_new_features.py` | Already applied |
| `adopt_residual_ranges.py` | Switches the range method (`hybrid`, `residual`, or back to `quantile`); run only if `compare_range_methods.py` recommends it |

## What the diagnostics showed

- **Hot streaks cool off, and the model knows it.** Veterans whose last 5 games were at least 1.5x their career average scored only about 10 points in their next game, and the model's predictions for them were as accurate as for everyone else.
- **Heavily used receivers are not under-rated on average.** Receivers with a 25%+ target share and 60%+ snap share were predicted at 14.8 points and scored 14.9.
- **Medians are lower than averages.** Fantasy scores are lopsided, so the median (the typical game) runs 1 to 2 points below the average for most players, and more for stars. Compare `expected`, not `median`, with ESPN-style projections.
- **FantasyPros weeks needed a one-week shift.** Their weekly scrapes happen on Fridays, after that week's Thursday game, so the week labels were one week late. This is corrected in `04_fantasypros_comparison.py`.
- **The recency-weighted and current-season features helped** at every position (average error 4.807 to 4.767 for the main model).

## Known limitations

- **No injury, inactive or depth-chart information.** A player who got hurt still gets a projection based on his recent form. Check injury reports before acting on any number.
- **Ranges for the very top projections are the least reliable.** With the default method, about 20% of the top 3% of players beat their ceiling in the 2025 backtest (the target is 10%). Treat `expected` as the main number for stars.
- **Off-season team changes are not handled** for upcoming-week rows; a player's team is taken from his most recent game.
- **Playoff games are not included** in a player's history.
- **Rookies and players with few games** get rougher ranges (the prediction script flags them).
- **PPR scoring only**, and one test season so far; small differences can change with another season of data.
- The FantasyPros archive only starts in September 2023, so the comparison currently covers the 2025 test season only.

## Data notes

nflverse froze its combined `player_stats` file after the 2024 season and now publishes one weekly file per season, which `download_player_stats.py` reads and converts to the older column layout the rest of the pipeline expects. Snap counts come from Pro Football Reference via nflverse.

## Disclaimer

For learning and entertainment. Not betting or financial advice.
