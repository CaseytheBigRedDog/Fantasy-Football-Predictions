# Fantasy Football Predictions

A Python pipeline that predicts weekly PPR fantasy football scores for QBs, RBs, WRs and TEs. For every player on a team with an upcoming game it produces a **floor / median / ceiling** (10th / 50th / 90th percentile) instead of a single number, then grades those predictions against what actually happened.

## Results

Trained on 2013-2023, tuned on 2024, tested on the full **2025 season** (about 5,200 player-weeks). Lower error is better.

| Model | MAE (points) | RMSE | R² |
|---|---|---|---|
| Naive baseline (player's last 3 games) | 5.207 | 7.095 | - |
| Ridge regression | 4.847 | 6.416 | 0.344 |
| Gradient boosting | 4.836 | 6.403 | 0.347 |
| **XGBoost** | **4.807** | **6.376** | **0.352** |

Position-specific quantile models (overall MAE 4.661):

| Position | MAE | Actual results inside predicted 10-90 range (target ~80%) |
|---|---|---|
| QB | 6.538 | 76.3% |
| RB | 4.732 | 82.0% |
| WR | 4.555 | 81.8% |
| TE | 3.677 | 81.7% |

**Versus FantasyPros expert consensus** (Spearman rank correlation with actual results, 2025, 4,104 matched player-weeks; higher is better):

| Position | This model | FantasyPros |
|---|---|---|
| QB | 0.399 | 0.475 |
| RB | 0.653 | 0.694 |
| WR | 0.564 | 0.598 |
| TE | 0.542 | 0.568 |

**In short:** the model misses by about 4.7 points per player-week, roughly 8% better than guessing each player's recent average, and its ranges are reasonably well calibrated. It does **not** beat FantasyPros, which is ahead at every position. One difference to keep in mind: FantasyPros rankings are scraped on Fridays and reflect injury news, while this model has no injury information at all.

## How it works

1. **Data** (all free): weekly player stats, schedules with Vegas lines, player ID crosswalk and snap counts from [nflverse](https://github.com/nflverse/nflverse-data); FantasyPros weekly expert rankings via [dynastyprocess](https://github.com/dynastyprocess/data).
2. **Features** use only information available *before* kickoff: rolling 3- and 5-game averages of opportunity and production (targets, carries, yards, target share, EPA, PPR points), career-to-date averages, snap share and its trend, opponent points allowed to the position, Vegas spread and total (and the team's implied score), home/away and rest days.
3. **Models** are evaluated with time-ordered splits (never training on the future). XGBoost is the pooled model; separate per-position XGBoost quantile models produce the floor / median / ceiling.
4. **Prediction** builds a row for each likely player on a team with an unplayed game, using exactly the same feature code as the training rows.

## Weekly routine

Run on Tuesday, after Monday night's game is final:

```
python refresh_data.py        # download new data, rebuild features, retrain, re-evaluate
python score_predictions.py   # grade last week's predictions
python predict_week.py        # project the coming week
```

Outputs: `predictions_<season>_week<N>.csv` (floor / median / ceiling per player), `scored_<season>_week<N>.csv` (predictions next to actual results) and `accuracy_log.csv` (one row per week and position, including the comparison with FantasyPros when the archive has that week). Committing predictions before the games are played makes the history in this repository a real track record; live tracking starts with Week 3 of the 2026 season.

`run_refresh.bat` and `SCHEDULING.md` explain how to schedule the refresh with Windows Task Scheduler.

## Setup

```
pip install -r requirements.txt
```

Needs Python with pandas, numpy, scikit-learn, xgboost (2.0 or newer, for quantile regression), scipy and pyarrow.

## Files

| File | Purpose |
|---|---|
| `refresh_data.py` | Downloads all raw data, then runs scripts 01-04 |
| `download_player_stats.py` | Builds `player_stats.csv` from nflverse's per-season weekly files |
| `01_load_data.py` | Cleans stats and schedules (regular season, 2013+, QB/RB/WR/TE) |
| `02_features.py` | Builds training features and the upcoming-week rows |
| `03_train_model.py` | Baseline, Ridge, gradient boosting and XGBoost; time-ordered evaluation |
| `03b_position_quantile_models.py` | Per-position floor / median / ceiling models |
| `04_fantasypros_comparison.py` | Compares rankings with FantasyPros |
| `predict_week.py` | Projects the next unplayed week |
| `score_predictions.py` | Grades a week's predictions and updates the accuracy log |
| `check_fp_alignment.py` | Diagnostic that verifies FantasyPros rankings are matched to the right week |
| `update_split.py`, `apply_fp_fix.py` | One-time maintenance scripts (already applied) |

## Known limitations

- **No injury, inactive or depth-chart information.** A player who got hurt still gets a projection based on his recent form. Check injury reports before acting on any number.
- **Off-season team changes are not handled** for upcoming-week rows; a player's team is taken from his most recent game.
- **Rookies and players with few games** get rougher ranges (the prediction script flags them).
- **PPR scoring only**, and one test season so far; small differences between models can change with another season of data.
- The FantasyPros archive only starts in September 2023, so the comparison currently covers the 2025 test season only. Its week labels are shifted back one week to match the Friday scrape schedule (see `check_fp_alignment.py`).

## Data notes

nflverse froze its combined `player_stats` file after the 2024 season and now publishes one weekly file per season, which `download_player_stats.py` reads and converts to the older column layout the rest of the pipeline expects. Snap counts come from Pro Football Reference via nflverse.

## Disclaimer

For learning and entertainment. Not betting or financial advice.
