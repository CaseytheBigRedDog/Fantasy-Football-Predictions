# Fantasy Football Weekly Points Predictor

Predicts weekly fantasy points (PPR) for QB/RB/WR/TE using 2013–2024 NFL data,
with position-specific models, prediction intervals (floor/median/ceiling),
and a backtest against FantasyPros expert consensus rankings.

## Data sources (all free, all pulled programmatically)
| File | Contents | Source |
|---|---|---|
| `player_stats.csv` | Weekly player stats + fantasy points | nflverse, release `player_stats` |
| `games.csv` | Schedules, results, Vegas lines, weather | nflverse, release `schedules` |
| `snap_data/snap_counts_*.csv` | Offensive snap counts/share per player-week | nflverse, release `snap_counts` (2013+) |
| `players_master.csv` | Cross-platform player ID crosswalk (GSIS ↔ PFR) | nflverse, release `players` |
| `fp_weekly_rankings.parquet` | FantasyPros weekly expert consensus rankings, 2019-2025 | dynastyprocess.com/data (ffverse mirror) |

## Pipeline (run in order)
```
python3 01_load_data.py                    # load + filter raw data
python3 02_features.py                     # leak-free features incl. snap share
python3 03_train_model.py                  # v1: single pooled model (Ridge/GBM/XGBoost)
python3 03b_position_quantile_models.py    # v2: position-specific quantile models
python3 04_fantasypros_comparison.py       # backtest vs. FantasyPros rankings
```
Requires: `pandas`, `pyarrow`, `scikit-learn`, `xgboost>=2.0`, `scipy`

## Results

### v2: Position-specific models with prediction intervals (2024 holdout)
| Position | MAE (median) | 10-90% interval coverage | Avg. interval width |
|---|---|---|---|
| QB | 6.24 | 77.4% (target ~80%) | 19.9 pts |
| RB | 4.72 | 82.4% | 16.3 pts |
| WR | 4.89 | 80.5% | 16.5 pts |
| TE | 3.68 | 77.9% | 12.3 pts |

Overall MAE: **4.77**, down from 4.89 in the v1 pooled model — driven mostly by
adding snap-share features and splitting TE out from the pooled model, where
it improved the most (3.87 → 3.68).

**Coverage is the number that matters most here.** An 80% predicted interval
should contain the actual result ~80% of the time, and it does (77–82% across
positions) — meaning the floor/ceiling numbers are genuinely calibrated, not
just a fixed-width band slapped on a point estimate.

### v1: Pooled single model, for reference (2024 holdout)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Naive (3-wk rolling avg) | 5.15 | 7.00 | — |
| Ridge regression | 4.94 | 6.51 | 0.34 |
| Gradient Boosting | 4.90 | 6.46 | 0.35 |
| XGBoost | 4.89 | 6.44 | 0.36 |

### vs. FantasyPros expert consensus rankings (2024 holdout, Spearman rank correlation with actual finish)
| Position | Our model | FantasyPros | N |
|---|---|---|---|
| QB | 0.376 | 0.310 | 386 |
| RB | 0.635 | 0.598 | 893 |
| WR | 0.580 | 0.552 | 1,417 |
| TE | 0.604 | 0.588 | 731 |

Our model edges out FantasyPros' expert consensus on rank correlation with
actual outcomes at every position, in this one-season backtest. **Read this
with real caution, not as a headline claim**: FantasyPros' rankings incorporate
late-breaking info our model doesn't see at all — Friday injury reports,
practice participation, beat-reporter role notes — so some of their misses
are "the game changed after the ranking was made," not "the model is worse."
A fairer comparison would need injury/inactive data as an input (see Next
Steps) before either model gets credit for its remaining edge.

**On Yahoo / Sleeper / ESPN specifically**: unlike FantasyPros, none of these
publish a free, bulk-downloadable historical archive of their own weekly
rankings or point projections, so there's no dataset available to backtest
against for prior seasons. The FantasyPros archive does include Yahoo/ESPN
*roster ownership %* (how many leagues have a player rostered) as a side
column, but that's not a points prediction — different signal entirely. If a
Yahoo/Sleeper/ESPN comparison matters going forward, the realistic version is
a live, current-week snapshot compared against our model's picks in-season,
not a historical backtest.

## Design decisions worth knowing about
1. **No data leakage**: every rolling/season-to-date feature is `.shift(1)`'d
   before rolling, so a player's week-N row only uses data through week N-1.
2. **Time-based split**: train 2013–2022, validate on 2023 (XGBoost early
   stopping), test on 2024. Never randomly shuffle weeks.
3. **2013+ only**: PFR snap count data has no coverage before 2013. Rather
   than fill "no data" as "zero snaps" for 2010-2012 (which would quietly lie
   to the model), those seasons were dropped entirely.
4. **Quantile regression via native XGBoost** (`reg:quantileerror`): three
   models per position (10th/50th/90th percentile) instead of one point
   estimate plus a manually-assumed error margin. This is what makes the
   floor/ceiling numbers empirically calibrated rather than guessed.
5. **Position-specific models**: QB has a very different scoring
   distribution (higher variance, rushing floor for mobile QBs) than
   RB/WR/TE, and pooling them costs the model information about what "normal"
   looks like at each position.
6. **FantasyPros join is name-based** (65.7% match rate) since the dynastyprocess
   mirror doesn't carry nflverse's GSIS player IDs. A stricter/fuzzier name
   normalization step could push this higher if the sample size becomes limiting.

## Natural next steps
- **Injury/inactive report data** (nflverse release `injuries`) — this is the
  single most likely lever to close the gap with FantasyPros, since it's the
  main information their human rankers have that a stats-only model lacks.
- **Next Gen Stats** (release `nextgen_stats`) — separation, air yards, time
  to throw; adds a layer beyond snap share and target share.
- **Live current-week comparison vs. Yahoo/Sleeper/ESPN** — since no historical
  archive exists for these three, a forward-looking weekly comparison (starting
  this coming week) is the realistic version of this ask.
- **Blend model + FantasyPros** rather than picking one — ensembling a stats
  model with expert consensus (which has information the model doesn't)
  usually beats either alone.
- **Fuzzy-match improvement** on the FantasyPros join to recover more of the
  ~34% of test rows that didn't match on name.
