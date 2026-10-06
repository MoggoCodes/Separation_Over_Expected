# Separation Over Expected

This project estimates how much separation a receiver creates compared with what an average NFL route runner would be expected to create in the same observable situation.

The first estimand is route-level separation creation:

```text
delta_sep = separation_at_pass_release - separation_at_snap
SOE_route = delta_sep_actual - E[delta_sep | route context at snap]
```

For the first version, "same situation" means the observable context available in the NFL Big Data Bowl 2023 data:

- receiver alignment and official position
- initial receiver location, speed, and acceleration
- nearest coverage defender location and leverage at the snap
- offensive formation, personnel, down, distance, field position, and play action
- defensive personnel, coverage family, and man/zone label
- time from snap to pass release

This framing avoids claiming full receiver value. It asks a narrower question: given where the receiver started and how the defense was structured, did he create more or less separation before the quarterback released the ball than an average NFL route runner would have created?

## BDB 2026 Training-Data Feasibility Pass

The 2026 prediction competition's 2023 training inputs provide pre-throw tracking for targeted receivers, other route runners, and coverage defenders. The first extraction builds one row per route runner and measures nearest-coverage-defender separation from the first to the last input frame:

```bash
uv run separation-over-expected build-bdb2026-route-table \
  --data-dir ../data/big_data_bowl_2026 \
  --output data/processed/bdb2026/route_level_input_window_2023.csv
```

The current extraction contains 64,751 route rows from 14,107 usable plays across all 18 weeks. One additional play is omitted because coverage context is missing at an endpoint. It includes both targeted and untargeted route runners and excludes ball-landing coordinates and post-throw output tracks.

The prediction inputs are described as pre-throw tracking and do not carry named `ball_snap` or `pass_forward` events, so the extracted outcome remains explicitly named `delta_sep_input_window`. A timing audit finds strong support that this is a snap-to-release-like sequence: the WR window median is 25 frames versus 27 in the event-anchored 2021 data; first-frame receiver speeds resemble 2021 snap speeds (93.1% versus 95.3% of WR rows below 0.5 yd/s); and 2,679 matched input/output player-play trajectories have a median boundary displacement of 0.454 yd. An [NFL/AWS Next Gen Stats presentation](https://d1.awsstatic.com/events/Summits/reinvent2023/PRO304_NFL-Next-Gen-Stats-Using-AI-ML-to-transform-fan-engagement.pdf) defines the corresponding pre-pass sequence as snap through release. Exact row-level event equivalence is still unavailable, so document that limitation in cross-season evaluation. This dataset also lacks down/distance, route labels, formation, and defender assignment, so the first expected-separation model must be limited to observed tracking context or joined to a reliable play-context source.

`notebooks/12_bdb2026_training_data.ipynb` shows extraction diagnostics and route coverage; `notebooks/15_frame_timing_audit.ipynb` evaluates window length, initial speed, and input/output-boundary evidence.

## Cross-Season Transfer Check

We ran a preliminary WR transfer experiment from the original 2021 tracking sample to the 2023 BDB 2026 training inputs. The 2023 route rows were joined to [nflverse play-by-play](https://github.com/nflverse/nflverse-data) using the competition's game/play identifiers, adding down, yards to go, and field position. All 14,107 usable tracking plays joined; the offense-relative yardline derived from tracking coordinates matched nflverse exactly.

Because the two route tables do not share every engineered variable, this experiment uses a reduced, static ridge model with only common fields: starting separation and location, nearest-defender leverage, receiver speed/acceleration, observed window length, down, distance, and field position. No receiver identifier is a model feature. Results:

| Evaluation | Model | Routes | R² | RMSE (yd) | MAE (yd) |
|---|---|---:|---:|---:|---:|
| 2021 held-out games | Shared-feature ridge | 3,527 | 0.442 | 2.036 | 1.496 |
| Train 2021, score 2023 | Shared-feature ridge | 38,002 | 0.448 | 1.959 | 1.483 |
| 2023 held-out games | Shared-feature ridge | 6,361 | 0.458 | 1.921 | 1.476 |

The transfer score is close to the within-2023 reference (RMSE +0.038 yd; R² -0.010), which is promising evidence that this shared-feature relationship carries across the samples. It does **not** yet establish that our preferred dynamic-defender model transfers: this is a smaller static model. The 2023 input window is strongly supported as snap-to-release-like by the timing audit, but named snap/release frame events are unavailable for direct confirmation. The samples also differ in season coverage (eight 2021 weeks versus the 2023 season), so treat this as encouraging external validation with a documented timing limitation, not definitive proof.

Reproduce the check with:

```bash
uv run separation-over-expected compare-cross-season
```

`notebooks/13_cross_season_transfer.ipynb` documents the joined features, data checks, results, and interpretation. The Kaggle competition dataset description documents the pre-throw input and play-ID crosswalk to nflverse ([official data page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/data)); `notebooks/15_frame_timing_audit.ipynb` assesses the snap-to-release timing alignment.

## Dynamic Feature Transfer Check

We also tested whether the dynamic defender-motion features transfer. The 2023 builder ranks the three nearest players tagged `Defensive Coverage` at the first input frame, holds those players fixed, and calculates the same eight pre-release motion summaries per defender as the 2021 pipeline. The final input frame is excluded. Since 2023 lacks several PFF charting, formation, and personnel fields used in our preferred model, we trained matched **common-feature** 2021 static and dynamic ridge models and scored both on the same 2023 WR routes.

| Evaluation | Static RMSE | Dynamic RMSE | Static R² | Dynamic R² |
|---|---:|---:|---:|---:|
| 2021 game holdout | 2.036 | 1.960 | 0.442 | 0.483 |
| Train 2021, score 2023 | 1.959 | 1.903 | 0.448 | 0.479 |
| 2023 game holdout | 1.921 | 1.867 | 0.458 | 0.487 |

The dynamic model lowers RMSE by about 0.06 yd on both the 2021-to-2023 transfer set and the within-2023 game holdout. Paired game-cluster bootstrap intervals for dynamic-minus-static RMSE are [-0.062, -0.052] yd on the transfer set and [-0.064, -0.044] yd on the 2023 holdout. This supports portable predictive information in the dynamic feature block. It is not a test of the full existing dynamic model, and the BDB `Defensive Coverage` role is only an approximation to the 2021 PFF coverage assignment.

Rebuild and reproduce with:

```bash
uv run separation-over-expected build-bdb2026-route-table \
  --include-dynamic-features \
  --output data/processed/bdb2026/route_level_input_window_2023_dynamic.csv

uv run separation-over-expected compare-cross-season-dynamic
```

`notebooks/16_dynamic_cross_season_transfer.ipynb` shows feature support, same-season comparisons, transferred scores, and game-cluster uncertainty.

## Cross-Season Receiver Reliability

Route-level transfer does not tell us whether receiver residuals repeat for the same players. `notebooks/17_cross_season_receiver_reliability.ipynb` compares receiver mean residuals from five-fold, game-held-out 2021 predictions with 2023 residuals from models frozen on all eligible 2021 routes. It evaluates the static and dynamic shared-feature ridge models on the 91 receivers with at least 20 routes across five games in each season. Residuals are centered within season to remove a common calibration shift; player uncertainty intervals resample games, and cross-player correlation intervals use a nested bootstrap of matched receivers and their games.

Dynamic defender-motion features show higher observed cross-season repeatability in this sample than the static model: Pearson correlation is 0.459 versus 0.375 and Spearman correlation is 0.122 versus -0.004. The nested paired bootstrap, which resamples matched receivers and their game clusters, estimates a Pearson gain of +0.083 (95% interval [+0.001, +0.135]) and a Spearman gain of +0.125 ([-0.029, +0.145]). The absolute-correlation intervals are broad and include zero; the ranking-agreement difference is also uncertain. This is tentative evidence that the dynamic specification may improve repeatability, not proof of stable player rankings or isolated receiver skill. The 2021 sample covers eight weeks; 2023 covers a full season and has only snap-to-release-like frame timing.

Reproduce the tables with:

```bash
uv run separation-over-expected cross-season-receiver-reliability
```

The route predictions, receiver summaries, and reliability metrics are written beneath the ignored `data/processed/cross_season_receiver_reliability/` directory.

## Calibration Diagnostics

The model is intended to estimate expected route-level separation change, so evaluation should check more than R². `notebooks/14_model_calibration.ipynb` uses game-grouped out-of-fold predictions from the current dynamic WR ridge model to compare mean observed and predicted separation change across prediction deciles. Its calibration chart includes game-cluster bootstrap intervals, and a second plot checks residual bias across starting-separation and observed-window-duration groups. In this run, the mean prediction is -2.342 yards versus -2.341 observed, with calibration slope 0.997 and intercept -0.006; prediction-decile mean residuals range from about -0.07 to +0.10 yards. The decile view assesses average calibration; the context plot helps reveal local bias. These results support the route-level baseline, but do not establish player-level reliability.

## Build the Route Table

From the project directory:

```bash
uv run separation-over-expected build-route-table \
  --data-dir ../data/big_data_bowl_2023 \
  --output data/processed/route_level_snap_to_release.csv
```

The output contains one row per route runner on plays with both a snap event and a pass-forward event. It includes raw and normalized coordinates, nearest-defender separation at snap and release, and contextual fields for the expected-separation model.

## Fit Baseline Models

```bash
uv run separation-over-expected fit-baselines \
  --route-table data/processed/route_level_snap_to_release.csv \
  --predictions data/processed/route_level_baseline_predictions.csv \
  --receiver-summary data/processed/receiver_baseline_summary.csv \
  --metrics data/processed/baseline_metrics.csv
```

The baseline target is `delta_sep`. The first split trains on weeks 1-6, validates on week 7, and tests on week 8.

Current baselines:

- `global_mean`: every route receives the training-set average `delta_sep`
- `smoothed_group_mean`: shrinkage mean by official position, alignment, and man/zone coverage type
- `ridge_context`: ridge regression using snap/release-time context, including initial separation, field location, defender leverage, receiver speed/acceleration, time to throw, formation, personnel, coverage, and alignment

The first baseline run produced:

| Model | Validation R2 | Test R2 | Test RMSE |
|---|---:|---:|---:|
| global_mean | -0.000 | -0.003 | 2.904 |
| smoothed_group_mean | 0.044 | 0.045 | 2.834 |
| ridge_context | 0.308 | 0.327 | 2.380 |

`ridge_context` is the first working expected-separation model. Its route-level residual is the initial `SOE_route` score:

```text
SOE_route = delta_sep_actual - delta_sep_predicted
```

The receiver summary aggregates those route-level residuals. Treat that leaderboard as exploratory because the current model pools WR, TE, RB, and FB routes and does not yet estimate player uncertainty.

## Fit Position-Specific Baselines

The pooled model is useful for diagnostics, but WR, TE, and RB routes represent different football jobs. Position-specific baselines fit separate expected-separation models within each position group:

```bash
uv run separation-over-expected fit-position-baselines \
  --route-table data/processed/route_level_snap_to_release.csv \
  --output-dir data/processed/position_baselines \
  --positions WR TE RB
```

Current test-set results:

| Position | Ridge Test R2 | Ridge Test RMSE | Test Rows |
|---|---:|---:|---:|
| All | 0.327 | 2.380 | 4,252 |
| WR | 0.503 | 1.848 | 2,474 |
| TE | 0.283 | 2.431 | 1,002 |
| RB | 0.151 | 3.414 | 745 |

The WR-specific model is the cleanest first project surface. It compares wide receivers to other wide receivers instead of mixing route jobs across positions.

Receiver summaries are sorted by `lower_95_soe`, an uncertainty-aware score:

```text
lower_95_soe = mean_soe - 1.96 * standard_error(mean_soe)
```

This keeps the leaderboard from overvaluing small samples with noisy high averages. The summary files also include `std_soe`, `se_soe`, and `upper_95_soe`.

## Check Split-Half Stability

The first validation check asks whether WR SOE in weeks 1-4 carries into weeks 5-8:

```bash
uv run separation-over-expected split-half-stability \
  --predictions data/processed/position_baselines/route_level_baseline_predictions_wr.csv \
  --output data/processed/stability/wr_split_half_stability.csv \
  --position WR \
  --min-routes-per-half 20
```

With at least 20 WR routes in each half, 107 receivers qualify. The early/late Pearson correlation is 0.420. That is meaningful signal for a first tracking-data metric, but it also leaves plenty of noise for future route-shape and defender-context improvements.

## Coverage Context Experiment

The coverage-context experiment adds local defensive context at the snap: second/third defender distance, defender density, nearest DB/LB distance, and leverage features. The WR model's test performance is nearly unchanged:

| WR Ridge Model | Test R2 | Test RMSE | Test MAE | Split-Half Corr |
|---|---:|---:|---:|---:|
| Baseline | 0.503 | 1.848 | 1.393 | 0.420 |
| Coverage context | 0.503 | 1.848 | 1.388 | 0.406 |

This simple defensive-context feature set does not materially improve the WR model. That suggests the next improvement needs route-shape or time-varying coverage-responsibility information rather than snap-only defender density.

## Random Split Across All Weeks

To estimate performance on a representative mix of the 2021 season, the same pooled and position-specific models can also use a seeded random route-level split across all eight weeks. The default is 70% train, 15% validation, and 15% test; seed 42 reproduces the checked-in experiment. The existing week-based evaluation remains available for measuring forward-in-time performance.

```bash
uv run separation-over-expected fit-position-baselines \
  --route-table data/processed/route_level_snap_to_release.csv \
  --output-dir data/processed/random_split/position_baselines \
  --positions WR TE RB \
  --split-strategy random \
  --seed 42
```

The random split test results are:

| Position | Model | Test R2 | Test RMSE | Test MAE | Routes |
|---|---|---:|---:|---:|---:|
| All | ridge_context | 0.359 | 2.363 | 1.711 | 5,256 |
| WR | ridge_context | 0.514 | 1.895 | 1.381 | 3,082 |
| TE | ridge_context | 0.288 | 2.373 | 1.704 | 1,173 |
| RB | ridge_context | 0.235 | 3.241 | 2.582 | 939 |

These metrics describe random held-out routes drawn from the same season-wide mix; they are not a future-week forecast. Routes are split independently, so routes from the same play can appear in more than one partition. Receiver summaries in this experiment use test routes only.

## Game-Grouped Validation

As a dependence check, the split can keep every route from a game together while assigning games within each week to train, validation, and test. This preserves coverage across all eight weeks and prevents any game from appearing in multiple partitions:

```bash
uv run separation-over-expected fit-position-baselines \
  --route-table data/processed/route_level_snap_to_release.csv \
  --output-dir data/processed/game_split/position_baselines \
  --positions WR TE RB \
  --split-strategy game \
  --seed 42
```

The 122 eligible games split into 85 train, 16 validation, and 21 test games. The grouped test results were:

| Position | Random-route Test R2 | Game-split Test R2 | Random-route RMSE | Game-split RMSE |
|---|---:|---:|---:|---:|
| All | 0.359 | 0.340 | 2.363 | 2.392 |
| WR | 0.514 | 0.479 | 1.895 | 1.967 |
| TE | 0.288 | 0.266 | 2.373 | 2.369 |
| RB | 0.235 | 0.237 | 3.241 | 3.243 |

The game split lowers ridge R2 modestly for all routes, WRs, and TEs, while RB performance is nearly unchanged. This suggests the route-level split may be somewhat optimistic, especially for WRs, but the test samples also differ, so it is a robustness comparison rather than a direct measurement of leakage.

## Dynamic Defender Context Experiment

The dynamic-context experiment retains the same snap-to-release target and adds motion summaries for the three PFF coverage defenders nearest each route runner at the snap. Defender identities stay fixed throughout the route; features summarize speed, acceleration, normalized displacement, path length, heading change, and frame coverage before release. The release frame itself is excluded from these dynamic summaries. Missing dynamic values are imputed with training-set means.

Build the extended route table and fit the static and dynamic ridge models on the same game-grouped split:

```bash
uv run separation-over-expected build-route-table \
  --data-dir ../data/big_data_bowl_2023 \
  --output data/processed/dynamic_features/route_level_snap_to_release_dynamic.csv \
  --include-dynamic-features

uv run separation-over-expected fit-position-baselines \
  --route-table data/processed/dynamic_features/route_level_snap_to_release_dynamic.csv \
  --output-dir data/processed/dynamic_features/position_baselines \
  --positions WR TE RB \
  --include-dynamic-features \
  --split-strategy game \
  --seed 42
```

On the held-out game test set, the WR ridge model improves from R2 0.479 / RMSE 1.967 to R2 0.500 / RMSE 1.928. The dynamic model also improves test R2 for pooled routes, TEs, and RBs. The static scores exactly reproduce the prior game-split metrics, which confirms that adding these columns did not change the target, route population, or split. These are predictive gains, not causal attribution: defensive motion can reflect coverage, route combinations, pressure, and play design as well as receiver behavior.

Further checks temper that result. On WR validation routes, using the nearest defender alone gives RMSE 1.820; adding the second and third defenders moves it only to 1.813 and 1.812. All three defenders have complete tracking in this dataset, so changing the 80/90/100% frame-coverage threshold has no effect. The aggregate validation gain is small and varies by subgroup. A split-half validation-game check gives a receiver residual correlation of -0.136 for 13 eligible receivers, with a wide 95% bootstrap interval of -0.634 to 0.371. This is too uncertain to support stable player rankings; the dynamic features currently show clearer value for route-level prediction than for player evaluation.

`notebooks/07_dynamic_context_features.ipynb` documents the feature construction, data and split checks, static-versus-dynamic results, and an illustrative route-separation trajectory. The chart is a diagnostic only; time-varying nearest-defender separation is not used as a model input.

## Game-Grouped Out-of-Fold Receiver Reliability

To test player-level consistency using more than one small validation subset, fit five game-grouped folds, stratified within week. Every WR route receives static and dynamic ridge predictions from a model that did not train on that route's game. The same routes are split into two balanced game halves for receiver SOE reliability:

```bash
uv run separation-over-expected cross-validate-position \
  --route-table data/processed/dynamic_features/route_level_snap_to_release_dynamic.csv \
  --output-dir data/processed/dynamic_features/cross_validation \
  --position WR \
  --folds 5 \
  --seed 42 \
  --min-routes-per-half 20
```

Across 20,415 WR routes, dynamic context modestly improves out-of-fold route prediction (RMSE 1.897 to 1.869; R2 0.493 to 0.508). Receiver split-half Pearson stability is 0.431 for static ridge and 0.452 for dynamic ridge across 115 eligible receivers. The paired dynamic-minus-static difference is 0.021 (95% bootstrap interval -0.033 to 0.054), so this does not establish stronger receiver evaluation; Spearman ranking stability is slightly lower for dynamic ridge. This is a within-season analysis, not year-to-year validation.

`notebooks/08_game_grouped_oof_receiver_reliability.ipynb` shows fold coverage, route prediction metrics, receiver-level correlations, uncertainty intervals, and SOE scatterplots for the two game halves.

## Pre-Release Pocket Context Experiment

The pocket-context experiment adds 14 pre-release features for quarterback movement and distance to PFF-tagged pass rushers. The release frame is excluded. It compares static ridge, existing dynamic defender-context ridge, and static + dynamic + pocket-context ridge on the same five game-grouped folds:

```bash
uv run separation-over-expected build-route-table \
  --data-dir ../data/big_data_bowl_2023 \
  --output data/processed/dynamic_features/pocket_context/route_level_snap_to_release_pocket.csv \
  --include-pocket-features

uv run separation-over-expected cross-validate-position \
  --route-table data/processed/dynamic_features/pocket_context/route_level_snap_to_release_pocket.csv \
  --output-dir data/processed/dynamic_features/pocket_context/cross_validation \
  --position WR --folds 5 --seed 42 --min-routes-per-half 20 \
  --bootstrap-samples 2000 --include-pocket-features
```

Across 20,415 out-of-fold WR routes, pocket context changes route RMSE from 1.869 to 1.867 and R² from 0.508 to 0.509 relative to dynamic context, a negligible gain. Receiver split-half Pearson reliability drops from 0.452 to 0.424; the paired difference is -0.028 (95% bootstrap interval -0.050 to -0.011). Spearman reliability also declines, with an interval that includes zero. The experiment therefore does not support adding this broad pocket feature set to the preferred receiver model. These results are within-season and do not demonstrate year-to-year stability.

`notebooks/09_pre_release_pocket_context.ipynb` documents the feature coverage, fold-level comparisons, receiver reliability, and interpretation.

## Pressure-Only Feature Experiment

To isolate the pressure signal from QB movement, an additional ridge model uses only six pass-rusher proximity/closing features, alongside the existing static, dynamic, and broad pocket models. The route table built for the pocket experiment already contains these fields. Run the same five-fold game-grouped comparison with:

```bash
uv run separation-over-expected cross-validate-position \
  --route-table data/processed/dynamic_features/pocket_context/route_level_snap_to_release_pocket.csv \
  --output-dir data/processed/dynamic_features/pressure_context/cross_validation \
  --position WR --folds 5 --seed 42 --min-routes-per-half 20 \
  --bootstrap-samples 2000 --include-pocket-features --include-pressure-context
```

Across the same 20,415 OOF WR routes, pressure-only slightly improves RMSE/R² over dynamic context (1.869 to 1.867; 0.508 to 0.509), but receiver split-half Pearson reliability falls from 0.452 to 0.434. The paired difference is -0.018 (95% bootstrap interval -0.035 to 0.005), so there is no clear reliability change. The broad pocket model gives similar route prediction but lower receiver reliability than pressure-only. Keep dynamic context as the current preferred model; the modest route-level gain does not establish better player evaluation.

`notebooks/10_pressure_only_features.ipynb` documents the feature subset, fold metrics, and receiver reliability comparison.

## Project Structure

- `src/separation_over_expected/features.py`: route-table construction from Big Data Bowl tracking, play, player, and PFF files
- `src/separation_over_expected/bdb2026.py`: route-level extraction from BDB 2026 pre-throw training inputs
- `src/separation_over_expected/models.py`: reusable baseline models, data splits, predictions, and metrics
- `src/separation_over_expected/reports.py`: CSV reading/writing and receiver-level summaries
- `src/separation_over_expected/cli.py`: thin command-line wrapper around the reusable modules
- `notebooks/01_baseline_journey.ipynb`: narrative notebook showing the data shape, target definition, baseline comparison, and first receiver summaries
- `notebooks/02_uncertainty_aware_wr_leaderboard.ipynb`: focused notebook for the uncertainty-aware WR leaderboard
- `notebooks/03_wr_split_half_stability.ipynb`: validation notebook checking whether WR SOE persists from weeks 1-4 to weeks 5-8
- `notebooks/04_coverage_context_features.ipynb`: experiment notebook showing that simple snap-level coverage context does not materially improve WR performance or stability
- `notebooks/05_random_split_model_evaluation.ipynb`: comparison of season-wide random-split results with the week-based evaluation
- `notebooks/06_game_grouped_validation.ipynb`: comparison of route-random and game-grouped season-wide evaluation
- `notebooks/07_dynamic_context_features.ipynb`: pre-release defender-motion feature experiment and model comparison
- `notebooks/08_game_grouped_oof_receiver_reliability.ipynb`: five-fold game-grouped out-of-fold predictions and receiver SOE stability
- `notebooks/09_pre_release_pocket_context.ipynb`: pre-release QB/pocket feature experiment and its route- and receiver-level results
- `notebooks/10_pressure_only_features.ipynb`: pressure-only feature subset compared with dynamic and broad pocket context
- `notebooks/12_bdb2026_training_data.ipynb`: feasibility analysis of BDB 2026's 2023 training tracking inputs
- `notebooks/13_cross_season_transfer.ipynb`: preliminary shared-feature model transfer from 2021 tracking data to 2023
- `notebooks/14_model_calibration.ipynb`: game-grouped out-of-fold calibration and context-slice diagnostics for the dynamic WR model
- `notebooks/15_frame_timing_audit.ipynb`: empirical and source-based audit of the 2023 pre-throw window against the 2021 snap-to-release interval
- `notebooks/16_dynamic_cross_season_transfer.ipynb`: frozen 2021 common-feature static/dynamic ridge evaluation on 2023 routes
