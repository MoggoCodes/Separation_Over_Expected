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

## Project Structure

- `src/separation_over_expected/features.py`: route-table construction from Big Data Bowl tracking, play, player, and PFF files
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
