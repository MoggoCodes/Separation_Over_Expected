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

## Project Structure

- `src/separation_over_expected/features.py`: route-table construction from Big Data Bowl tracking, play, player, and PFF files
- `src/separation_over_expected/models.py`: reusable baseline models, data splits, predictions, and metrics
- `src/separation_over_expected/reports.py`: CSV reading/writing and receiver-level summaries
- `src/separation_over_expected/cli.py`: thin command-line wrapper around the reusable modules
- `notebooks/01_baseline_journey.ipynb`: narrative notebook showing the data shape, target definition, baseline comparison, and first receiver summaries
- `notebooks/02_uncertainty_aware_wr_leaderboard.ipynb`: focused notebook for the uncertainty-aware WR leaderboard
- `notebooks/03_wr_split_half_stability.ipynb`: validation notebook checking whether WR SOE persists from weeks 1-4 to weeks 5-8
