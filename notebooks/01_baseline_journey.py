# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Context-Adjusted Receiver Separation
#
# This notebook documents the first baseline for the receiver separation project. The research question is:
#
# > Given the situation a route runner is in, how much separation does he create or preserve compared with what an average NFL route runner would be expected to create in the same situation?
#
# The initial target is intentionally narrow:
#
# ```text
# delta_sep = separation_at_pass_release - separation_at_snap
# SOE_route = delta_sep_actual - E[delta_sep | observable snap context]
# ```
#
# That framing avoids claiming total receiver value. It only measures route-level separation creation or preservation from snap to pass release.

# %%
from pathlib import Path
import csv
import statistics
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from separation_over_expected.models import evaluate_models, fit_models, split_rows
from separation_over_expected.reports import dataset_overview, read_csv_rows, receiver_summary_rows

ROUTE_TABLE = PROJECT_ROOT / "data" / "processed" / "route_level_snap_to_release.csv"
PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "route_level_baseline_predictions.csv"
METRICS = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.csv"
RECEIVER_SUMMARY = PROJECT_ROOT / "data" / "processed" / "receiver_baseline_summary.csv"

PROJECT_ROOT

# %% [markdown]
# ## Data Shape
#
# The route table has one row per route runner on plays where we observe both a snap event and a pass-forward event. It includes targeted and untargeted route runners because `pffScoutingData.csv` identifies every player with `pff_role == "Pass Route"`.

# %%
rows = read_csv_rows(ROUTE_TABLE)
overview = dataset_overview(rows)
overview


# %%
def count_by(rows, column):
    counts = {}
    for row in rows:
        counts[row[column]] = counts.get(row[column], 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

count_by(rows, "officialPosition")[:10]


# %% [markdown]
# ## Target Behavior
#
# `delta_sep` is often negative. That is not automatically bad route running. Many receivers start with defensive cushion at the snap, then defenders close space before the throw. The model's job is to learn how much closing is expected from the starting context.

# %%
def describe(values):
    values = sorted(values)
    def q(p):
        return values[int((len(values) - 1) * p)]
    return {
        "min": round(min(values), 3),
        "p05": round(q(0.05), 3),
        "median": round(q(0.50), 3),
        "mean": round(statistics.fmean(values), 3),
        "p95": round(q(0.95), 3),
        "max": round(max(values), 3),
    }

for column in ["sep_snap", "sep_release", "delta_sep", "route_depth", "time_to_throw_frames"]:
    print(column, describe([float(row[column]) for row in rows]))

# %% [markdown]
# ## Baseline Models
#
# We start with transparent baselines before adding heavier modeling libraries:
#
# - `global_mean`: every route gets the training-set mean
# - `smoothed_group_mean`: shrinkage mean by official position, alignment, and man/zone label
# - `ridge_context`: regularized linear model using numeric snap context and one-hot categorical football context
#
# The first split trains on weeks 1-6, validates on week 7, and tests on week 8.

# %%
splits = split_rows(rows)
models = fit_models(splits["train"])
metrics = evaluate_models(models, splits)
metrics

# %% [markdown]
# The useful comparison is not whether the model is perfect. The useful comparison is whether context explains more than a naive average. If the context model reduces error on week 8, we have a workable first definition of expected separation creation.

# %%
for row in metrics:
    if row["split"] in {"validation", "test"}:
        print(f"{row['model']:>20} {row['split']:>10}  R2={row['r2']}  RMSE={row['rmse']}  MAE={row['mae']}")

# %% [markdown]
# ## First Receiver-Level View
#
# For the first leaderboard, `SOE_route` is the residual from the ridge context model. This should be treated as exploratory. It does not yet include player uncertainty, position-specific models, route clusters, or coverage-responsibility modeling.

# %%
scoring_model = next(model for model in models if model.name == "ridge_context")
overall = receiver_summary_rows(rows, scoring_model, min_routes=50)
overall[:15]

# %% [markdown]
# A pooled leaderboard mixes different jobs. Running backs, tight ends, and wide receivers face different route distributions and coverage structures, so the next view separates wide receivers.

# %%
wr_summary = receiver_summary_rows(rows, scoring_model, min_routes=50, position="WR")
wr_summary[:20]

# %% [markdown]
# ## Next Questions
#
# The baseline is good enough to move from feasibility into research. The next useful improvements are:
#
# 1. Fit and report position-specific models for WR, TE, and RB.
# 2. Add uncertainty intervals for receiver summaries so small samples do not masquerade as signal.
# 3. Replace nearest defender with a coverage-aware assignment or weighted defender distance.
# 4. Cluster route shapes using the normalized snap-to-release movement features.
# 5. Validate whether early-season SOE predicts later-season separation or receiving outcomes.
