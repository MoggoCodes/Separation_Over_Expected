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
# # Random Split Model Evaluation
#
# This experiment trains on a random 70% of route rows drawn from all eight weeks, compares models on a 15% validation sample, and reports final metrics on a separate 15% test sample. Seed 42 makes the split reproducible.
#
# This answers how well the model predicts another route from the same season-wide mix. It complements the earlier week-based split, which asks whether a model trained on earlier weeks carries forward to a later week.

# %%
from pathlib import Path
import csv
import sys
from collections import Counter

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from separation_over_expected.reports import read_csv_rows

ROUTE_TABLE = PROJECT_ROOT / "data" / "processed" / "route_level_snap_to_release.csv"
RANDOM_DIR = PROJECT_ROOT / "data" / "processed" / "random_split" / "position_baselines"
WEEK_DIR = PROJECT_ROOT / "data" / "processed" / "position_baselines"

rows = read_csv_rows(RANDOM_DIR / "route_level_baseline_predictions_all.csv")
len(rows)

# %% [markdown]
# ## Split Coverage
#
# Each partition contains routes from all eight weeks. A route is the sampling unit, so routes from the same play can land in different partitions. Notebook 06 follows up with a game-grouped split to check the effect of keeping games together.

# %%
split_counts = Counter(row["split"] for row in rows)
week_counts = {
    split: dict(sorted(Counter(row["week"] for row in rows if row["split"] == split).items()))
    for split in ("train", "validation", "test")
}
{"split_counts": dict(split_counts), "routes_by_week": week_counts}

# %% [markdown]
# ## Test Metrics: All Models and Positions
#
# The table compares the original week-based evaluation with the new season-wide random evaluation. Both use the same model definitions and tracked route table. Random split receiver summaries are computed from test routes only.

# %%
positions = [("All", "all"), ("WR", "wr"), ("TE", "te"), ("RB", "rb")]
comparison = []
for position, suffix in positions:
    for strategy, directory in (("Week holdout", WEEK_DIR), ("Random all weeks", RANDOM_DIR)):
        metric_rows = read_csv_rows(directory / f"baseline_metrics_{suffix}.csv")
        for row in metric_rows:
            if row["split"] == "test":
                comparison.append({
                    "position": position,
                    "evaluation": strategy,
                    "model": row["model"],
                    "routes": row["n"],
                    "r2": row["r2"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                })
comparison

# %% [markdown]
# ## Wide Receiver Test-Set Summary
#
# The WR ridge model is the main project surface. These receiver-level summaries use only held-out test routes, with at least 25 routes in the test partition.

# %%
wr_summary = read_csv_rows(RANDOM_DIR / "receiver_baseline_summary_wr.csv")
wr_summary[:15]

# %% [markdown]
# ## Interpretation
#
# The random split evaluates interpolation across the observed season-wide distribution. The week holdout evaluates later-week transfer. Similar results would suggest the model is not highly sensitive to that particular week boundary; differences can reflect distribution shift as well as test-sample variation.
#
# This experiment uses a route-level random split. It does not establish that routes within a play are statistically independent, and the test results should be described as same-season route prediction rather than future-season performance. See notebook 06 for the grouped-game robustness check.
