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
# # Game-Grouped Validation
#
# A route-random split can place routes from the same game in train and test. This experiment assigns whole games to one partition, stratifying the random assignment within each week so every week appears in train, validation, and test.
#
# This tests whether performance changes when the model is evaluated on entirely unseen games from the same season.

# %%
from pathlib import Path
import sys
from collections import Counter

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from separation_over_expected.reports import read_csv_rows

GAME_DIR = PROJECT_ROOT / "data" / "processed" / "game_split" / "position_baselines"
ROUTE_DIR = PROJECT_ROOT / "data" / "processed" / "random_split" / "position_baselines"
rows = read_csv_rows(GAME_DIR / "route_level_baseline_predictions_all.csv")
len(rows)

# %% [markdown]
# ## Confirm Game Separation
#
# Game assignments are made within week, then shared by all route rows from that game. The counts below verify that no game appears in multiple partitions and that the split covers every week.

# %%
rows_by_split = {
    split: [row for row in rows if row["split"] == split]
    for split in ("train", "validation", "test")
}
games_by_split = {
    split: {row["gameId"] for row in split_rows}
    for split, split_rows in rows_by_split.items()
}
overlaps = {
    f"{left}/{right}": len(games_by_split[left] & games_by_split[right])
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
}
split_summary = {
    split: {
        "routes": len(split_rows),
        "games": len(games_by_split[split]),
        "routes_by_week": dict(sorted(Counter(row["week"] for row in split_rows).items())),
    }
    for split, split_rows in rows_by_split.items()
}
{"splits": split_summary, "game_overlap": overlaps}

# %% [markdown]
# ## Test-Set Comparison
#
# Compare every baseline model under route-random and game-grouped splits. Both use seed 42 and all eight weeks; the game split is stratified by week. Because grouping changes the test sample, score differences are a sensitivity check, not a pure estimate of within-game leakage.

# %%
positions = [("All", "all"), ("WR", "wr"), ("TE", "te"), ("RB", "rb")]
comparison = []
for position, suffix in positions:
    metrics_by_strategy = {}
    for strategy, directory in (("Route random", ROUTE_DIR), ("Game grouped", GAME_DIR)):
        metric_rows = read_csv_rows(directory / f"baseline_metrics_{suffix}.csv")
        metrics_by_strategy[strategy] = {
            row["model"]: row for row in metric_rows if row["split"] == "test"
        }
    for model in ("global_mean", "smoothed_group_mean", "ridge_context"):
        route = metrics_by_strategy["Route random"][model]
        game = metrics_by_strategy["Game grouped"][model]
        comparison.append({
            "position": position,
            "model": model,
            "route_test_n": route["n"],
            "game_test_n": game["n"],
            "route_r2": route["r2"],
            "game_r2": game["r2"],
            "route_rmse": route["rmse"],
            "game_rmse": game["rmse"],
            "route_mae": route["mae"],
            "game_mae": game["mae"],
        })
comparison

# %% [markdown]
# ## Ridge Context Results
#
# The grouped split modestly lowers test R2 for the pooled model, WRs, and TEs. RB R2 is nearly unchanged. WR RMSE increases by about 0.07 yards, while TE and RB RMSE are nearly unchanged. This pattern suggests some optimism in the route-level split, especially for WRs, but it does not isolate the cause because the held-out games differ.

# %%
[row for row in comparison if row["model"] == "ridge_context"]

# %% [markdown]
# ## Interpretation
#
# Treat route-random performance as the estimate for held-out routes from the season-wide mix, and game-grouped performance as a stricter check on shared game context. Neither split tests generalization to a different NFL season.
