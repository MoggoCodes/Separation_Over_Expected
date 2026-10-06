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
# # Cross-Season Transfer: 2021 to 2023
#
# Does a receiver separation model trained on the original 2021 tracking sample carry over to the 2023 tracking inputs from Big Data Bowl 2026? This is a **preliminary transfer check** using a reduced ridge model restricted to features available in both datasets. It does not evaluate the project's richer dynamic-defender model.
#
# The target in 2021 is snap-to-pass-release change in separation. The 2023 competition files are labeled as pre-throw input and do not expose named snap/release events. A separate timing audit finds strong evidence the input window is snap-to-release-like, but exact row-level event equivalence remains unavailable. Consequently, similar scores are encouraging, with that timing limitation documented, and do not validate the full current model.

# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from separation_over_expected.cross_season import (
    evaluate_cross_season_transfer,
    load_cross_season_wr_rows,
)

LEGACY_ROUTE_TABLE = PROJECT_ROOT / "data" / "processed" / "route_level_snap_to_release.csv"
BDB2023_ROUTE_TABLE = PROJECT_ROOT / "data" / "processed" / "bdb2026" / "route_level_input_window_2023.csv"
NFLVERSE_PBP = PROJECT_ROOT.parent / "data" / "big_data_bowl_2026" / "nflverse_play_by_play_2023.csv"

print(f"2021 route table exists: {LEGACY_ROUTE_TABLE.exists()}")
print(f"2023 route table exists: {BDB2023_ROUTE_TABLE.exists()}")
print(f"2023 play-by-play context exists: {NFLVERSE_PBP.exists()}")

# %% [markdown]
# ## Join checks and shared feature set
#
# The 2023 Big Data Bowl `gameId`/`playId` keys are joined to nflverse `old_game_id`/`play_id` to add down, yards to go, and offense-relative field position. Field position is independently reconstructed from tracking coordinates and checked against nflverse. All features used below are available in both seasons; receiver identity is deliberately excluded from the predictor set.

# %%
legacy_wr, bdb2023_wr, data_checks = load_cross_season_wr_rows(
    LEGACY_ROUTE_TABLE, BDB2023_ROUTE_TABLE, NFLVERSE_PBP
)
data_checks

# %% [markdown]
# ## Game-held-out and cross-season results
#
# The comparison has three evaluation settings: held-out 2021 games, train on all eligible 2021 routes and score 2023, and a game-held-out 2023 reference. The last comparison helps distinguish a year-transfer gap from a general difference between these datasets. All metrics below use the same shared-feature ridge specification and its global-mean baseline.

# %%
results = evaluate_cross_season_transfer(legacy_wr, bdb2023_wr, seed=42)
results

# %% [markdown]
# ## Interpretation and limits
#
# The transferred model scores close to the model trained and game-tested within 2023: RMSE differs by about 0.04 yards and R² by about 0.01. That is encouraging evidence that the shared relationship between starting context and separation change carries across these samples. The global-mean baseline is substantially worse in each setting, so the result is not just a stable target mean.
#
# Treat this as preliminary. The 2021 benchmark uses only eight weeks while 2023 spans the season; the transfer model is a static common-feature model, not our preferred dynamic model; and 2023's frame endpoints lack explicit event labels even though the timing audit supports snap-to-release alignment. A stronger next check is to compare the same feature specification across years, with uncertainty intervals and position subgroups, while carrying this timing limitation forward.
