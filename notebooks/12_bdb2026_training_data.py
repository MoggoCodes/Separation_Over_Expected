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
# # BDB 2026 Training Data: First-to-Last Input Separation
#
# This notebook checks whether the 2023 training inputs from the 2026 Big Data Bowl can support our receiver-separation research. It builds a reusable route-runner table from tracking roles and compares each player's distance to the nearest coverage defender at the first and last observed input frames.
#
# **Measurement caveat:** the prediction files identify pre-throw tracking frames, but the rows do not include named `ball_snap` or `pass_forward` events. This notebook therefore calls the endpoints *first input frame* and *last input frame*. A separate timing audit finds strong evidence this is a snap-to-release-like window while preserving that the exact event frames are not provided; see `15_frame_timing_audit.ipynb`.

# %%
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from separation_over_expected.bdb2026 import build_bdb2026_route_table
from separation_over_expected.reports import read_csv_rows

DATA_DIR = PROJECT_ROOT.parent / "data" / "big_data_bowl_2026"
ROUTE_TABLE = PROJECT_ROOT / "data" / "processed" / "bdb2026" / "route_level_input_window_2023.csv"

print(f"Training input directory exists: {DATA_DIR.is_dir()}")
print(f"Route table: {ROUTE_TABLE.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## Build and Validate the Reusable Route Table
#
# The builder retains the first and latest observation for each player in each play. It calculates nearest-coverage-defender distance at both endpoints for the `Targeted Receiver` and `Other Route Runner` roles. It does not use `ball_land_x`, `ball_land_y`, post-throw output tracks, or the Kaggle evaluation sample.

# %%
diagnostics = build_bdb2026_route_table(DATA_DIR, ROUTE_TABLE)
diagnostics

# %%
rows = read_csv_rows(ROUTE_TABLE)
play_keys = {(row["gameId"], row["playId"]) for row in rows}
route_keys = {(row["gameId"], row["playId"], row["nflId"]) for row in rows}
role_counts = Counter(row["player_role"] for row in rows)
position_counts = Counter(row["officialPosition"] for row in rows)
weekly_counts = Counter(int(row["week"]) for row in rows)
targeted_per_play = Counter()
for row in rows:
    if row["targeted"] == "true":
        targeted_per_play[(row["gameId"], row["playId"])] += 1

quality_summary = {
    "route_runner_rows": len(rows),
    "unique_plays_with_usable_endpoints": len(play_keys),
    "unique_game_ids": len({row["gameId"] for row in rows}),
    "duplicate_game_play_player_keys": len(rows) - len(route_keys),
    "plays_with_exactly_one_targeted_receiver": sum(value == 1 for value in targeted_per_play.values()),
    "plays_with_multiple_targeted_receivers": sum(value > 1 for value in targeted_per_play.values()),
    "blank_values_in_rows": sum(any(value == "" for value in row.values()) for row in rows),
    "player_roles": dict(role_counts),
    "positions": dict(position_counts.most_common()),
    "routes_by_week": dict(sorted(weekly_counts.items())),
}
quality_summary

# %% [markdown]
# The extracted table contains 64,751 routes from 14,107 plays. It includes targeted and untargeted route runners, with one route row per player-play. The source has one additional play that could not be retained because coverage context was missing at an endpoint.

# %% [markdown]
# ## Endpoint and Separation Checks
#
# `delta_sep_input_window` is last-input-frame separation minus first-input-frame separation. It uses the nearest player labeled `Defensive Coverage` at each endpoint; the identity can change between endpoints. This is a transparent starting target, not a coverage-responsibility or receiver-credit estimate.

# %%
def describe(column, subset):
    values = sorted(float(row[column]) for row in subset)
    if not values:
        return {"n": 0}
    def q(probability):
        index = int((len(values) - 1) * probability)
        return round(values[index], 3)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "sd": round(statistics.pstdev(values), 3),
        "p05": q(0.05),
        "median": q(0.50),
        "p95": q(0.95),
    }

separation_summary = {
    "first-frame separation": describe("sep_first_input", rows),
    "last-frame separation": describe("sep_last_input", rows),
    "change across input window": describe("delta_sep_input_window", rows),
    "targeted receivers": describe("delta_sep_input_window", [r for r in rows if r["targeted"] == "true"]),
    "other route runners": describe("delta_sep_input_window", [r for r in rows if r["targeted"] == "false"]),
}
separation_summary

# %% [markdown]
# Separation usually decreases over the observed window. That is a raw descriptive result: defenders close space, and the nearest defender can change. We need an expected-outcome model before interpreting a route residual as separation above expected.

# %%
window_frames = [int(row["input_window_frames"]) for row in rows]
window_summary = {
    "minimum frames after first observation": min(window_frames),
    "median frames after first observation": sorted(window_frames)[len(window_frames) // 2],
    "95th percentile frames after first observation": sorted(window_frames)[int(0.95 * (len(window_frames) - 1))],
    "maximum frames after first observation": max(window_frames),
    "approximate seconds at 10 Hz (median)": round(sorted(window_frames)[len(window_frames) // 2] / 10, 1),
}
window_summary

# %% [markdown]
# ## What This Dataset Supports
#
# The 2023 training inputs are large enough to fit and validate a first tracking-context expected-separation model. Observable candidate predictors include player position, field-normalized starting location, starting separation and defender leverage, receiver speed/acceleration, field position, and the length of the observed pre-throw window. Splits should keep all routes from a game together while sampling games across the 18 weeks.
#
# The files do not supply down/distance, formation, route labels, or defender-to-receiver assignments. We should join reliable play context before claiming a broad "same situation" estimate. `player_to_predict` and ball-landing coordinates are unrelated to the separation target and must stay out of its feature set.
#
# The next modeling step is therefore a game-grouped, all-weeks baseline using only verified pre-throw tracking features, followed by a sensitivity check once we establish that frame 1 and the last input frame align with snap and pass release.
