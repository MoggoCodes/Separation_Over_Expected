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
# # Frame-Timing Audit: Is 2023 Comparable to Snap-to-Release?
#
# The 2023 BDB 2026 input files have frame IDs but no `ball_snap` or `pass_forward` event labels. The official Kaggle description says these are tracking frames before the pass is thrown. An NFL/AWS Next Gen Stats presentation describes the corresponding pre-pass time series as starting at the snap and ending at QB release ([AWS/NFL presentation, slide 12](https://d1.awsstatic.com/events/Summits/reinvent2023/PRO304_NFL-Next-Gen-Stats-Using-AI-ML-to-transform-fan-engagement.pdf)); this supports that convention, but it is not an event annotation attached to each competition row. This audit checks whether local duration and endpoint evidence is consistent with that convention.

# %%
from collections import defaultdict
from pathlib import Path
import csv
import math
import statistics
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from separation_over_expected.reports import read_csv_rows

OLD_ROUTES = PROJECT_ROOT / "data" / "processed" / "route_level_snap_to_release.csv"
NEW_ROUTES = PROJECT_ROOT / "data" / "processed" / "bdb2026" / "route_level_input_window_2023.csv"
BDB_DATA = PROJECT_ROOT.parent / "data" / "big_data_bowl_2026"

old_rows = read_csv_rows(OLD_ROUTES)
new_rows = read_csv_rows(NEW_ROUTES)
print(f"2021 route rows: {len(old_rows):,}; 2023 route rows: {len(new_rows):,}")

# %% [markdown]
# ## Compare observed window lengths
#
# The 2021 table measures frames from its explicit `ball_snap` event to `pass_forward`. The 2023 table measures from the first to last input frame. At 10 frames per second, similar frame-count distributions are a useful consistency check, though they do not prove each 2023 endpoint's event identity.

# %%
def duration_summary(rows, field):
    values = sorted(int(row[field]) for row in rows if row.get("officialPosition") == "WR")
    return {
        "WR routes": len(values),
        "mean frames": round(statistics.fmean(values), 2),
        "median frames": statistics.median(values),
        "p10 frames": values[int(0.10 * (len(values) - 1))],
        "p90 frames": values[int(0.90 * (len(values) - 1))],
        "median seconds at 10 Hz": round(statistics.median(values) / 10, 2),
    }


duration_comparison = {
    "2021 explicit snap-to-release": duration_summary(old_rows, "time_to_throw_frames"),
    "2023 first-to-last input": duration_summary(new_rows, "input_window_frames"),
}
duration_comparison

# %% [markdown]
# ## Compare the initial player state
#
# If frame 1 is at or very near the snap, route runners should usually still be stationary, as they are at the explicitly tagged 2021 snap. We compare the share of WR/TE/RB route rows with speed below 0.5 yards/second. This is corroborating evidence, not a snap detector: motion rules, player mix, and tracking processing may differ across seasons.

# %%
initial_speed_comparison = {}
for position in ("WR", "TE", "RB"):
    initial_speed_comparison[position] = {}
    for label, data, column in (
        ("2021 at ball_snap", old_rows, "receiver_speed_snap"),
        ("2023 at first input frame", new_rows, "receiver_speed_first"),
    ):
        speeds = [float(row[column]) for row in data if row.get("officialPosition") == position]
        initial_speed_comparison[position][label] = {
            "routes": len(speeds),
            "median speed": round(statistics.median(speeds), 3),
            "share below 0.5 yd/s": round(sum(speed < 0.5 for speed in speeds) / len(speeds), 3),
        }
initial_speed_comparison

# %% [markdown]
# ## Check continuity across the input/output boundary
#
# The competition describes output tracks as post-throw positions. In week 1, match players observed in both files and compare each player's last input location with their first output location. Small displacement is consistent with adjacent 10 Hz frames across the throw boundary. The IDs are reset by file type, so this checks continuity, not the named release event itself.

# %%
input_path = BDB_DATA / "train" / "input_2023_w01.csv"
output_path = BDB_DATA / "train" / "output_2023_w01.csv"
input_tracks = defaultdict(dict)
with input_path.open(newline="") as source:
    for row in csv.DictReader(source):
        key = (row["game_id"], row["play_id"], row["nfl_id"])
        input_tracks[key][int(row["frame_id"])] = (float(row["x"]), float(row["y"]))

output_first = {}
with output_path.open(newline="") as source:
    for row in csv.DictReader(source):
        if int(row["frame_id"]) == 1:
            key = (row["game_id"], row["play_id"], row["nfl_id"])
            output_first[key] = (float(row["x"]), float(row["y"]))

boundary_steps = []
for key, output_xy in output_first.items():
    if key in input_tracks:
        input_xy = input_tracks[key][max(input_tracks[key])]
        boundary_steps.append(math.dist(input_xy, output_xy))

boundary_continuity = {
    "matched player-play boundaries": len(boundary_steps),
    "median displacement (yd)": round(statistics.median(boundary_steps), 3),
    "90th percentile displacement (yd)": round(sorted(boundary_steps)[int(0.90 * (len(boundary_steps) - 1))], 3),
    "share at most 2 yd": round(sum(distance <= 2 for distance in boundary_steps) / len(boundary_steps), 3),
}
boundary_continuity

# %% [markdown]
# ## Assessment
#
# The evidence supports treating the 2023 pre-throw input as a **snap-to-release-like window**: the duration distribution is close to the 2021 event-anchored interval, initial receiver speeds are similarly near zero, and input/output positions are continuous across the throw boundary. The NFL/AWS description of the NGS pre-pass sequence also states the snap-to-release convention.
#
# The 2023 CSVs still lack per-frame event markers, so exact frame-by-frame equivalence cannot be demonstrated from these files alone. For cross-season testing, use the common window as the working alignment, document this provenance limitation, and keep a timing sensitivity check in the results. The transfer result is now better supported than a generic first-to-last-window comparison, though it is not an event-label-verified replication.
