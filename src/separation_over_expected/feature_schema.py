from __future__ import annotations

DYNAMIC_DEFENDER_RANKS = (1, 2, 3)
DYNAMIC_DEFENDER_SUMMARIES = (
    "mean_speed",
    "max_speed",
    "mean_accel",
    "depth_displacement",
    "width_displacement",
    "path_length",
    "total_turn_degrees",
    "observed_fraction",
)
DYNAMIC_CONTEXT_FEATURES = tuple(
    f"defender_{rank}_{summary}_pre_release"
    for rank in DYNAMIC_DEFENDER_RANKS
    for summary in DYNAMIC_DEFENDER_SUMMARIES
)
