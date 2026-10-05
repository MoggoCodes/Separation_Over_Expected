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

POCKET_CONTEXT_FEATURES = (
    "qb_depth_drop_pre_release",
    "qb_lateral_drift_pre_release",
    "qb_path_length_pre_release",
    "qb_mean_speed_pre_release",
    "qb_max_speed_pre_release",
    "qb_mean_accel_pre_release",
    "qb_nearest_rusher_dist_snap",
    "qb_nearest_rusher_min_dist_pre_release",
    "qb_nearest_rusher_mean_dist_pre_release",
    "qb_rusher_closing_rate_pre_release",
    "qb_rusher_within_3yd_frame_share_pre_release",
    "qb_rusher_within_5yd_frame_share_pre_release",
    "qb_pressure_observed_fraction_pre_release",
    "pff_pass_rusher_count",
)

PRESSURE_CONTEXT_FEATURES = (
    "qb_nearest_rusher_dist_snap",
    "qb_nearest_rusher_min_dist_pre_release",
    "qb_nearest_rusher_mean_dist_pre_release",
    "qb_rusher_closing_rate_pre_release",
    "qb_rusher_within_3yd_frame_share_pre_release",
    "qb_rusher_within_5yd_frame_share_pre_release",
)

ROUTE_GEOMETRY_FEATURES = (
    "route_path_length_pre_release",
    "route_chord_length_pre_release",
    "route_directness_pre_release",
    "route_depth_excursion_max_pre_release",
    "route_depth_excursion_min_pre_release",
    "route_lateral_excursion_max_pre_release",
    "route_lateral_excursion_min_pre_release",
    "route_cumulative_turn_degrees_pre_release",
    "route_max_turn_degrees_pre_release",
    "route_max_chord_deviation_pre_release",
)
ROUTE_SHAPE_COLUMN = "route_shape_points_pre_release"
