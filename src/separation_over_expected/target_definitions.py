from __future__ import annotations

import math


def _distance(a: dict[str, float | str], b: dict[str, float | str]) -> float:
    return math.dist((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])))


def compute_separation_targets(
    receiver_snap: dict[str, float | str],
    receiver_release: dict[str, float | str],
    snap_positions: dict[str, dict[str, float | str]],
    release_positions: dict[str, dict[str, float | str]],
    coverage_ids: set[str],
) -> dict[str, str]:
    """Return endpoint-nearest, snap-anchor, and snap-top-three separation changes."""
    at_snap = [
        (defender_id, _distance(receiver_snap, snap_positions[defender_id]))
        for defender_id in coverage_ids
        if defender_id in snap_positions
    ]
    at_snap.sort(key=lambda pair: (pair[1], pair[0]))
    at_release = [
        (defender_id, _distance(receiver_release, release_positions[defender_id]))
        for defender_id in coverage_ids
        if defender_id in release_positions
    ]
    at_release.sort(key=lambda pair: (pair[1], pair[0]))

    if not at_snap or not at_release:
        raise ValueError("Both endpoints need at least one observed coverage defender")

    snap_nearest_id, snap_nearest_distance = at_snap[0]
    release_nearest_id, release_nearest_distance = at_release[0]
    anchor_release = next(
        (distance for defender_id, distance in at_release if defender_id == snap_nearest_id),
        None,
    )
    snap_cohort = at_snap[:3]
    cohort_release = {
        defender_id: distance
        for defender_id, distance in at_release
        if defender_id in {candidate for candidate, _ in snap_cohort}
    }
    top3_valid = len(snap_cohort) == 3 and len(cohort_release) == 3

    return {
        "delta_sep_endpoint_nearest": f"{release_nearest_distance - snap_nearest_distance:.3f}",
        "delta_sep_snap_anchor": (
            f"{anchor_release - snap_nearest_distance:.3f}" if anchor_release is not None else ""
        ),
        "delta_sep_snap_top3": (
            f"{min(cohort_release.values()) - min(distance for _, distance in snap_cohort):.3f}"
            if top3_valid
            else ""
        ),
        "nearest_defender_switched": "1" if snap_nearest_id != release_nearest_id else "0",
        "snap_top3_release_count": str(len(cohort_release)),
    }
