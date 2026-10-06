from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .feature_schema import DYNAMIC_CONTEXT_FEATURES, DYNAMIC_DEFENDER_RANKS, DYNAMIC_DEFENDER_SUMMARIES
from .utils import fmt, normalize_xy


OUTPUT_COLUMNS = [
    "gameId",
    "playId",
    "week",
    "nflId",
    "displayName",
    "officialPosition",
    "player_role",
    "targeted",
    "playDirection",
    "absoluteYardlineNumber",
    "first_input_frame",
    "last_input_frame",
    "input_window_frames",
    "first_x",
    "first_y",
    "first_x_norm",
    "first_y_norm",
    "last_x",
    "last_y",
    "last_x_norm",
    "last_y_norm",
    "receiver_speed_first",
    "receiver_accel_first",
    "sep_first_input",
    "sep_last_input",
    "delta_sep_input_window",
    "nearest_defender_first_id",
    "nearest_defender_last_id",
    "nearest_defender_dx_first",
    "nearest_defender_dy_first",
]

ROUTE_ROLES = {"Targeted Receiver", "Other Route Runner"}


def _float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", "NA"} else 0.0


def _snapshot(row: dict[str, str]) -> dict[str, Any]:
    return {
        "frame_id": int(row["frame_id"]),
        "nfl_id": row["nfl_id"],
        "player_name": row["player_name"],
        "player_position": row["player_position"],
        "player_side": row["player_side"],
        "player_role": row["player_role"],
        "x": _float(row, "x"),
        "y": _float(row, "y"),
        "s": _float(row, "s"),
        "a": _float(row, "a"),
        "dir": _float(row, "dir"),
    }


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _nearest(receiver: dict[str, Any], defenders: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    defender = min(defenders, key=lambda item: _distance(receiver, item))
    return defender, _distance(receiver, defender)


def _read_week(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read a week while retaining only each player's first and latest input frame."""
    plays: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["game_id"], row["play_id"])
            frame_id = int(row["frame_id"])
            play = plays.setdefault(
                key,
                {
                    "max_frame": 0,
                    "direction": row["play_direction"],
                    "yardline": row["absolute_yardline_number"],
                    "players": {},
                },
            )
            play["max_frame"] = max(play["max_frame"], frame_id)
            player = play["players"].setdefault(
                row["nfl_id"], {"first": None, "last": None, "last_frame": -1}
            )
            if frame_id == 1:
                player["first"] = _snapshot(row)
            if frame_id > player["last_frame"]:
                player["last"] = _snapshot(row)
                player["last_frame"] = frame_id
    return plays


def _route_rows(
    plays: dict[tuple[str, str], dict[str, Any]], week: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    diagnostics = defaultdict(int)
    for (game_id, play_id), play in plays.items():
        max_frame = play["max_frame"]
        participants = list(play["players"].values())
        defenders_start = [
            p["first"]
            for p in participants
            if p["first"]
            and p["first"]["player_role"] == "Defensive Coverage"
        ]
        defenders_end = [
            p["last"]
            for p in participants
            if p["last"]
            and p["last"]["player_role"] == "Defensive Coverage"
            and p["last_frame"] == max_frame
        ]
        if not defenders_start:
            diagnostics["plays_missing_start_coverage"] += 1
        if not defenders_end:
            diagnostics["plays_missing_release_coverage"] += 1

        for participant in participants:
            first, last = participant["first"], participant["last"]
            role = (first or last or {}).get("player_role", "")
            if role not in ROUTE_ROLES:
                continue
            diagnostics["route_candidates"] += 1
            if first is None:
                diagnostics["routes_missing_first_frame"] += 1
                continue
            if last is None or participant["last_frame"] != max_frame:
                diagnostics["routes_missing_release_frame"] += 1
                continue
            if not defenders_start or not defenders_end:
                diagnostics["routes_missing_coverage_context"] += 1
                continue

            start_defender, sep_snap = _nearest(first, defenders_start)
            end_defender, sep_release = _nearest(last, defenders_end)
            direction = play["direction"]
            snap_x_norm, snap_y_norm = normalize_xy(first["x"], first["y"], direction)
            release_x_norm, release_y_norm = normalize_xy(last["x"], last["y"], direction)
            defender_x_norm, defender_y_norm = normalize_xy(
                start_defender["x"], start_defender["y"], direction
            )
            rows.append(
                {
                    "gameId": game_id,
                    "playId": play_id,
                    "week": str(week),
                    "nflId": first["nfl_id"],
                    "displayName": first["player_name"],
                    "officialPosition": first["player_position"],
                    "player_role": role,
                    "targeted": str(role == "Targeted Receiver").lower(),
                    "playDirection": direction,
                    "absoluteYardlineNumber": play["yardline"],
                    "first_input_frame": "1",
                    "last_input_frame": str(max_frame),
                    "input_window_frames": str(max_frame - 1),
                    "first_x": fmt(first["x"]),
                    "first_y": fmt(first["y"]),
                    "first_x_norm": fmt(snap_x_norm),
                    "first_y_norm": fmt(snap_y_norm),
                    "last_x": fmt(last["x"]),
                    "last_y": fmt(last["y"]),
                    "last_x_norm": fmt(release_x_norm),
                    "last_y_norm": fmt(release_y_norm),
                    "receiver_speed_first": fmt(first["s"]),
                    "receiver_accel_first": fmt(first["a"]),
                    "sep_first_input": fmt(sep_snap),
                    "sep_last_input": fmt(sep_release),
                    "delta_sep_input_window": fmt(sep_release - sep_snap),
                    "nearest_defender_first_id": start_defender["nfl_id"],
                    "nearest_defender_last_id": end_defender["nfl_id"],
                    "nearest_defender_dx_first": fmt(defender_x_norm - snap_x_norm),
                    "nearest_defender_dy_first": fmt(defender_y_norm - snap_y_norm),
                }
            )
    return rows, dict(diagnostics)


def build_bdb2026_route_table(
    data_dir: Path,
    output_path: Path,
    weeks: list[int] | None = None,
) -> dict[str, Any]:
    """Build endpoint-based route observations from BDB 2026 prediction inputs.

    The input tracks contain pre-throw frames but no named snap/release events.
    This first version therefore defines the measurement window as frame 1 through
    the last input frame and reports those frame indices explicitly. It does not
    label those endpoints as snap and release without an independent event crosswalk.
    """
    train_dir = data_dir / "train"
    if weeks is None:
        weeks = sorted(
            int(path.stem.split("_w")[-1])
            for path in train_dir.glob("input_2023_w*.csv")
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(int)
    written = 0
    with output_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for week in weeks:
            path = train_dir / f"input_2023_w{week:02d}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing tracking input: {path}")
            plays = _read_week(path)
            rows, diagnostics = _route_rows(plays, week)
            writer.writerows(rows)
            written += len(rows)
            for name, value in diagnostics.items():
                totals[name] += value
            totals["plays"] += len(plays)
            totals[f"week_{week:02d}_routes"] = len(rows)
    return {"weeks": len(weeks), "route_rows": written, **dict(totals)}


def _dynamic_features_for_route(
    route: dict[str, str],
    play: dict[str, Any],
) -> dict[str, str]:
    """Match the 2021 fixed-nearest-defender pre-release summaries on BDB inputs."""
    players = play["players"]
    receiver = players[route["nflId"]]["first"]
    direction = play["direction"]
    coverage = [
        player["first"]
        for player in players.values()
        if player["first"]
        and player["first"]["player_role"] == "Defensive Coverage"
    ]
    ranked = sorted(
        (
            math.hypot(defender["x"] - receiver["x"], defender["y"] - receiver["y"]),
            defender["nfl_id"],
        )
        for defender in coverage
    )
    last_frame = play["max_frame"]
    expected_frames = max(1, last_frame - 1)
    frames = play["frames"]
    features: dict[str, str] = {}

    for rank in DYNAMIC_DEFENDER_RANKS:
        prefix = f"defender_{rank}_"
        if len(ranked) < rank:
            features.update({
                f"{prefix}{summary}_pre_release": (
                    "0.000" if summary == "observed_fraction" else ""
                )
                for summary in DYNAMIC_DEFENDER_SUMMARIES
            })
            continue

        defender_id = ranked[rank - 1][1]
        samples = [
            (frame, frame_players[defender_id])
            for frame, frame_players in frames.items()
            if 1 <= frame < last_frame and defender_id in frame_players
        ]
        samples.sort(key=lambda item: item[0])
        observed_fraction = min(1.0, len(samples) / expected_frames)
        features[f"{prefix}observed_fraction_pre_release"] = fmt(observed_fraction)
        if not samples or observed_fraction < 0.8:
            for summary in DYNAMIC_DEFENDER_SUMMARIES[:-1]:
                features[f"{prefix}{summary}_pre_release"] = ""
            continue

        positions = [
            normalize_xy(sample["x"], sample["y"], direction)
            for _, sample in samples
        ]
        path_length = sum(
            math.dist(left, right)
            for left, right in zip(positions, positions[1:])
        )
        turn_degrees = sum(
            abs(((current["dir"] - previous["dir"] + 180) % 360) - 180)
            for (_, previous), (_, current) in zip(samples, samples[1:])
        )
        speeds = [sample["s"] for _, sample in samples]
        accelerations = [sample["a"] for _, sample in samples]
        features.update({
            f"{prefix}mean_speed_pre_release": fmt(statistics.fmean(speeds)),
            f"{prefix}max_speed_pre_release": fmt(max(speeds)),
            f"{prefix}mean_accel_pre_release": fmt(statistics.fmean(accelerations)),
            f"{prefix}depth_displacement_pre_release": fmt(positions[-1][0] - positions[0][0]),
            f"{prefix}width_displacement_pre_release": fmt(positions[-1][1] - positions[0][1]),
            f"{prefix}path_length_pre_release": fmt(path_length),
            f"{prefix}total_turn_degrees_pre_release": fmt(turn_degrees),
        })
    return features


def _iter_input_plays(path: Path):
    """Stream sorted tracking rows, yielding one play's full trajectory at a time."""
    current_key = None
    play: dict[str, Any] | None = None
    seen: set[tuple[str, str]] = set()
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["game_id"], row["play_id"])
            if key != current_key:
                if play is not None:
                    yield current_key, play
                    seen.add(current_key)
                if key in seen:
                    raise ValueError(f"Rows for play {key} are not contiguous in {path}")
                current_key = key
                play = {
                    "max_frame": 0,
                    "direction": row["play_direction"],
                    "yardline": row["absolute_yardline_number"],
                    "players": {},
                    "frames": defaultdict(dict),
                }

            assert play is not None
            snapshot = _snapshot(row)
            frame_id = snapshot["frame_id"]
            play["max_frame"] = max(play["max_frame"], frame_id)
            participant = play["players"].setdefault(
                snapshot["nfl_id"], {"first": None, "last": None, "last_frame": -1}
            )
            if frame_id == 1:
                participant["first"] = snapshot
            if frame_id > participant["last_frame"]:
                participant["last"] = snapshot
                participant["last_frame"] = frame_id
            play["frames"][frame_id][snapshot["nfl_id"]] = snapshot

    if play is not None:
        yield current_key, play


def build_bdb2026_dynamic_route_table(
    data_dir: Path,
    output_path: Path,
    weeks: list[int] | None = None,
) -> dict[str, Any]:
    """Build route rows with dynamic summaries for the three nearest tagged defenders.

    Defender identities are selected by distance at the first input frame and kept
    fixed through the pre-throw window. The final input frame is excluded from
    motion summaries, matching the 2021 dynamic feature convention.
    """
    train_dir = data_dir / "train"
    if weeks is None:
        weeks = sorted(
            int(path.stem.split("_w")[-1])
            for path in train_dir.glob("input_2023_w*.csv")
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = defaultdict(int)
    columns = [*OUTPUT_COLUMNS, *DYNAMIC_CONTEXT_FEATURES]
    with output_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for week in weeks:
            input_path = train_dir / f"input_2023_w{week:02d}.csv"
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing tracking input: {input_path}")
            week_rows = 0
            for key, play in _iter_input_plays(input_path):
                diagnostics["plays"] += 1
                rows, play_diagnostics = _route_rows({key: play}, week)
                for name, count in play_diagnostics.items():
                    diagnostics[name] += count
                for row in rows:
                    row.update(_dynamic_features_for_route(row, play))
                    writer.writerow(row)
                    week_rows += 1
            diagnostics[f"week_{week:02d}_routes"] = week_rows
            diagnostics["route_rows"] += week_rows
            print(f"week {week}: {week_rows:,} dynamic route rows")
    return {"weeks": len(weeks), **dict(diagnostics)}
