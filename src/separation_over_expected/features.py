from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .utils import fmt, normalize_xy


ROUTE_TABLE_COLUMNS = [
    "gameId",
    "playId",
    "nflId",
    "displayName",
    "officialPosition",
    "pff_positionLinedUp",
    "week",
    "down",
    "yardsToGo",
    "absoluteYardlineNumber",
    "offenseFormation",
    "personnelO",
    "personnelD",
    "defendersInBox",
    "dropBackType",
    "pff_playAction",
    "pff_passCoverage",
    "pff_passCoverageType",
    "passResult",
    "playDirection",
    "snap_frame",
    "release_frame",
    "time_to_throw_frames",
    "snap_x",
    "snap_y",
    "release_x",
    "release_y",
    "snap_x_norm",
    "snap_y_norm",
    "release_x_norm",
    "release_y_norm",
    "route_depth",
    "route_width_change",
    "receiver_speed_snap",
    "receiver_accel_snap",
    "receiver_speed_release",
    "receiver_accel_release",
    "nearest_defender_snap_id",
    "nearest_defender_snap_name",
    "nearest_defender_release_id",
    "nearest_defender_release_name",
    "sep_snap",
    "sep_release",
    "delta_sep",
    "nearest_defender_dx_snap",
    "nearest_defender_dy_snap",
    "nearest_defender_dx_release",
    "nearest_defender_dy_release",
    "defender_1_dist_snap",
    "defender_2_dist_snap",
    "defender_3_dist_snap",
    "defenders_within_3_snap",
    "defenders_within_5_snap",
    "defenders_within_10_snap",
    "nearest_db_dist_snap",
    "nearest_lb_dist_snap",
    "nearest_defender_depth_leverage_snap",
    "nearest_defender_width_leverage_snap",
    "defender_depth_density_0_10_snap",
    "defender_inside_count_5_snap",
    "defender_outside_count_5_snap",
]


def build_route_table(data_dir: Path, output_path: Path, weeks: list[int]) -> None:
    plays = read_plays(data_dir / "plays.csv")
    players = read_players(data_dir / "players.csv")
    routes_by_play, coverage_by_play = read_pff_roles(data_dir / "pffScoutingData.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_plays = 0
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROUTE_TABLE_COLUMNS)
        writer.writeheader()
        for week in weeks:
            week_path = data_dir / f"week{week}.csv"
            if not week_path.exists():
                raise FileNotFoundError(f"Missing tracking file: {week_path}")

            event_frames = find_event_frames(week_path)
            eligible_keys = {
                key
                for key, frames in event_frames.items()
                if "snap" in frames
                and "release" in frames
                and key in routes_by_play
                and key in coverage_by_play
            }
            frame_positions = read_selected_frames(week_path, eligible_keys, event_frames)

            week_rows = 0
            for key in sorted(eligible_keys):
                play = plays.get(key)
                if play is None:
                    continue
                frames = event_frames[key]
                snap_frame = frames["snap"]
                release_frame = frames["release"]
                snap_positions = frame_positions.get((key[0], key[1], snap_frame), {})
                release_positions = frame_positions.get((key[0], key[1], release_frame), {})
                if not snap_positions or not release_positions:
                    continue

                play_direction = frame_play_direction(snap_positions)
                for route in routes_by_play[key]:
                    row = make_route_row(
                        key=key,
                        week=week,
                        route=route,
                        play=play,
                        players=players,
                        coverage_ids=coverage_by_play[key],
                        snap_frame=snap_frame,
                        release_frame=release_frame,
                        snap_positions=snap_positions,
                        release_positions=release_positions,
                        play_direction=play_direction,
                    )
                    if row is None:
                        continue
                    writer.writerow(row)
                    week_rows += 1
            total_rows += week_rows
            total_plays += len(eligible_keys)
            print(
                f"week {week}: {week_rows:,} route rows from {len(eligible_keys):,} eligible plays"
            )

    print(f"wrote {total_rows:,} route rows across {total_plays:,} eligible play-weeks")
    print(f"output: {output_path}")


def make_route_row(
    key: tuple[str, str],
    week: int,
    route: dict[str, str],
    play: dict[str, str],
    players: dict[str, dict[str, str]],
    coverage_ids: set[str],
    snap_frame: int,
    release_frame: int,
    snap_positions: dict[str, dict[str, float | str]],
    release_positions: dict[str, dict[str, float | str]],
    play_direction: str,
) -> dict[str, str] | None:
    rid = route["nflId"]
    if rid not in snap_positions or rid not in release_positions:
        return None

    snap = snap_positions[rid]
    release = release_positions[rid]
    snap_nearest = nearest_defender(snap, snap_positions, coverage_ids)
    release_nearest = nearest_defender(release, release_positions, coverage_ids)
    if snap_nearest is None or release_nearest is None:
        return None

    snap_norm = normalize_xy(float(snap["x"]), float(snap["y"]), play_direction)
    release_norm = normalize_xy(float(release["x"]), float(release["y"]), play_direction)
    snap_def_norm = normalize_xy(
        float(snap_nearest["position"]["x"]),
        float(snap_nearest["position"]["y"]),
        play_direction,
    )
    release_def_norm = normalize_xy(
        float(release_nearest["position"]["x"]),
        float(release_nearest["position"]["y"]),
        play_direction,
    )
    snap_context = coverage_context(
        receiver=snap,
        receiver_norm=snap_norm,
        positions=snap_positions,
        defender_ids=coverage_ids,
        players=players,
        play_direction=play_direction,
    )
    player = players.get(rid, {})
    return {
        "gameId": key[0],
        "playId": key[1],
        "nflId": rid,
        "displayName": player.get("displayName", ""),
        "officialPosition": player.get("officialPosition", ""),
        "pff_positionLinedUp": route["pff_positionLinedUp"],
        "week": str(week),
        "down": play["down"],
        "yardsToGo": play["yardsToGo"],
        "absoluteYardlineNumber": play["absoluteYardlineNumber"],
        "offenseFormation": play["offenseFormation"],
        "personnelO": play["personnelO"],
        "personnelD": play["personnelD"],
        "defendersInBox": play["defendersInBox"],
        "dropBackType": play["dropBackType"],
        "pff_playAction": play["pff_playAction"],
        "pff_passCoverage": play["pff_passCoverage"],
        "pff_passCoverageType": play["pff_passCoverageType"],
        "passResult": play["passResult"],
        "playDirection": play_direction,
        "snap_frame": str(snap_frame),
        "release_frame": str(release_frame),
        "time_to_throw_frames": str(release_frame - snap_frame),
        "snap_x": fmt(float(snap["x"])),
        "snap_y": fmt(float(snap["y"])),
        "release_x": fmt(float(release["x"])),
        "release_y": fmt(float(release["y"])),
        "snap_x_norm": fmt(snap_norm[0]),
        "snap_y_norm": fmt(snap_norm[1]),
        "release_x_norm": fmt(release_norm[0]),
        "release_y_norm": fmt(release_norm[1]),
        "route_depth": fmt(release_norm[0] - snap_norm[0]),
        "route_width_change": fmt(release_norm[1] - snap_norm[1]),
        "receiver_speed_snap": fmt(float(snap["s"])),
        "receiver_accel_snap": fmt(float(snap["a"])),
        "receiver_speed_release": fmt(float(release["s"])),
        "receiver_accel_release": fmt(float(release["a"])),
        "nearest_defender_snap_id": str(snap_nearest["nflId"]),
        "nearest_defender_snap_name": players.get(str(snap_nearest["nflId"]), {}).get("displayName", ""),
        "nearest_defender_release_id": str(release_nearest["nflId"]),
        "nearest_defender_release_name": players.get(str(release_nearest["nflId"]), {}).get("displayName", ""),
        "sep_snap": fmt(float(snap_nearest["distance"])),
        "sep_release": fmt(float(release_nearest["distance"])),
        "delta_sep": fmt(float(release_nearest["distance"]) - float(snap_nearest["distance"])),
        "nearest_defender_dx_snap": fmt(snap_def_norm[0] - snap_norm[0]),
        "nearest_defender_dy_snap": fmt(snap_def_norm[1] - snap_norm[1]),
        "nearest_defender_dx_release": fmt(release_def_norm[0] - release_norm[0]),
        "nearest_defender_dy_release": fmt(release_def_norm[1] - release_norm[1]),
        **snap_context,
    }


def read_plays(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as f:
        return {(row["gameId"], row["playId"]): row for row in csv.DictReader(f)}


def read_players(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["nflId"]: row for row in csv.DictReader(f)}


def read_pff_roles(
    path: Path,
) -> tuple[dict[tuple[str, str], list[dict[str, str]]], dict[tuple[str, str], set[str]]]:
    routes: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["gameId"], row["playId"])
            if row["pff_role"] == "Pass Route":
                routes[key].append(
                    {
                        "nflId": row["nflId"],
                        "pff_positionLinedUp": row["pff_positionLinedUp"],
                    }
                )
            elif row["pff_role"] == "Coverage":
                coverage[key].add(row["nflId"])
    return routes, coverage


def find_event_frames(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    event_frames: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            event = row["event"]
            if event not in {
                "ball_snap",
                "autoevent_ballsnap",
                "pass_forward",
                "autoevent_passforward",
            }:
                continue
            key = (row["gameId"], row["playId"])
            frame = int(row["frameId"])
            if event in {"ball_snap", "autoevent_ballsnap"}:
                event_frames[key]["snap"] = min(
                    frame, event_frames[key].get("snap", frame)
                )
            elif event in {"pass_forward", "autoevent_passforward"}:
                event_frames[key]["release"] = min(
                    frame, event_frames[key].get("release", frame)
                )
    return event_frames


def read_selected_frames(
    path: Path,
    eligible_keys: set[tuple[str, str]],
    event_frames: dict[tuple[str, str], dict[str, int]],
) -> dict[tuple[str, str, int], dict[str, dict[str, float | str]]]:
    selected = {
        (game_id, play_id, frames[event_name])
        for game_id, play_id in eligible_keys
        for event_name in ("snap", "release")
        for frames in [event_frames[(game_id, play_id)]]
    }
    positions: dict[tuple[str, str, int], dict[str, dict[str, float | str]]] = defaultdict(dict)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame_key = (row["gameId"], row["playId"], int(row["frameId"]))
            if frame_key not in selected or row["team"] == "football":
                continue
            positions[frame_key][row["nflId"]] = {
                "nflId": row["nflId"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "s": float(row["s"]),
                "a": float(row["a"]),
                "playDirection": row["playDirection"],
            }
    return positions


def nearest_defender(
    receiver: dict[str, float | str],
    positions: dict[str, dict[str, float | str]],
    defender_ids: set[str],
) -> dict[str, object] | None:
    nearest = None
    rx = float(receiver["x"])
    ry = float(receiver["y"])
    for defender_id in defender_ids:
        defender = positions.get(defender_id)
        if defender is None:
            continue
        distance = math.dist((rx, ry), (float(defender["x"]), float(defender["y"])))
        if nearest is None or distance < nearest["distance"]:
            nearest = {
                "nflId": defender_id,
                "distance": distance,
                "position": defender,
            }
    return nearest


def coverage_context(
    receiver: dict[str, float | str],
    receiver_norm: tuple[float, float],
    positions: dict[str, dict[str, float | str]],
    defender_ids: set[str],
    players: dict[str, dict[str, str]],
    play_direction: str,
) -> dict[str, str]:
    rx = float(receiver["x"])
    ry = float(receiver["y"])
    defender_rows = []
    for defender_id in defender_ids:
        defender = positions.get(defender_id)
        if defender is None:
            continue
        defender_norm = normalize_xy(
            float(defender["x"]),
            float(defender["y"]),
            play_direction,
        )
        dx = defender_norm[0] - receiver_norm[0]
        dy = defender_norm[1] - receiver_norm[1]
        distance = math.dist((rx, ry), (float(defender["x"]), float(defender["y"])))
        official_position = players.get(defender_id, {}).get("officialPosition", "")
        defender_rows.append(
            {
                "nflId": defender_id,
                "distance": distance,
                "dx": dx,
                "dy": dy,
                "officialPosition": official_position,
            }
        )
    defender_rows.sort(key=lambda row: row["distance"])

    nearest = defender_rows[0] if defender_rows else {"distance": 0.0, "dx": 0.0, "dy": 0.0}
    distance_at = lambda idx: defender_rows[idx]["distance"] if len(defender_rows) > idx else 0.0
    nearest_db = nearest_position_group_distance(defender_rows, {"CB", "DB", "FS", "SS"})
    nearest_lb = nearest_position_group_distance(defender_rows, {"LB", "ILB", "MLB", "OLB"})
    close_rows = [row for row in defender_rows if row["distance"] <= 5.0]

    return {
        "defender_1_dist_snap": fmt(distance_at(0)),
        "defender_2_dist_snap": fmt(distance_at(1)),
        "defender_3_dist_snap": fmt(distance_at(2)),
        "defenders_within_3_snap": str(sum(row["distance"] <= 3.0 for row in defender_rows)),
        "defenders_within_5_snap": str(len(close_rows)),
        "defenders_within_10_snap": str(sum(row["distance"] <= 10.0 for row in defender_rows)),
        "nearest_db_dist_snap": fmt(nearest_db),
        "nearest_lb_dist_snap": fmt(nearest_lb),
        "nearest_defender_depth_leverage_snap": fmt(float(nearest["dx"])),
        "nearest_defender_width_leverage_snap": fmt(float(nearest["dy"])),
        "defender_depth_density_0_10_snap": str(
            sum(0.0 <= row["dx"] <= 10.0 for row in defender_rows)
        ),
        "defender_inside_count_5_snap": str(sum(row["dy"] < 0 for row in close_rows)),
        "defender_outside_count_5_snap": str(sum(row["dy"] > 0 for row in close_rows)),
    }


def nearest_position_group_distance(
    defender_rows: list[dict[str, object]],
    positions: set[str],
) -> float:
    for row in defender_rows:
        if str(row["officialPosition"]) in positions:
            return float(row["distance"])
    return 0.0


def frame_play_direction(positions: dict[str, dict[str, float | str]]) -> str:
    first = next(iter(positions.values()))
    return str(first["playDirection"])
