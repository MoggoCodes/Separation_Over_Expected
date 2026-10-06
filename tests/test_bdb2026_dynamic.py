import csv

from separation_over_expected.bdb2026 import build_bdb2026_dynamic_route_table
from separation_over_expected.feature_schema import DYNAMIC_CONTEXT_FEATURES
from separation_over_expected.reports import read_csv_rows


def test_dynamic_builder_uses_fixed_nearest_defenders_and_excludes_last_frame(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    input_path = train_dir / "input_2023_w01.csv"
    fields = [
        "game_id", "play_id", "player_to_predict", "nfl_id", "frame_id",
        "play_direction", "absolute_yardline_number", "player_name",
        "player_height", "player_weight", "player_birth_date", "player_position",
        "player_side", "player_role", "x", "y", "s", "a", "dir", "o",
        "num_frames_output", "ball_land_x", "ball_land_y",
    ]
    players = [
        ("10", "WR", "Offense", "Other Route Runner", 40.0, 20.0),
        ("20", "CB", "Defense", "Defensive Coverage", 42.0, 20.0),
        ("30", "CB", "Defense", "Defensive Coverage", 40.0, 24.0),
        ("40", "FS", "Defense", "Defensive Coverage", 40.0, 27.0),
    ]
    with input_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for nfl_id, position, side, role, x, y in players:
            for frame, speed, dx, heading in (
                (1, 1.0, 0.0, 0.0),
                (2, 2.0, 1.0, 10.0),
                (3, 9.0, 9.0, 180.0),
            ):
                direction = dx if nfl_id == "20" else 0.0
                writer.writerow({
                    "game_id": "2023090700",
                    "play_id": "101",
                    "player_to_predict": "False",
                    "nfl_id": nfl_id,
                    "frame_id": frame,
                    "play_direction": "right",
                    "absolute_yardline_number": "42",
                    "player_name": f"player-{nfl_id}",
                    "player_height": "6-0",
                    "player_weight": "200",
                    "player_birth_date": "2000-01-01",
                    "player_position": position,
                    "player_side": side,
                    "player_role": role,
                    "x": x + direction,
                    "y": y,
                    "s": speed if nfl_id == "20" else 0.5,
                    "a": speed / 10,
                    "dir": heading if nfl_id == "20" else 0.0,
                    "o": 0.0,
                    "num_frames_output": 5,
                    "ball_land_x": 60.0,
                    "ball_land_y": 20.0,
                })

    output_path = tmp_path / "route_dynamic.csv"
    diagnostics = build_bdb2026_dynamic_route_table(
        tmp_path, output_path, weeks=[1]
    )
    rows = read_csv_rows(output_path)

    assert diagnostics["route_rows"] == 1
    assert len(rows) == 1
    assert set(DYNAMIC_CONTEXT_FEATURES).issubset(rows[0])
    assert rows[0]["defender_1_mean_speed_pre_release"] == "1.500"
    assert rows[0]["defender_1_max_speed_pre_release"] == "2.000"
    assert rows[0]["defender_1_depth_displacement_pre_release"] == "1.000"
    assert rows[0]["defender_1_path_length_pre_release"] == "1.000"
    assert rows[0]["defender_1_total_turn_degrees_pre_release"] == "10.000"
    assert rows[0]["defender_1_observed_fraction_pre_release"] == "1.000"
