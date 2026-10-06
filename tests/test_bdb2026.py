import csv

from separation_over_expected.bdb2026 import build_bdb2026_route_table


def _record(frame, nfl_id, role, x, y, position="WR"):
    return {
        "game_id": "2023090700",
        "play_id": "101",
        "player_to_predict": "False",
        "nfl_id": str(nfl_id),
        "frame_id": str(frame),
        "play_direction": "right",
        "absolute_yardline_number": "42",
        "player_name": f"Player {nfl_id}",
        "player_height": "6-0",
        "player_weight": "200",
        "player_birth_date": "2000-01-01",
        "player_position": position,
        "player_side": "Defense" if role == "Defensive Coverage" else "Offense",
        "player_role": role,
        "x": str(x),
        "y": str(y),
        "s": "1.0",
        "a": "0.2",
        "dir": "90",
        "o": "90",
        "num_frames_output": "5",
        "ball_land_x": "80",
        "ball_land_y": "25",
    }


def test_build_bdb2026_route_table_uses_all_route_runners_and_endpoint_separation(tmp_path):
    data_dir = tmp_path / "bdb2026"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    input_path = train_dir / "input_2023_w01.csv"
    fields = list(_record(1, 1, "Targeted Receiver", 10, 10))
    rows = [
        _record(1, 1, "Targeted Receiver", 10, 10),
        _record(1, 2, "Other Route Runner", 20, 20, "TE"),
        _record(1, 3, "Defensive Coverage", 11, 10, "CB"),
        _record(1, 4, "Defensive Coverage", 20, 21, "CB"),
        _record(3, 1, "Targeted Receiver", 14, 10),
        _record(3, 2, "Other Route Runner", 20, 19, "TE"),
        _record(3, 3, "Defensive Coverage", 15, 10, "CB"),
        _record(3, 4, "Defensive Coverage", 20, 20, "CB"),
    ]
    with input_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "routes.csv"
    report = build_bdb2026_route_table(data_dir, output)
    with output.open(newline="") as stream:
        result = list(csv.DictReader(stream))

    assert report["route_rows"] == 2
    assert {row["player_role"] for row in result} == {
        "Targeted Receiver",
        "Other Route Runner",
    }
    targeted = next(row for row in result if row["targeted"] == "true")
    assert targeted["sep_first_input"] == "1.000"
    assert targeted["sep_last_input"] == "1.000"
    assert targeted["delta_sep_input_window"] == "0.000"
    assert targeted["last_input_frame"] == "3"
    assert "ball_land_x" not in result[0]
