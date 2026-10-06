import csv

from separation_over_expected.cross_season import (
    load_cross_season_wr_rows,
    paired_game_bootstrap_rmse_delta,
    yardline_100_from_tracking,
)


def test_yardline_conversion_respects_offensive_direction():
    assert yardline_100_from_tracking(42.0, "right") == 68.0
    assert yardline_100_from_tracking(42.0, "left") == 32.0


def test_game_bootstrap_compares_paired_rmse_by_whole_game():
    result = paired_game_bootstrap_rmse_delta(
        actual=[1.0, -1.0, 2.0, -2.0],
        static_predictions=[0.0, 0.0, 0.0, 0.0],
        dynamic_predictions=[1.0, -1.0, 2.0, -2.0],
        game_ids=["g1", "g1", "g2", "g2"],
        iterations=100,
        seed=3,
    )
    assert result["point_delta"] < 0
    assert result["upper_95"] < 0
    assert result["bootstrap_share_dynamic_better"] == 1.0


def test_load_cross_season_rows_joins_pbp_context_and_maps_tracking_fields(tmp_path):
    legacy_path = tmp_path / "legacy.csv"
    with legacy_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "gameId",
                "playId",
                "week",
                "nflId",
                "officialPosition",
                "absoluteYardlineNumber",
                "playDirection",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "gameId": "2021090900",
                "playId": "97",
                "week": "1",
                "nflId": "10",
                "officialPosition": "WR",
                "absoluteYardlineNumber": "43",
                "playDirection": "right",
            }
        )

    bdb_path = tmp_path / "bdb.csv"
    with bdb_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "gameId",
                "playId",
                "week",
                "nflId",
                "officialPosition",
                "absoluteYardlineNumber",
                "playDirection",
                "delta_sep_input_window",
                "sep_first_input",
                "first_x_norm",
                "first_y_norm",
                "nearest_defender_dx_first",
                "nearest_defender_dy_first",
                "receiver_speed_first",
                "receiver_accel_first",
                "input_window_frames",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "gameId": "2023090700",
                "playId": "101",
                "week": "1",
                "nflId": "20",
                "officialPosition": "WR",
                "absoluteYardlineNumber": "42",
                "playDirection": "right",
                "delta_sep_input_window": "-2.5",
                "sep_first_input": "4.0",
                "first_x_norm": "40.0",
                "first_y_norm": "20.0",
                "nearest_defender_dx_first": "1.0",
                "nearest_defender_dy_first": "2.0",
                "receiver_speed_first": "0.5",
                "receiver_accel_first": "0.1",
                "input_window_frames": "24",
            }
        )

    pbp_path = tmp_path / "pbp.csv"
    with pbp_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["old_game_id", "play_id", "down", "ydstogo", "yardline_100"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "old_game_id": "2023090700",
                "play_id": "101.0",
                "down": "3",
                "ydstogo": "3",
                "yardline_100": "68",
            }
        )

    legacy, bdb2023, diagnostics = load_cross_season_wr_rows(
        legacy_path, bdb_path, pbp_path
    )

    assert legacy[0]["yardline_100"] == "67.000"
    assert bdb2023[0]["delta_sep"] == "-2.5"
    assert bdb2023[0]["down"] == "3"
    assert bdb2023[0]["yardsToGo"] == "3"
    assert bdb2023[0]["yardline_100"] == "68.000"
    assert diagnostics["bdb2023_plays_without_context"] == 0
    assert diagnostics["yardline_max_absolute_error"] == 0
