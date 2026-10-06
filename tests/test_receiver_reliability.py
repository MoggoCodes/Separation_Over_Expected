from separation_over_expected.receiver_reliability import (
    cross_season_receiver_predictions,
    summarize_cross_season_receivers,
)
from separation_over_expected.feature_schema import DYNAMIC_CONTEXT_FEATURES


def _route(season, week, game, player):
    value = float(week + game + player)
    row = {
        "gameId": f"{season}-{week}-{game}",
        "playId": str(player),
        "week": str(week),
        "nflId": str(player),
        "displayName": f"Receiver {player}",
        "delta_sep": f"{value / 10.0:.3f}",
        "sep_snap": f"{value:.3f}",
        "snap_x_norm": f"{value + 1:.3f}",
        "snap_y_norm": f"{value + 2:.3f}",
        "nearest_defender_dx_snap": f"{value + 3:.3f}",
        "nearest_defender_dy_snap": f"{value + 4:.3f}",
        "receiver_speed_snap": f"{value / 5:.3f}",
        "receiver_accel_snap": f"{value / 7:.3f}",
        "time_to_throw_frames": f"{value + 10:.3f}",
        "down": str(1 + game % 4),
        "yardsToGo": str(1 + player),
        "yardline_100": f"{value + 20:.3f}",
    }
    row.update({feature: f"{value + i / 10:.3f}" for i, feature in enumerate(DYNAMIC_CONTEXT_FEATURES)})
    return row


def test_cross_season_predictions_hold_out_whole_2021_games():
    legacy = [
        _route("2021", week, game, player)
        for week in (1, 2)
        for game in range(4)
        for player in range(1, 4)
    ]
    bdb2023 = [
        _route("2023", week, game, player)
        for week in (1, 2)
        for game in range(2)
        for player in range(1, 4)
    ]

    predictions = cross_season_receiver_predictions(
        legacy, bdb2023, n_folds=2, seed=19
    )

    legacy_predictions = [row for row in predictions if row["season"] == "2021"]
    folds_by_game = {}
    for row in legacy_predictions:
        folds_by_game.setdefault((row["week"], row["gameId"]), set()).add(row["fold"])
    assert len(predictions) == len(legacy) + len(bdb2023)
    assert len({(r["season"], r["gameId"], r["playId"], r["nflId"]) for r in predictions}) == len(predictions)
    assert all(len(folds) == 1 for folds in folds_by_game.values())
    assert {row["fold"] for row in legacy_predictions} == {"1", "2"}
    assert {row["fold"] for row in predictions if row["season"] == "2023"} == {"ALL_2021"}


def test_cross_season_receiver_summary_centers_season_bias_and_pairs_players():
    predictions = []
    for season, year_offset in (("2021", -3.0), ("2023", 5.0)):
        for player_index in range(6):
            for game_index in range(5):
                row = {
                    "season": season,
                    "gameId": f"{season}-{game_index}",
                    "playId": str(game_index),
                    "week": str(game_index + 1),
                    "nflId": str(player_index),
                    "displayName": f"Receiver {player_index}",
                }
                for model_name, scale in (("static", 1.0), ("dynamic", 1.2)):
                    row[f"residual_{model_name}"] = str(
                        year_offset + player_index * scale + game_index * 0.01
                    )
                predictions.append(row)

    receivers, metrics, diagnostics = summarize_cross_season_receivers(
        predictions,
        min_routes=5,
        min_games=5,
        bootstrap_samples=100,
        seed=7,
    )

    assert diagnostics["shared_player_ids"] == 6
    assert diagnostics["eligible_shared_players"] == 6
    assert all(row["eligible"] == "true" for row in receivers)
    assert all(abs(float(row["centered_residual_2021_static"])) < 3.0 for row in receivers)
    assert float(metrics[0]["pearson"]) > 0.99
    assert metrics[-1]["comparison"] == "dynamic_minus_static"
    assert float(receivers[0]["ci_lower_2021_static"]) < float(
        receivers[0]["centered_residual_2021_static"]
    )
    assert float(receivers[0]["centered_residual_2021_static"]) < float(
        receivers[0]["ci_upper_2021_static"]
    )


def test_cross_season_receiver_summary_excludes_short_or_few_game_players():
    predictions = []
    for season in ("2021", "2023"):
        for game_index in range(5):
            predictions.append(
                {
                    "season": season,
                    "gameId": f"{season}-{game_index}",
                    "playId": str(game_index),
                    "week": str(game_index + 1),
                    "nflId": "short-sample",
                    "displayName": "Short Sample",
                    "residual_static": "0.2",
                    "residual_dynamic": "0.3",
                }
            )
    _, metrics, diagnostics = summarize_cross_season_receivers(
        predictions,
        min_routes=6,
        min_games=5,
        bootstrap_samples=20,
    )
    assert diagnostics["shared_player_ids"] == 1
    assert diagnostics["eligible_shared_players"] == 0
    assert metrics == []
