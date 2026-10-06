import pytest

from separation_over_expected.models import RidgeContextModel
from separation_over_expected.ridge_comparison import compare_ridge_specifications


class TinyRidge(RidgeContextModel):
    numeric_features = ["x"]
    categorical_features = ["group"]


def _row(week, game, player, index):
    row = {
        "gameId": f"g-{week}-{game}",
        "playId": str(player),
        "nflId": str(player),
        "displayName": f"Receiver {player}",
        "officialPosition": "WR",
        "week": str(week),
        "delta_sep": f"{(week + game + player) / 10:.3f}",
        "group": f"group-{player % 2}",
    }
    row["x"] = f"{index + player / 10:.3f}"
    return row


def _full_feature_row(week, game, player):
    from separation_over_expected.models import (
        RidgeContextModel,
        RidgeDynamicContextModel,
        RidgePocketContextModel,
        RidgePressureContextModel,
    )

    models = (
        RidgeContextModel,
        RidgeDynamicContextModel,
        RidgePressureContextModel,
        RidgePocketContextModel,
    )
    row = _row(week, game, player, week * 10 + game)
    for model in models:
        for feature in model.numeric_features:
            row.setdefault(feature, f"{week + game + player / 10:.3f}")
        for feature in model.categorical_features:
            row.setdefault(feature, f"level-{(week + game + player) % 3}")
    return row


def test_ridge_penalty_refit_matches_a_fresh_fit():
    rows = [
        {"x": str(value), "group": f"g{value % 2}", "delta_sep": str(value ** 2 / 10)}
        for value in range(1, 9)
    ]
    refitted = TinyRidge(l2=2.0, max_levels_per_feature=30)
    refitted.fit(rows)
    refitted.refit_l2(25.0)
    fresh = TinyRidge(l2=25.0, max_levels_per_feature=30)
    fresh.fit(rows)

    assert [refitted.predict(row) for row in rows] == pytest.approx(
        [fresh.predict(row) for row in rows]
    )
    with pytest.raises(ValueError):
        refitted.refit_l2(-1)


def test_ridge_grid_uses_same_game_folds_and_returns_all_diagnostics():
    rows = [
        _full_feature_row(week, game, player)
        for week in (1, 2)
        for game in range(5)
        for player in range(1, 4)
    ]

    predictions, metrics, folds, deciles, player_halves, reliability = (
        compare_ridge_specifications(
            rows,
            l2_values=(5.0, 25.0),
            n_folds=5,
            min_routes_per_half=1,
            bootstrap_samples=20,
            seed=11,
        )
    )

    assert len(predictions) == len(rows)
    assert len(metrics) == 8
    assert len(folds) == 5 * 8
    assert len(deciles) == 10 * 8
    assert len(player_halves) == 3
    assert len(reliability) == 8
    assert all(row["rmse_delta_upper_95"] for row in metrics)
    assert {row["candidate"] for row in reliability} == {
        row["candidate"] for row in metrics
    }
