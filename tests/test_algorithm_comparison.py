from separation_over_expected.algorithm_comparison import compare_algorithms
from separation_over_expected.models import RidgeDynamicContextModel


def test_algorithm_comparison_uses_same_game_folds_for_each_model():
    rows = []
    for week in (1, 2):
        for game in range(4):
            game_id = f"{week}-{game}"
            for receiver in range(4):
                for route in range(3):
                    row = {
                        name: str(0.1 * (route + receiver + game))
                        for name in RidgeDynamicContextModel.numeric_features
                    }
                    row.update({name: f"level-{(game + receiver) % 2}"
                                for name in RidgeDynamicContextModel.categorical_features})
                    row.update({
                        "gameId": game_id, "playId": str(route), "nflId": str(receiver),
                        "displayName": f"receiver-{receiver}", "officialPosition": "WR",
                        "week": str(week), "delta_sep": str((receiver - 1.5) * 0.5 + route * 0.1),
                    })
                    rows.append(row)

    predictions, metrics, folds, deciles, halves, reliability = compare_algorithms(
        rows, n_folds=2, min_routes_per_half=1, bootstrap_samples=20, seed=7,
    )

    assert len(predictions) == len(rows)
    assert {row["model"] for row in metrics} == {
        "ridge_dynamic_context", "extra_trees", "hist_gradient_boosting",
    }
    assert {row["model"] for row in folds} == {
        "ridge_dynamic_context", "extra_trees", "hist_gradient_boosting",
    }
    assert len(deciles) == 30
    assert len(reliability) == 3
    assert all(row["fold"] in {"1", "2"} for row in predictions)
