import pytest

from separation_over_expected.receiver_effects import evaluate_partial_pooled_receiver_effects


def _predictions():
    rows = []
    for week in range(1, 7):
        for game in range(10):
            game_id = f"2023-{week}-{game}"
            for player in range(5):
                for route in range(3):
                    residual = player * 0.25 + (game % 3) * 0.015 + route * 0.003
                    rows.append({
                        "season": "2023",
                        "week": str(week),
                        "gameId": game_id,
                        "playId": f"{player}-{route}",
                        "nflId": str(player),
                        "displayName": f"Receiver {player}",
                        "residual_dynamic": str(residual),
                    })
    rows.append({
        "season": "2021", "week": "1", "gameId": "old", "playId": "old",
        "nflId": "0", "displayName": "Old season", "residual_dynamic": "999",
    })
    return rows


def test_partial_pooled_receiver_effects_predict_heldout_games():
    split_rows, summaries, effects = evaluate_partial_pooled_receiver_effects(
        _predictions(), split_seeds=4, bootstrap_samples=30, seed=13
    )

    assert len(split_rows) == 8
    assert all(int(row["eligible_receivers"]) == 5 for row in split_rows)
    assert all(float(row["tau2"]) >= 0 for row in split_rows)
    assert len(summaries) == 17
    assert len(effects) == 5
    for row in effects:
        raw = float(row["training_unpooled_effect"])
        partial = float(row["training_partial_pool_effect"])
        shrinkage = float(row["shrinkage_factor"])
        assert 0 <= shrinkage <= 1
        assert abs(partial) <= abs(raw) + 1e-12
        assert float(row["posterior_ci_lower_plugin"]) <= partial
        assert partial <= float(row["posterior_ci_upper_plugin"])
    deltas = [row for row in summaries if row["metric"] == "rmse_delta_partial_minus_baseline"]
    assert len(deltas) == 2
    assert all(float(row["bootstrap_ci_lower"]) <= float(row["bootstrap_ci_upper"]) for row in deltas)


def test_partial_pooled_receiver_effects_rejects_invalid_inputs():
    rows = _predictions()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_partial_pooled_receiver_effects(rows + [rows[0]], split_seeds=1)
    with pytest.raises(ValueError, match="split_seeds"):
        evaluate_partial_pooled_receiver_effects(rows, split_seeds=0)
    with pytest.raises(ValueError, match="No 2023"):
        evaluate_partial_pooled_receiver_effects([rows[-1]], split_seeds=1)
