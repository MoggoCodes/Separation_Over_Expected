import pytest

from separation_over_expected.within_season_reliability import (
    evaluate_within_season_receiver_reliability,
)


def _predictions():
    rows = []
    for week in range(1, 13):
        for game in range(10):
            game_id = f"2023-{week}-{game}"
            for player in range(6):
                for route in range(3):
                    rows.append({
                        "season": "2023",
                        "week": str(week),
                        "gameId": game_id,
                        "playId": f"{game}-{player}-{route}",
                        "nflId": str(player),
                        "displayName": f"Receiver {player}",
                        "residual_dynamic": str(player * 0.4 + game * 0.002 + route * 0.001),
                    })
    rows.append({
        "season": "2021", "week": "1", "gameId": "old", "playId": "old",
        "nflId": "0", "displayName": "Old season", "residual_dynamic": "999",
    })
    return rows


def test_within_season_reliability_splits_games_and_excludes_other_seasons():
    split_rows, summary_rows = evaluate_within_season_receiver_reliability(
        _predictions(), cohorts=((10, 3), (20, 5)), split_seeds=4, seed=17
    )

    assert len(split_rows) == 8
    assert all(row["season"] == "2023" for row in split_rows)
    assert {row["seed"] for row in split_rows} == {str(10017 + i * 37) for i in range(4)}
    assert {row["eligible_receivers"] for row in split_rows} == {"6"}
    assert len(summary_rows) == 2
    assert all(int(row["valid_splits"]) == 4 for row in summary_rows)
    assert all(float(row["pearson_median"]) > 0.99 for row in summary_rows)
    assert all(float(row["spearman_median"]) > 0.99 for row in summary_rows)
    assert all(float(row["pearson_bootstrap_ci_lower"]) <= float(row["pearson_bootstrap_ci_upper"]) for row in summary_rows)
    assert all(float(row["spearman_bootstrap_ci_lower"]) <= float(row["spearman_bootstrap_ci_upper"]) for row in summary_rows)


def test_within_season_reliability_rejects_duplicate_routes_and_empty_season():
    rows = _predictions()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_within_season_receiver_reliability(rows + [rows[0]], split_seeds=1)
    with pytest.raises(ValueError, match="No 2023"):
        evaluate_within_season_receiver_reliability([rows[-1]], split_seeds=1)


def test_within_season_reliability_requires_positive_split_and_cutoffs():
    with pytest.raises(ValueError, match="split_seeds"):
        evaluate_within_season_receiver_reliability(_predictions(), split_seeds=0)
    with pytest.raises(ValueError, match="cohorts"):
        evaluate_within_season_receiver_reliability(_predictions(), cohorts=((0, 3),))
    with pytest.raises(ValueError, match="bootstrap_samples"):
        evaluate_within_season_receiver_reliability(_predictions(), bootstrap_samples=0)
