import pytest

from separation_over_expected.context_mix_audit import audit_context_mix


def _paired_rows():
    contexts, predictions = [], []
    for week in range(1, 7):
        for game in range(6):
            game_id = f"2023-{week}-{game}"
            for player in range(5):
                for route in range(4):
                    ytg = 2 + (game % 4) * 4
                    field_position = 5 + (game % 6) * 15
                    sep = 1 + (game % 4) * 2
                    context = {
                        "season": "2023",
                        "gameId": game_id,
                        "playId": f"{player}-{route}",
                        "nflId": str(player),
                        "displayName": f"Receiver {player}",
                        "officialPosition": "WR",
                        "week": str(week),
                        "down": str(1 + game % 4),
                        "yardsToGo": str(ytg),
                        "yardline_100": str(field_position),
                        "sep_snap": str(sep),
                        "snap_y_norm": str(4 + player * 9),
                        "nearest_defender_dx_snap": str((game % 5) - 2),
                        "nearest_defender_dy_snap": str((player % 5) - 2),
                        "receiver_speed_snap": str(game % 4),
                    }
                    residual = player * 0.25 + (game % 3) * 0.08 + (ytg > 8) * 0.2
                    contexts.append(context)
                    predictions.append({
                        "season": "2023",
                        "gameId": game_id,
                        "playId": f"{player}-{route}",
                        "nflId": str(player),
                        "week": str(week),
                        "residual_dynamic": str(residual),
                    })
    return contexts, predictions


def test_context_mix_audit_crossfits_by_game_and_returns_paired_reliability():
    contexts, predictions = _paired_rows()
    splits, fit_metrics, summaries = audit_context_mix(
        contexts,
        predictions,
        cohorts=((10, 3), (20, 5)),
        split_seeds=4,
        bootstrap_samples=20,
        seed=7,
    )

    assert len(splits) == 8
    assert len(fit_metrics) == 4
    assert all(int(row["routes"]) == len(predictions) for row in fit_metrics)
    assert {row["eligible_receivers"] for row in splits} == {"5"}
    assert len(summaries) == 2
    assert all(int(row["valid_splits"]) == 4 for row in summaries)
    for row in summaries:
        for metric in ("pearson", "spearman"):
            assert float(row[f"{metric}_raw_bootstrap_ci_lower"]) <= float(
                row[f"{metric}_raw_bootstrap_ci_upper"]
            )
            assert float(row[f"{metric}_change_bootstrap_ci_lower"]) <= float(
                row[f"{metric}_change_bootstrap_ci_upper"]
            )


def test_context_mix_audit_rejects_mismatched_and_duplicate_keys():
    contexts, predictions = _paired_rows()
    with pytest.raises(ValueError, match="different route keys"):
        audit_context_mix(contexts[:-1], predictions, split_seeds=1)
    with pytest.raises(ValueError, match="duplicate"):
        audit_context_mix(contexts + [contexts[0]], predictions, split_seeds=1)
