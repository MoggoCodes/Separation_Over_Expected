from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from .validation import (
    assign_game_halves,
    average_ranks,
    pearson_correlation,
    percentile,
)


MIN_ROUTES_PER_HALF = 20
MIN_GAMES_PER_HALF = 5
MODEL_NAMES = ("no_player_effect", "unpooled_mean", "partial_pool")


def _mean_variance_from_game_clusters(
    game_routes: dict[str, list[float]],
) -> float:
    """Cluster-robust sampling variance of a route-weighted player mean."""
    game_totals = [(sum(values), len(values)) for values in game_routes.values()]
    if len(game_totals) < 2:
        return 0.0
    route_count = sum(count for _, count in game_totals)
    player_mean = sum(total for total, _ in game_totals) / route_count
    squared_cluster_sums = sum(
        (total - count * player_mean) ** 2
        for total, count in game_totals
    )
    return len(game_totals) / (len(game_totals) - 1) * squared_cluster_sums / (route_count**2)


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [prediction - observed for observed, prediction in zip(actual, predicted)]
    return {
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "mae": statistics.fmean(abs(error) for error in errors),
        "bias": statistics.fmean(errors),
        "pearson": pearson_correlation(actual, predicted),
        "spearman": pearson_correlation(average_ranks(actual), average_ranks(predicted)),
    }


def evaluate_partial_pooled_receiver_effects(
    predictions: list[dict[str, str]],
    split_seeds: int = 100,
    bootstrap_samples: int = 300,
    seed: int = 42,
    min_routes_per_half: int = MIN_ROUTES_PER_HALF,
    min_games_per_half: int = MIN_GAMES_PER_HALF,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Test empirical-Bayes receiver effects on held-out 2023 games.

    The frozen route model and its route residuals do not change. Whole games
    are repeatedly split, balanced within week. Receiver effects estimated in
    one half predict centered receiver residual means in the other half. The
    empirical-Bayes prior variance is estimated by method of moments on the
    training half after subtracting game-cluster sampling variance.
    """
    if split_seeds < 1 or bootstrap_samples < 1:
        raise ValueError("split_seeds and bootstrap_samples must be positive")
    if min_routes_per_half < 1 or min_games_per_half < 2:
        raise ValueError("min_routes_per_half must be positive and min_games_per_half at least 2")
    required = {"season", "week", "gameId", "playId", "nflId", "residual_dynamic"}
    rows = [row for row in predictions if row.get("season") == "2023"]
    if not rows:
        raise ValueError("No 2023 route predictions were provided")
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"2023 rows must contain {sorted(required)}")
    keys = [(row["gameId"], row["playId"], row["nflId"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("2023 route predictions contain duplicate route keys")
    try:
        for row in rows:
            int(row["week"])
            if not math.isfinite(float(row["residual_dynamic"])):
                raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("2023 predictions contain invalid week or residual values") from exc

    split_rows: list[dict[str, str]] = []
    effect_rows: list[dict[str, str]] = []
    bootstrap_deltas = {
        "partial_minus_no_player_effect": [],
        "partial_minus_unpooled_mean": [],
    }
    for split_index in range(split_seeds):
        split_seed = seed + 10_000 + split_index * 37
        half_by_game = assign_game_halves(rows, seed=split_seed)
        rows_by_half: dict[str, list[dict[str, str]]] = {"A": [], "B": []}
        for row in rows:
            half = half_by_game[(str(int(row["week"])), row["gameId"])]
            rows_by_half[half].append(row)

        for train_half, test_half in (("A", "B"), ("B", "A")):
            train_by_player_game: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            test_by_player_game: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for row in rows_by_half[train_half]:
                train_by_player_game[row["nflId"]][row["gameId"]].append(float(row["residual_dynamic"]))
            for row in rows_by_half[test_half]:
                test_by_player_game[row["nflId"]][row["gameId"]].append(float(row["residual_dynamic"]))

            eligible = sorted(
                player for player in set(train_by_player_game) & set(test_by_player_game)
                if sum(len(routes) for routes in train_by_player_game[player].values()) >= min_routes_per_half
                and len(train_by_player_game[player]) >= min_games_per_half
                and sum(len(routes) for routes in test_by_player_game[player].values()) >= min_routes_per_half
                and len(test_by_player_game[player]) >= min_games_per_half
            )
            if len(eligible) < 3:
                split_rows.append({
                    "seed": str(split_seed),
                    "direction": f"{train_half}_to_{test_half}",
                    "eligible_receivers": str(len(eligible)),
                    "tau2": "",
                    "mean_shrinkage": "",
                    **{f"{model}_{metric}": "" for model in MODEL_NAMES for metric in ("rmse", "mae", "bias", "pearson", "spearman")},
                })
                continue

            train_routes = [value for player in eligible for values in train_by_player_game[player].values() for value in values]
            test_routes = [value for player in eligible for values in test_by_player_game[player].values() for value in values]
            train_league_mean = statistics.fmean(train_routes)
            test_league_mean = statistics.fmean(test_routes)
            train_effects: dict[str, float] = {}
            train_mean_variances: dict[str, float] = {}
            for player in eligible:
                routes = [value for values in train_by_player_game[player].values() for value in values]
                train_effects[player] = statistics.fmean(routes) - train_league_mean
                train_mean_variances[player] = _mean_variance_from_game_clusters(train_by_player_game[player])
            observed_between_player_variance = statistics.variance(train_effects.values())
            mean_sampling_variance = statistics.fmean(train_mean_variances.values())
            tau2 = max(0.0, observed_between_player_variance - mean_sampling_variance)

            estimates: dict[str, dict[str, float]] = {}
            test_effects: dict[str, float] = {}
            for player in eligible:
                test_routes_for_player = [value for values in test_by_player_game[player].values() for value in values]
                test_effects[player] = statistics.fmean(test_routes_for_player) - test_league_mean
                sampling_variance = train_mean_variances[player]
                shrinkage = tau2 / (tau2 + sampling_variance) if tau2 + sampling_variance > 0 else 0.0
                partial = shrinkage * train_effects[player]
                posterior_variance = (
                    tau2 * sampling_variance / (tau2 + sampling_variance)
                    if tau2 + sampling_variance > 0 else 0.0
                )
                estimates[player] = {
                    "no_player_effect": 0.0,
                    "unpooled_mean": train_effects[player],
                    "partial_pool": partial,
                    "shrinkage": shrinkage,
                    "posterior_sd": math.sqrt(posterior_variance),
                }

            actual = [test_effects[player] for player in eligible]
            model_predictions = {
                model: [estimates[player][model] for player in eligible]
                for model in MODEL_NAMES
            }
            model_metrics = {
                model: _metrics(actual, predicted)
                for model, predicted in model_predictions.items()
            }
            output_row: dict[str, str] = {
                "seed": str(split_seed),
                "direction": f"{train_half}_to_{test_half}",
                "eligible_receivers": str(len(eligible)),
                "tau2": f"{tau2:.8f}",
                "mean_shrinkage": f"{statistics.fmean(estimates[player]['shrinkage'] for player in eligible):.8f}",
            }
            for model, metrics in model_metrics.items():
                output_row.update({f"{model}_{metric}": f"{value:.8f}" for metric, value in metrics.items()})
            split_rows.append(output_row)

            rng = random.Random(split_seed + (0 if train_half == "A" else 1) + 1)
            for _ in range(bootstrap_samples):
                indices = [rng.randrange(len(eligible)) for _ in eligible]
                y = [actual[index] for index in indices]
                partial_pred = [model_predictions["partial_pool"][index] for index in indices]
                null_pred = [model_predictions["no_player_effect"][index] for index in indices]
                raw_pred = [model_predictions["unpooled_mean"][index] for index in indices]
                partial_rmse = math.sqrt(statistics.fmean(
                    (prediction - observed) ** 2
                    for observed, prediction in zip(y, partial_pred)
                ))
                null_rmse = math.sqrt(statistics.fmean(
                    (prediction - observed) ** 2
                    for observed, prediction in zip(y, null_pred)
                ))
                raw_rmse = math.sqrt(statistics.fmean(
                    (prediction - observed) ** 2
                    for observed, prediction in zip(y, raw_pred)
                ))
                bootstrap_deltas["partial_minus_no_player_effect"].append(
                    partial_rmse - null_rmse
                )
                bootstrap_deltas["partial_minus_unpooled_mean"].append(
                    partial_rmse - raw_rmse
                )

            if split_index == 0 and train_half == "A":
                for player in eligible:
                    estimate = estimates[player]
                    posterior_mean = estimate["partial_pool"]
                    posterior_sd = estimate["posterior_sd"]
                    effect_rows.append({
                        "nflId": player,
                        "displayName": next((row.get("displayName", "") for row in rows if row["nflId"] == player), ""),
                        "train_routes": str(sum(len(values) for values in train_by_player_game[player].values())),
                        "train_games": str(len(train_by_player_game[player])),
                        "training_unpooled_effect": f"{train_effects[player]:.8f}",
                        "training_partial_pool_effect": f"{posterior_mean:.8f}",
                        "shrinkage_factor": f"{estimate['shrinkage']:.8f}",
                        "posterior_sd_plugin": f"{posterior_sd:.8f}",
                        "posterior_ci_lower_plugin": f"{posterior_mean - 1.96 * posterior_sd:.8f}",
                        "posterior_ci_upper_plugin": f"{posterior_mean + 1.96 * posterior_sd:.8f}",
                        "heldout_routes": str(sum(len(values) for values in test_by_player_game[player].values())),
                        "heldout_games": str(len(test_by_player_game[player])),
                        "heldout_centered_effect": f"{test_effects[player]:.8f}",
                    })

    summary_rows: list[dict[str, str]] = []
    valid_rows = [row for row in split_rows if row["tau2"] != ""]
    for model in MODEL_NAMES:
        for metric in ("rmse", "mae", "bias", "pearson", "spearman"):
            values_for_metric = sorted(float(row[f"{model}_{metric}"]) for row in valid_rows)
            summary_rows.append({
                "comparison": model,
                "metric": metric,
                "median": f"{statistics.median(values_for_metric):.8f}" if values_for_metric else "",
                "p10": f"{percentile(values_for_metric, .10):.8f}" if values_for_metric else "",
                "p90": f"{percentile(values_for_metric, .90):.8f}" if values_for_metric else "",
                "directions": str(len(values_for_metric)),
            })
    for comparison, delta_values in bootstrap_deltas.items():
        ordered = sorted(delta_values)
        raw_delta = [
            float(row["partial_pool_rmse"]) - float(row["no_player_effect_rmse" if comparison == "partial_minus_no_player_effect" else "unpooled_mean_rmse"])
            for row in valid_rows
        ]
        summary_rows.append({
            "comparison": comparison,
            "metric": "rmse_delta_partial_minus_baseline",
            "median": f"{statistics.median(raw_delta):.8f}" if raw_delta else "",
            "p10": f"{percentile(sorted(raw_delta), .10):.8f}" if raw_delta else "",
            "p90": f"{percentile(sorted(raw_delta), .90):.8f}" if raw_delta else "",
            "bootstrap_ci_lower": f"{percentile(ordered, .025):.8f}" if ordered else "",
            "bootstrap_ci_upper": f"{percentile(ordered, .975):.8f}" if ordered else "",
            "directions": str(len(valid_rows)),
        })
    return split_rows, summary_rows, effect_rows
