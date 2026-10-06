from __future__ import annotations

import random
import statistics
from collections import defaultdict

from .cross_season import SharedDynamicFeatureRidgeModel, SharedFeatureRidgeModel
from .models import target
from .validation import assign_game_folds, average_ranks, pearson_correlation, percentile


MODEL_TYPES = {
    "static": SharedFeatureRidgeModel,
    "dynamic": SharedDynamicFeatureRidgeModel,
}


def cross_season_receiver_predictions(
    legacy_rows: list[dict[str, str]],
    bdb2023_rows: list[dict[str, str]],
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Create game-held-out 2021 and frozen-2021-to-2023 route predictions.

    The returned rows contain residuals for both common-feature ridge models.
    Each 2021 route is scored out of game, while every 2023 route is scored by
    models fitted on all eligible 2021 routes.
    """
    if not legacy_rows or not bdb2023_rows:
        raise ValueError("Both seasons must contain eligible WR routes")
    required = {"week", "gameId", "nflId", "playId", "delta_sep"}
    if any(not required.issubset(row) for row in (*legacy_rows[:1], *bdb2023_rows[:1])):
        raise ValueError(f"Route rows must contain {sorted(required)}")
    folds = assign_game_folds(legacy_rows, n_folds=n_folds, seed=seed)
    predictions: list[dict[str, str]] = []
    for fold in range(n_folds):
        training = [
            row for row in legacy_rows
            if folds[(str(int(row["week"])), row["gameId"])] != fold
        ]
        heldout = [
            row for row in legacy_rows
            if folds[(str(int(row["week"])), row["gameId"])] == fold
        ]
        models = {
            name: model_type(l2=25.0, max_levels_per_feature=0)
            for name, model_type in MODEL_TYPES.items()
        }
        for model in models.values():
            model.fit(training)
        for row in heldout:
            output = {
                "season": "2021",
                "fold": str(fold + 1),
                "gameId": row["gameId"],
                "playId": row["playId"],
                "week": row["week"],
                "nflId": row["nflId"],
                "displayName": row.get("displayName", ""),
                "delta_sep": f"{target(row):.8f}",
            }
            for name, model in models.items():
                prediction = model.predict(row)
                output[f"pred_{name}"] = f"{prediction:.8f}"
                output[f"residual_{name}"] = f"{target(row) - prediction:.8f}"
            predictions.append(output)

    full_models = {
        name: model_type(l2=25.0, max_levels_per_feature=0)
        for name, model_type in MODEL_TYPES.items()
    }
    for model in full_models.values():
        model.fit(legacy_rows)
    for row in bdb2023_rows:
        output = {
            "season": "2023",
            "fold": "ALL_2021",
            "gameId": row["gameId"],
            "playId": row["playId"],
            "week": row["week"],
            "nflId": row["nflId"],
            "displayName": row.get("displayName", ""),
            "delta_sep": f"{target(row):.8f}",
        }
        for name, model in full_models.items():
            prediction = model.predict(row)
            output[f"pred_{name}"] = f"{prediction:.8f}"
            output[f"residual_{name}"] = f"{target(row) - prediction:.8f}"
        predictions.append(output)

    expected = len(legacy_rows) + len(bdb2023_rows)
    if len(predictions) != expected:
        raise RuntimeError(f"Expected {expected} route predictions, got {len(predictions)}")
    return predictions


def _cluster_mean_interval(
    game_values: dict[str, tuple[float, int]],
    iterations: int,
    rng: random.Random,
) -> tuple[float, float]:
    games = list(game_values.values())
    if len(games) < 2 or iterations < 1:
        return (0.0, 0.0)
    samples = []
    for _ in range(iterations):
        draw = [rng.choice(games) for _ in games]
        count = sum(n for _, n in draw)
        samples.append(sum(total for total, _ in draw) / count)
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def summarize_cross_season_receivers(
    predictions: list[dict[str, str]],
    min_routes: int = 20,
    min_games: int = 5,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int | float]]:
    """Aggregate receiver residuals, game-cluster intervals, and year stability.

    Receiver residuals are centered by the route-weighted league residual for
    that season, removing a common season-level calibration shift. Eligibility
    requires minimum route and game counts in both years. Correlation intervals
    resample receivers and their games; player-level error bars resample games.
    """
    if min_routes < 1 or min_games < 2 or bootstrap_samples < 1:
        raise ValueError(
            "min_routes and bootstrap_samples must be positive; min_games must be at least 2"
        )
    if not predictions:
        raise ValueError("At least one route prediction is required")

    residual_fields = {name: f"residual_{name}" for name in MODEL_TYPES}
    sums: dict[tuple[str, str], float] = defaultdict(float)
    route_counts: dict[tuple[str, str], int] = defaultdict(int)
    game_values: dict[tuple[str, str, str], dict[str, tuple[float, int]]] = defaultdict(dict)
    player_names: dict[str, set[str]] = defaultdict(set)
    ids_by_season: dict[str, set[str]] = defaultdict(set)
    for row in predictions:
        season = row["season"]
        player = row["nflId"]
        game = row["gameId"]
        if row.get("displayName"):
            player_names[player].add(row["displayName"])
        ids_by_season[season].add(player)
        for name, field in residual_fields.items():
            residual = float(row[field])
            key = (season, name)
            sums[key] += residual
            route_counts[key] += 1
            group_key = (season, name, player)
            previous_sum, previous_count = game_values[group_key].get(game, (0.0, 0))
            game_values[group_key][game] = (previous_sum + residual, previous_count + 1)

    league_means = {
        key: sums[key] / route_counts[key]
        for key in sums
    }
    routes_by_player: dict[tuple[str, str, str], int] = {}
    for key, games in game_values.items():
        routes_by_player[key] = sum(count for _, count in games.values())

    shared_ids = ids_by_season["2021"] & ids_by_season["2023"]
    rng = random.Random(seed)
    summaries: list[dict[str, str]] = []
    for player in sorted(shared_ids):
        names = sorted(player_names[player])
        row: dict[str, str] = {
            "nflId": player,
            "displayName": names[0] if names else "",
            "name_consistent": str(len(names) <= 1).lower(),
        }
        for season in ("2021", "2023"):
            for model_name in MODEL_TYPES:
                key = (season, model_name, player)
                games = game_values[key]
                count = routes_by_player.get(key, 0)
                total = sum(value for value, _ in games.values())
                raw_mean = total / count if count else 0.0
                centered_mean = raw_mean - league_means[(season, model_name)]
                lower, upper = _cluster_mean_interval(games, bootstrap_samples, rng)
                # League centering is constant over each bootstrap draw.
                lower -= league_means[(season, model_name)]
                upper -= league_means[(season, model_name)]
                prefix = f"{season}_{model_name}"
                row.update({
                    f"routes_{prefix}": str(count),
                    f"games_{prefix}": str(len(games)),
                    f"mean_residual_{prefix}": f"{raw_mean:.8f}",
                    f"centered_residual_{prefix}": f"{centered_mean:.8f}",
                    f"ci_lower_{prefix}": f"{lower:.8f}",
                    f"ci_upper_{prefix}": f"{upper:.8f}",
                })
        eligible = all(
            int(row[f"routes_{season}_{model}"]) >= min_routes
            and int(row[f"games_{season}_{model}"]) >= min_games
            for season in ("2021", "2023")
            for model in MODEL_TYPES
        )
        row["eligible"] = str(eligible).lower()
        summaries.append(row)

    eligible_rows = [row for row in summaries if row["eligible"] == "true"]
    metrics = _cross_season_correlations(
        eligible_rows,
        game_values,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    diagnostics: dict[str, int | float] = {
        "players_2021": len(ids_by_season["2021"]),
        "players_2023": len(ids_by_season["2023"]),
        "shared_player_ids": len(shared_ids),
        "eligible_shared_players": len(eligible_rows),
        "min_routes_per_season": min_routes,
        "min_games_per_season": min_games,
        "2021_static_route_bias": league_means[("2021", "static")],
        "2023_static_route_bias": league_means[("2023", "static")],
        "2021_dynamic_route_bias": league_means[("2021", "dynamic")],
        "2023_dynamic_route_bias": league_means[("2023", "dynamic")],
    }
    return summaries, metrics, diagnostics


def _cross_season_correlations(
    eligible_rows: list[dict[str, str]],
    game_values: dict[tuple[str, str, str], dict[str, tuple[float, int]]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, str]]:
    if len(eligible_rows) < 3:
        return []
    data = {
        model: (
            [float(row[f"centered_residual_2021_{model}"]) for row in eligible_rows],
            [float(row[f"centered_residual_2023_{model}"]) for row in eligible_rows],
        )
        for model in MODEL_TYPES
    }
    estimates = {
        model: {
            "pearson": pearson_correlation(*values),
            "spearman": pearson_correlation(average_ranks(values[0]), average_ranks(values[1])),
        }
        for model, values in data.items()
    }
    rng = random.Random(seed + 1)
    distributions = {
        model: {stat: [] for stat in ("pearson", "spearman")}
        for model in MODEL_TYPES
    }
    differences = {stat: [] for stat in ("pearson", "spearman")}
    n = len(eligible_rows)
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled = {model: ([], []) for model in MODEL_TYPES}
        for index in indices:
            player = eligible_rows[index]["nflId"]
            resampled_means = {model: {} for model in MODEL_TYPES}
            for season in ("2021", "2023"):
                player_games = game_values[(season, "static", player)]
                game_ids = list(player_games)
                draw = [rng.choice(game_ids) for _ in game_ids]
                draw_count = sum(
                    game_values[(season, "static", player)][game][1]
                    for game in draw
                )
                for model in MODEL_TYPES:
                    game_totals = game_values[(season, model, player)]
                    draw_total = sum(game_totals[game][0] for game in draw)
                    resampled_means[model][season] = draw_total / draw_count
            for model in MODEL_TYPES:
                sampled[model][0].append(resampled_means[model]["2021"])
                sampled[model][1].append(resampled_means[model]["2023"])
        sampled_stats = {}
        for model, (xs, ys) in sampled.items():
            pearson = pearson_correlation(xs, ys)
            spearman = pearson_correlation(average_ranks(xs), average_ranks(ys))
            sampled_stats[model] = {"pearson": pearson, "spearman": spearman}
            for stat, value in sampled_stats[model].items():
                distributions[model][stat].append(value)
        for stat in differences:
            differences[stat].append(
                sampled_stats["dynamic"][stat] - sampled_stats["static"][stat]
            )

    output = []
    for model in MODEL_TYPES:
        row = {
            "comparison": model,
            "receivers": str(n),
        }
        for stat in ("pearson", "spearman"):
            values = sorted(distributions[model][stat])
            row[stat] = f"{estimates[model][stat]:.6f}"
            row[f"{stat}_ci_lower"] = f"{percentile(values, 0.025):.6f}"
            row[f"{stat}_ci_upper"] = f"{percentile(values, 0.975):.6f}"
        output.append(row)
    difference_row = {"comparison": "dynamic_minus_static", "receivers": str(n)}
    for stat, values in differences.items():
        ordered = sorted(values)
        difference_row[stat] = f"{estimates['dynamic'][stat] - estimates['static'][stat]:.6f}"
        difference_row[f"{stat}_ci_lower"] = f"{percentile(ordered, 0.025):.6f}"
        difference_row[f"{stat}_ci_upper"] = f"{percentile(ordered, 0.975):.6f}"
    output.append(difference_row)
    return output
