from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .feature_schema import DYNAMIC_CONTEXT_FEATURES
from .models import RidgeContextModel, RidgeDynamicContextModel, regression_metrics, target


def assign_game_folds(
    rows: list[dict[str, str]], n_folds: int = 5, seed: int = 42
) -> dict[tuple[str, str], int]:
    """Assign whole games to folds, balancing game counts within each week."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    games_by_week: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        games_by_week[int(row["week"])].add(row["gameId"])
    assignments: dict[tuple[str, str], int] = {}
    for week, games in sorted(games_by_week.items()):
        if len(games) < n_folds:
            raise ValueError(
                f"Week {week} has {len(games)} games; need at least {n_folds} for grouped folds"
            )
        ordered = sorted(games)
        random.Random(seed + week).shuffle(ordered)
        for index, game_id in enumerate(ordered):
            assignments[(str(week), game_id)] = index % n_folds
    return assignments


def assign_game_halves(
    rows: list[dict[str, str]], seed: int = 10_042
) -> dict[tuple[str, str], str]:
    """Split games into two balanced halves within each week."""
    games_by_week: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        games_by_week[int(row["week"])].add(row["gameId"])
    assignments: dict[tuple[str, str], str] = {}
    for week, games in sorted(games_by_week.items()):
        ordered = sorted(games)
        random.Random(seed + week).shuffle(ordered)
        for index, game_id in enumerate(ordered):
            assignments[(str(week), game_id)] = "A" if index % 2 == 0 else "B"
    return assignments


def cross_validate_position(
    rows: list[dict[str, str]],
    position: str = "WR",
    n_folds: int = 5,
    seed: int = 42,
    min_routes_per_half: int = 20,
    bootstrap_samples: int = 2000,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Return out-of-game predictions, model metrics, player-half scores, and stability metrics."""
    rows = [row for row in rows if row["officialPosition"] == position]
    if not rows:
        raise ValueError(f"No route rows available for position={position!r}")
    missing = [feature for feature in DYNAMIC_CONTEXT_FEATURES if feature not in rows[0]]
    if missing:
        raise ValueError(
            "Dynamic features are required for cross-validation. Rebuild the route table "
            "with build-route-table --include-dynamic-features."
        )

    fold_by_game = assign_game_folds(rows, n_folds=n_folds, seed=seed)
    half_by_game = assign_game_halves(rows, seed=seed + 10_000)
    predictions: list[dict[str, str]] = []
    fold_metrics: list[dict[str, str]] = []

    for fold in range(n_folds):
        train_rows = [
            row for row in rows
            if fold_by_game[(str(int(row["week"])), row["gameId"])] != fold
        ]
        heldout_rows = [
            row for row in rows
            if fold_by_game[(str(int(row["week"])), row["gameId"])] == fold
        ]
        static_model = RidgeContextModel(l2=25.0, max_levels_per_feature=30)
        dynamic_model = RidgeDynamicContextModel(l2=25.0, max_levels_per_feature=30)
        static_model.fit(train_rows)
        dynamic_model.fit(train_rows)

        actual = [target(row) for row in heldout_rows]
        static_predictions = [static_model.predict(row) for row in heldout_rows]
        dynamic_predictions = [dynamic_model.predict(row) for row in heldout_rows]
        for model_name, values in (
            ("ridge_context", static_predictions),
            ("ridge_dynamic_context", dynamic_predictions),
        ):
            fold_metrics.append(
                {
                    "fold": str(fold + 1),
                    "model": model_name,
                    **regression_metrics(actual, values),
                }
            )

        for row, static_pred, dynamic_pred in zip(
            heldout_rows, static_predictions, dynamic_predictions
        ):
            game_key = (str(int(row["week"])), row["gameId"])
            predictions.append(
                {
                    "gameId": row["gameId"],
                    "playId": row["playId"],
                    "nflId": row["nflId"],
                    "displayName": row["displayName"],
                    "officialPosition": row["officialPosition"],
                    "week": row["week"],
                    "fold": str(fold + 1),
                    "reliability_half": half_by_game[game_key],
                    "delta_sep": f"{target(row):.8f}",
                    "pred_delta_sep_ridge_context": f"{static_pred:.8f}",
                    "pred_delta_sep_ridge_dynamic_context": f"{dynamic_pred:.8f}",
                    "soe_route_ridge_context": f"{target(row) - static_pred:.8f}",
                    "soe_route_ridge_dynamic_context": f"{target(row) - dynamic_pred:.8f}",
                }
            )

    if len(predictions) != len(rows):
        raise RuntimeError(
            f"Expected one out-of-fold prediction per route ({len(rows)}), got {len(predictions)}"
        )
    key_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    game_folds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in predictions:
        key_counts[(row["gameId"], row["playId"], row["nflId"])] += 1
        game_folds[(row["week"], row["gameId"])].add(row["fold"])
    if any(count != 1 for count in key_counts.values()):
        raise RuntimeError("Out-of-fold route keys are not unique")
    if any(len(folds) != 1 for folds in game_folds.values()):
        raise RuntimeError("A game was assigned to multiple folds")

    for model_name, prediction_column in (
        ("ridge_context", "pred_delta_sep_ridge_context"),
        ("ridge_dynamic_context", "pred_delta_sep_ridge_dynamic_context"),
    ):
        fold_metrics.append(
            {
                "fold": "OOF_ALL",
                "model": model_name,
                **regression_metrics(
                    [float(row["delta_sep"]) for row in predictions],
                    [float(row[prediction_column]) for row in predictions],
                ),
            }
        )

    player_rows, reliability_metrics = receiver_half_reliability(
        predictions,
        min_routes_per_half=min_routes_per_half,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return predictions, fold_metrics, player_rows, reliability_metrics


def receiver_half_reliability(
    predictions: list[dict[str, str]],
    min_routes_per_half: int = 20,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_player: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {
            "A": {"ridge_context": [], "ridge_dynamic_context": []},
            "B": {"ridge_context": [], "ridge_dynamic_context": []},
        }
    )
    for row in predictions:
        player = (row["nflId"], row["displayName"])
        half = row["reliability_half"]
        by_player[player][half]["ridge_context"].append(
            float(row["soe_route_ridge_context"])
        )
        by_player[player][half]["ridge_dynamic_context"].append(
            float(row["soe_route_ridge_dynamic_context"])
        )

    eligible = {
        player: halves
        for player, halves in by_player.items()
        if all(
            len(halves[half][model]) >= min_routes_per_half
            for half in ("A", "B")
            for model in ("ridge_context", "ridge_dynamic_context")
        )
    }
    player_rows = []
    for (nfl_id, name), halves in sorted(eligible.items()):
        row = {"nflId": nfl_id, "displayName": name}
        for model in ("ridge_context", "ridge_dynamic_context"):
            for half in ("A", "B"):
                values = halves[half][model]
                row[f"routes_{model}_{half}"] = str(len(values))
                row[f"mean_soe_{model}_{half}"] = f"{statistics.fmean(values):.8f}"
        player_rows.append(row)

    metrics = reliability_metrics(
        player_rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        min_routes_per_half=min_routes_per_half,
    )
    return player_rows, metrics


def reliability_metrics(
    player_rows: list[dict[str, str]],
    bootstrap_samples: int = 2000,
    seed: int = 42,
    min_routes_per_half: int = 20,
) -> list[dict[str, str]]:
    model_pairs = {
        model: (
            [float(row[f"mean_soe_{model}_A"]) for row in player_rows],
            [float(row[f"mean_soe_{model}_B"]) for row in player_rows],
        )
        for model in ("ridge_context", "ridge_dynamic_context")
    }
    observed = {model: correlation_stats(*pairs) for model, pairs in model_pairs.items()}
    rng = random.Random(seed)
    bootstraps = {model: {"pearson": [], "spearman": []} for model in model_pairs}
    difference_bootstrap = []
    difference_spearman_bootstrap = []
    n = len(player_rows)
    if n >= 3 and bootstrap_samples > 0:
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(n) for _ in range(n)]
            sampled = {
                model: (
                    [pairs[0][i] for i in indices],
                    [pairs[1][i] for i in indices],
                )
                for model, pairs in model_pairs.items()
            }
            for model, (xs, ys) in sampled.items():
                stats = correlation_stats(xs, ys)
                for statistic, value in stats.items():
                    bootstraps[model][statistic].append(value)
            difference_bootstrap.append(
                bootstraps["ridge_dynamic_context"]["pearson"][-1]
                - bootstraps["ridge_context"]["pearson"][-1]
            )
            difference_spearman_bootstrap.append(
                bootstraps["ridge_dynamic_context"]["spearman"][-1]
                - bootstraps["ridge_context"]["spearman"][-1]
            )

    results = []
    for model in model_pairs:
        pearson_values = sorted(bootstraps[model]["pearson"])
        spearman_values = sorted(bootstraps[model]["spearman"])
        results.append(
            {
                "comparison": model,
                "eligible_receivers": str(n),
                "pearson": f"{observed[model]['pearson']:.6f}",
                "spearman": f"{observed[model]['spearman']:.6f}",
                "pearson_ci_lower": f"{percentile(pearson_values, 0.025):.6f}" if pearson_values else "",
                "pearson_ci_upper": f"{percentile(pearson_values, 0.975):.6f}" if pearson_values else "",
                "spearman_ci_lower": f"{percentile(spearman_values, 0.025):.6f}" if spearman_values else "",
                "spearman_ci_upper": f"{percentile(spearman_values, 0.975):.6f}" if spearman_values else "",
                "minimum_routes_each_half": str(min_routes_per_half),
            }
    )
    differences = sorted(difference_bootstrap)
    spearman_differences = sorted(difference_spearman_bootstrap)
    results.append(
        {
            "comparison": "dynamic_minus_static_pearson",
            "eligible_receivers": str(n),
            "pearson": f"{observed['ridge_dynamic_context']['pearson'] - observed['ridge_context']['pearson']:.6f}",
            "spearman": f"{observed['ridge_dynamic_context']['spearman'] - observed['ridge_context']['spearman']:.6f}",
            "pearson_ci_lower": f"{percentile(differences, 0.025):.6f}" if differences else "",
            "pearson_ci_upper": f"{percentile(differences, 0.975):.6f}" if differences else "",
            "spearman_ci_lower": f"{percentile(spearman_differences, 0.025):.6f}" if spearman_differences else "",
            "spearman_ci_upper": f"{percentile(spearman_differences, 0.975):.6f}" if spearman_differences else "",
            "minimum_routes_each_half": str(min_routes_per_half),
        }
    )
    return results


def correlation_stats(xs: list[float], ys: list[float]) -> dict[str, float]:
    return {
        "pearson": pearson_correlation(xs, ys),
        "spearman": pearson_correlation(average_ranks(xs), average_ranks(ys)),
    }


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    denominator = (x_ss * y_ss) ** 0.5
    return numerator / denominator if denominator else 0.0


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    location = (len(values) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(values) - 1)
    fraction = location - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def write_oof_outputs(
    output_dir: Path,
    position: str,
    predictions: list[dict[str, str]],
    fold_metrics: list[dict[str, str]],
    player_rows: list[dict[str, str]],
    reliability: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = position.lower()
    write_dict_rows(
        output_dir / f"route_level_oof_predictions_{suffix}.csv",
        list(predictions[0]) if predictions else [],
        predictions,
    )
    metric_fields = ["fold", "model", "mae", "rmse", "r2", "bias", "mean_actual", "mean_predicted", "n"]
    write_dict_rows(output_dir / f"oof_metrics_{suffix}.csv", metric_fields, fold_metrics)
    write_dict_rows(
        output_dir / f"receiver_oof_half_scores_{suffix}.csv",
        list(player_rows[0]) if player_rows else ["nflId", "displayName"],
        player_rows,
    )
    reliability_fields = [
        "comparison", "eligible_receivers", "pearson", "spearman",
        "pearson_ci_lower", "pearson_ci_upper", "spearman_ci_lower",
        "spearman_ci_upper", "minimum_routes_each_half",
    ]
    write_dict_rows(
        output_dir / f"receiver_oof_reliability_{suffix}.csv",
        reliability_fields,
        reliability,
    )


def write_dict_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
