from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from .models import (
    RidgeContextModel,
    RidgeDynamicContextModel,
    RidgePocketContextModel,
    RidgePressureContextModel,
    regression_metrics,
    target,
)
from .validation import (
    assign_game_folds,
    assign_game_halves,
    receiver_half_reliability,
    reliability_metrics,
)


FEATURE_SPECS = (
    ("static", "Static context", RidgeContextModel),
    ("dynamic", "Dynamic defender motion", RidgeDynamicContextModel),
    ("dynamic_pressure", "Dynamic + pressure", RidgePressureContextModel),
    ("dynamic_pocket", "Dynamic + full pocket", RidgePocketContextModel),
)
DEFAULT_L2_VALUES = (1.0, 5.0, 25.0, 100.0, 500.0)
DEFAULT_RELIABILITY_BASELINE = "dynamic_l2_25"


def _penalty_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _candidate_id(feature_set: str, l2: float) -> str:
    return f"{feature_set}_l2_{_penalty_label(l2)}"


def _calibration_summary(
    actual: list[float], predicted: list[float], bins: int = 10
) -> tuple[dict[str, float], list[dict[str, str]]]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Calibration requires paired, non-empty predictions")
    mean_actual = statistics.fmean(actual)
    mean_predicted = statistics.fmean(predicted)
    pred_variance = sum((value - mean_predicted) ** 2 for value in predicted)
    slope = (
        sum((prediction - mean_predicted) * (observed - mean_actual)
            for prediction, observed in zip(predicted, actual)) / pred_variance
        if pred_variance
        else 0.0
    )
    intercept = mean_actual - slope * mean_predicted
    ordered = sorted(zip(predicted, actual), key=lambda pair: pair[0])
    rows = []
    absolute_bias_sum = 0.0
    maximum_absolute_bias = 0.0
    for index in range(bins):
        start = index * len(ordered) // bins
        end = (index + 1) * len(ordered) // bins
        group = ordered[start:end]
        if not group:
            continue
        group_predicted = statistics.fmean(prediction for prediction, _ in group)
        group_actual = statistics.fmean(observed for _, observed in group)
        residual = group_actual - group_predicted
        absolute_bias_sum += len(group) * abs(residual)
        maximum_absolute_bias = max(maximum_absolute_bias, abs(residual))
        rows.append({
            "decile": str(index + 1),
            "routes": str(len(group)),
            "mean_predicted": f"{group_predicted:.8f}",
            "mean_actual": f"{group_actual:.8f}",
            "mean_residual": f"{residual:.8f}",
        })
    return (
        {
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "mean_absolute_decile_bias": absolute_bias_sum / len(actual),
            "maximum_absolute_decile_bias": maximum_absolute_bias,
        },
        rows,
    )


def _paired_game_rmse_bootstrap(
    predictions: list[dict[str, str]],
    candidate: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in predictions:
        observed = float(row["delta_sep"])
        candidate_residual = observed - float(row[f"pred_{candidate}"])
        baseline_residual = observed - float(row[f"pred_{baseline}"])
        grouped[row["gameId"]].append((candidate_residual ** 2, baseline_residual ** 2, 1.0))
    games = list(grouped)
    if len(games) < 2:
        raise ValueError("At least two games are required for paired game bootstrap")

    def delta(sample: list[str]) -> float:
        n = sum(len(grouped[game]) for game in sample)
        candidate_sse = sum(
            sum(value[0] for value in grouped[game]) for game in sample
        )
        baseline_sse = sum(
            sum(value[1] for value in grouped[game]) for game in sample
        )
        return math.sqrt(candidate_sse / n) - math.sqrt(baseline_sse / n)

    point = delta(games)
    rng = random.Random(seed)
    samples = sorted(
        delta([rng.choice(games) for _ in games])
        for _ in range(iterations)
    )
    return {
        "rmse_delta_vs_dynamic_l2_25": point,
        "rmse_delta_lower_95": samples[int(0.025 * (iterations - 1))],
        "rmse_delta_upper_95": samples[int(0.975 * (iterations - 1))],
    }


def compare_ridge_specifications(
    rows: list[dict[str, str]],
    l2_values: tuple[float, ...] = DEFAULT_L2_VALUES,
    n_folds: int = 5,
    min_routes_per_half: int = 20,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Compare ridge feature blocks and penalties on identical game-held-out folds.

    Returns route OOF predictions, aggregate candidate metrics, fold metrics,
    calibration deciles, receiver split-half scores, and reliability metrics.
    Each model's feature scaling and category levels are fit on that fold's training rows;
    sufficient statistics are then reused to solve the full penalty grid.
    """
    wr_rows = [row for row in rows if row.get("officialPosition") == "WR"]
    if not wr_rows:
        raise ValueError("No WR route rows are available")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if min_routes_per_half < 1 or bootstrap_samples < 1:
        raise ValueError("min_routes_per_half and bootstrap_samples must be positive")
    penalties = tuple(dict.fromkeys(float(value) for value in l2_values))
    if not penalties or any(value < 0 for value in penalties):
        raise ValueError("l2_values must contain at least one non-negative penalty")
    candidate_names = [
        _candidate_id(spec, penalty)
        for spec, _, _ in FEATURE_SPECS
        for penalty in penalties
    ]
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("The requested feature/penalty grid contains duplicate candidates")
    if DEFAULT_RELIABILITY_BASELINE not in candidate_names:
        raise ValueError(
            f"l2_values must include 25 for baseline {DEFAULT_RELIABILITY_BASELINE}"
        )

    required_by_spec = {
        name: set(model_type.numeric_features) | set(model_type.categorical_features)
        for name, _, model_type in FEATURE_SPECS
    }
    available = set(wr_rows[0])
    missing = {
        name: sorted(features - available)
        for name, features in required_by_spec.items()
        if features - available
    }
    if missing:
        raise ValueError(f"Route table is missing model features: {missing}")

    fold_by_game = assign_game_folds(wr_rows, n_folds=n_folds, seed=seed)
    half_by_game = assign_game_halves(wr_rows, seed=seed + 10_000)
    predictions: list[dict[str, str]] = []
    fold_metrics: list[dict[str, str]] = []

    for fold in range(n_folds):
        training = [
            row for row in wr_rows
            if fold_by_game[(str(int(row["week"])), row["gameId"])] != fold
        ]
        heldout = [
            row for row in wr_rows
            if fold_by_game[(str(int(row["week"])), row["gameId"])] == fold
        ]
        fold_predictions: dict[str, list[float]] = {}
        for feature_set, _, model_type in FEATURE_SPECS:
            model = model_type(l2=min(penalties), max_levels_per_feature=30)
            model.fit(training)
            for penalty in penalties:
                model.refit_l2(penalty)
                name = _candidate_id(feature_set, penalty)
                predictions_for_fold = [model.predict(row) for row in heldout]
                fold_predictions[name] = predictions_for_fold
                actual = [target(row) for row in heldout]
                model_metrics = regression_metrics(actual, predictions_for_fold)
                calibration, _ = _calibration_summary(actual, predictions_for_fold)
                fold_metrics.append({
                    "fold": str(fold + 1),
                    "candidate": name,
                    "feature_set": feature_set,
                    "l2": f"{penalty:g}",
                    "routes": str(len(heldout)),
                    **model_metrics,
                    **{key: f"{value:.8f}" for key, value in calibration.items()},
                })

        for index, row in enumerate(heldout):
            game_key = (str(int(row["week"])), row["gameId"])
            prediction_row = {
                "gameId": row["gameId"],
                "playId": row["playId"],
                "nflId": row["nflId"],
                "displayName": row["displayName"],
                "officialPosition": row["officialPosition"],
                "week": row["week"],
                "fold": str(fold + 1),
                "reliability_half": half_by_game[game_key],
                "delta_sep": f"{target(row):.8f}",
            }
            for candidate, values in fold_predictions.items():
                prediction = values[index]
                prediction_row[f"pred_{candidate}"] = f"{prediction:.8f}"
                prediction_row[f"soe_route_{candidate}"] = f"{target(row) - prediction:.8f}"
            predictions.append(prediction_row)

    if len(predictions) != len(wr_rows):
        raise RuntimeError(
            f"Expected {len(wr_rows)} out-of-fold predictions, got {len(predictions)}"
        )
    unique_routes = {
        (row["gameId"], row["playId"], row["nflId"])
        for row in predictions
    }
    if len(unique_routes) != len(predictions):
        raise RuntimeError("Out-of-fold route predictions contain duplicate route keys")

    fold_values: dict[str, list[float]] = defaultdict(list)
    for row in fold_metrics:
        fold_values[row["candidate"]].append(float(row["rmse"]))

    aggregate_metrics = []
    calibration_rows = []
    for feature_set, _, _ in FEATURE_SPECS:
        for penalty in penalties:
            candidate = _candidate_id(feature_set, penalty)
            actual = [float(row["delta_sep"]) for row in predictions]
            predicted = [float(row[f"pred_{candidate}"]) for row in predictions]
            metrics = regression_metrics(actual, predicted)
            calibration, deciles = _calibration_summary(actual, predicted)
            fold_rmse = fold_values[candidate]
            aggregate_metrics.append({
                "candidate": candidate,
                "feature_set": feature_set,
                "l2": f"{penalty:g}",
                "routes": str(len(predictions)),
                **metrics,
                **{key: f"{value:.8f}" for key, value in calibration.items()},
                "fold_rmse_mean": f"{statistics.fmean(fold_rmse):.8f}",
                "fold_rmse_sd": f"{statistics.stdev(fold_rmse):.8f}" if len(fold_rmse) > 1 else "0.00000000",
            })
            for decile in deciles:
                calibration_rows.append({
                    "candidate": candidate,
                    "feature_set": feature_set,
                    "l2": f"{penalty:g}",
                    **decile,
                })

    baseline = DEFAULT_RELIABILITY_BASELINE
    rmse_bootstrap = {
        candidate: _paired_game_rmse_bootstrap(
            predictions,
            candidate,
            baseline,
            iterations=bootstrap_samples,
            seed=seed,
        )
        for candidate in candidate_names
    }
    for row in aggregate_metrics:
        row.update({key: f"{value:.8f}" for key, value in rmse_bootstrap[row["candidate"]].items()})

    player_halves, _ = receiver_half_reliability(
        predictions,
        min_routes_per_half=min_routes_per_half,
        bootstrap_samples=0,
        seed=seed,
        model_names=tuple(candidate_names),
    )
    reliability_results = []
    if len(player_halves) >= 3:
        for candidate in candidate_names:
            selected_names = (candidate,) if candidate == baseline else (baseline, candidate)
            result = reliability_metrics(
                player_halves,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                min_routes_per_half=min_routes_per_half,
                model_names=selected_names,
            )
            model_row = next(row for row in result if row["comparison"] == candidate)
            comparison_row = result[-1] if candidate != baseline else None
            output = {
                "candidate": candidate,
                "feature_set": candidate.rsplit("_l2_", 1)[0],
                "l2": candidate.rsplit("_l2_", 1)[1].replace("p", "."),
                "eligible_receivers": model_row["eligible_receivers"],
                "pearson": model_row["pearson"],
                "pearson_ci_lower": model_row["pearson_ci_lower"],
                "pearson_ci_upper": model_row["pearson_ci_upper"],
                "spearman": model_row["spearman"],
                "spearman_ci_lower": model_row["spearman_ci_lower"],
                "spearman_ci_upper": model_row["spearman_ci_upper"],
                "pearson_delta_vs_dynamic_l2_25": comparison_row["pearson"] if comparison_row else "0.000000",
                "pearson_delta_lower_95": comparison_row["pearson_ci_lower"] if comparison_row else "0.000000",
                "pearson_delta_upper_95": comparison_row["pearson_ci_upper"] if comparison_row else "0.000000",
                "spearman_delta_vs_dynamic_l2_25": comparison_row["spearman"] if comparison_row else "0.000000",
                "spearman_delta_lower_95": comparison_row["spearman_ci_lower"] if comparison_row else "0.000000",
                "spearman_delta_upper_95": comparison_row["spearman_ci_upper"] if comparison_row else "0.000000",
            }
            reliability_results.append(output)

    return (
        predictions,
        aggregate_metrics,
        fold_metrics,
        calibration_rows,
        player_halves,
        reliability_results,
    )
