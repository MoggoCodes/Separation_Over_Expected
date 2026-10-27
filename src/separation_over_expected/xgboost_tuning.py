from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

import numpy as np
from xgboost import XGBRegressor

from .algorithm_comparison import (
    _calibration,
    _features,
    _matrix,
    _paired_game_bootstrap,
    _preprocessor,
)
from .models import RidgeDynamicContextModel, regression_metrics, target
from .validation import (
    assign_game_folds,
    assign_game_halves,
    receiver_half_reliability,
    reliability_metrics,
)


MODEL_NAMES = ("ridge_dynamic_context", "xgboost_fixed", "xgboost_tuned")
MAX_BOOSTING_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 60


def _sample_configs(seed: int) -> list[dict[str, float | int]]:
    """Return a small deterministic search, including the prior fixed settings."""
    rng = random.Random(seed)
    configs: list[dict[str, float | int]] = [
        {
            "max_depth": 5,
            "min_child_weight": 10,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
            "reg_alpha": 0.0,
            "gamma": 0.0,
        }
    ]
    while len(configs) < 12:
        config = {
            "max_depth": rng.choice((3, 4, 5, 6)),
            "min_child_weight": rng.choice((3, 5, 10, 20)),
            "learning_rate": math.exp(rng.uniform(math.log(0.02), math.log(0.10))),
            "subsample": rng.uniform(0.7, 1.0),
            "colsample_bytree": rng.uniform(0.6, 1.0),
            "reg_lambda": math.exp(rng.uniform(math.log(0.5), math.log(50.0))),
            "reg_alpha": rng.choice((0.0, 0.1, 1.0, 5.0)),
            "gamma": rng.choice((0.0, 0.1, 0.3, 0.5)),
        }
        if config not in configs:
            configs.append(config)
    return configs


def _rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(
        statistics.fmean((y - yhat) ** 2 for y, yhat in zip(actual, predicted))
    )


def _paired_rmse_bootstrap(
    predictions: list[dict[str, str]],
    candidate: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    by_game: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in predictions:
        actual = float(row["delta_sep"])
        by_game[row["gameId"]].append((
            (actual - float(row[f"pred_delta_sep_{candidate}"])) ** 2,
            (actual - float(row[f"pred_delta_sep_{baseline}"])) ** 2,
        ))
    games = list(by_game)

    def difference(sampled_games: list[str]) -> float:
        n = sum(len(by_game[game]) for game in sampled_games)
        candidate_sse = sum(sum(pair[0] for pair in by_game[game]) for game in sampled_games)
        baseline_sse = sum(sum(pair[1] for pair in by_game[game]) for game in sampled_games)
        return math.sqrt(candidate_sse / n) - math.sqrt(baseline_sse / n)

    rng = random.Random(seed)
    draws = sorted(
        difference([rng.choice(games) for _ in games])
        for _ in range(iterations)
    )
    return {
        "rmse_delta_vs_fixed_xgboost": difference(games),
        "rmse_delta_vs_fixed_xgboost_lower_95": draws[int(0.025 * (iterations - 1))],
        "rmse_delta_vs_fixed_xgboost_upper_95": draws[int(0.975 * (iterations - 1))],
    }


def _fit_prepared(
    config: dict[str, float | int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[XGBRegressor, int, float]:
    model = XGBRegressor(
        **config,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=MAX_BOOSTING_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
        verbosity=0,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    predictions = model.predict(x_valid).tolist()
    best_iteration = int(model.best_iteration)
    return model, best_iteration, _rmse(y_valid.tolist(), predictions)


def tune_xgboost_nested(
    rows: list[dict[str, str]],
    outer_folds: int = 5,
    inner_folds: int = 3,
    bootstrap_samples: int = 1000,
    min_routes_per_half: int = 20,
    seed: int = 42,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Nested game-held-out tuning, returning OOF predictions and diagnostics.

    The outer games are never used for configuration selection or early stopping.
    Categorical encoding and numeric imputation are independently fit inside
    each inner training fold and again on each complete outer training fold.
    """
    rows = [row for row in rows if row.get("officialPosition") == "WR"]
    if not rows:
        raise ValueError("No WR route rows are available")
    if outer_folds < 2 or inner_folds < 2 or bootstrap_samples < 1:
        raise ValueError("fold counts and bootstrap_samples must be positive")
    numeric, categorical = _features()
    missing = sorted((set(numeric) | set(categorical)) - set(rows[0]))
    if missing:
        raise ValueError(f"Dynamic route table is missing features: {missing}")

    configs = _sample_configs(seed)
    outer_by_game = assign_game_folds(rows, n_folds=outer_folds, seed=seed)
    half_by_game = assign_game_halves(rows, seed=seed + 10_000)
    predictions: list[dict[str, str]] = []
    search_rows: list[dict[str, str]] = []
    selected_rows: list[dict[str, str]] = []
    fold_rows: list[dict[str, str]] = []

    for outer_fold in range(outer_folds):
        outer_train = [
            row for row in rows
            if outer_by_game[(str(int(row["week"])), row["gameId"])] != outer_fold
        ]
        outer_test = [
            row for row in rows
            if outer_by_game[(str(int(row["week"])), row["gameId"])] == outer_fold
        ]
        inner_by_game = assign_game_folds(
            outer_train, n_folds=inner_folds, seed=seed + 1000 + outer_fold
        )
        prepared_inner = []
        for inner_fold in range(inner_folds):
            inner_train = [
                row for row in outer_train
                if inner_by_game[(str(int(row["week"])), row["gameId"])] != inner_fold
            ]
            inner_valid = [
                row for row in outer_train
                if inner_by_game[(str(int(row["week"])), row["gameId"])] == inner_fold
            ]
            preprocessor = _preprocessor(len(numeric), len(categorical))
            x_train = preprocessor.fit_transform(_matrix(inner_train, numeric, categorical))
            x_valid = preprocessor.transform(_matrix(inner_valid, numeric, categorical))
            y_train = np.asarray([target(row) for row in inner_train])
            y_valid = np.asarray([target(row) for row in inner_valid])
            prepared_inner.append((inner_fold, x_train, y_train, x_valid, y_valid))

        config_results = []
        for config_index, config in enumerate(configs):
            fold_scores, best_iterations = [], []
            for inner_fold, x_train, y_train, x_valid, y_valid in prepared_inner:
                _, best_iteration, score = _fit_prepared(
                    config, x_train, y_train, x_valid, y_valid
                )
                fold_scores.append(score)
                best_iterations.append(best_iteration)
                search_rows.append({
                    "outer_fold": str(outer_fold + 1),
                    "config_id": str(config_index + 1),
                    "inner_fold": str(inner_fold + 1),
                    **{key: f"{value:.8g}" for key, value in config.items()},
                    "inner_rmse": f"{score:.8f}",
                    "best_iteration": str(best_iteration + 1),
                })
            mean_score = statistics.fmean(fold_scores)
            config_results.append((mean_score, config_index, config, best_iterations))

        mean_score, config_index, best_config, best_iterations = min(
            config_results, key=lambda item: item[0]
        )
        selected_rounds = max(1, int(round(statistics.median(best_iterations))) + 1)
        selected_rows.append({
            "outer_fold": str(outer_fold + 1),
            "selected_config_id": str(config_index + 1),
            "inner_mean_rmse": f"{mean_score:.8f}",
            "selected_n_estimators": str(selected_rounds),
            **{key: f"{value:.8g}" for key, value in best_config.items()},
        })

        x_outer_train = _matrix(outer_train, numeric, categorical)
        x_outer_test = _matrix(outer_test, numeric, categorical)
        outer_preprocessor = _preprocessor(len(numeric), len(categorical))
        x_outer_train = outer_preprocessor.fit_transform(x_outer_train)
        x_outer_test = outer_preprocessor.transform(x_outer_test)
        y_outer_train = np.asarray([target(row) for row in outer_train])

        ridge = RidgeDynamicContextModel(l2=25, max_levels_per_feature=30)
        ridge.fit(outer_train)
        ridge_values = [ridge.predict(row) for row in outer_test]
        tuned = XGBRegressor(
            **best_config,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=selected_rounds,
            tree_method="hist",
            n_jobs=4,
            random_state=seed + outer_fold,
            verbosity=0,
        )
        tuned.fit(x_outer_train, y_outer_train, verbose=False)
        tuned_values = tuned.predict(x_outer_test).tolist()

        fixed_config = configs[0]
        fixed = XGBRegressor(
            **fixed_config,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=500,
            tree_method="hist",
            n_jobs=4,
            random_state=seed + outer_fold,
            verbosity=0,
        )
        fixed.fit(x_outer_train, y_outer_train, verbose=False)
        fixed_values = fixed.predict(x_outer_test).tolist()

        model_values = {
            "ridge_dynamic_context": ridge_values,
            "xgboost_fixed": fixed_values,
            "xgboost_tuned": tuned_values,
        }
        actual = [target(row) for row in outer_test]
        for model_name, values in model_values.items():
            fold_rows.append({
                "fold": str(outer_fold + 1),
                "model": model_name,
                **regression_metrics(actual, values),
            })
        for index, row in enumerate(outer_test):
            game_key = (str(int(row["week"])), row["gameId"])
            out = {
                "week": row["week"], "gameId": row["gameId"], "playId": row["playId"],
                "nflId": row["nflId"], "displayName": row.get("displayName", ""),
                "fold": str(outer_fold + 1), "reliability_half": half_by_game[game_key],
                "delta_sep": f"{target(row):.8f}",
            }
            for model_name, values in model_values.items():
                value = values[index]
                out[f"pred_delta_sep_{model_name}"] = f"{value:.8f}"
                out[f"soe_route_{model_name}"] = f"{target(row) - value:.8f}"
            predictions.append(out)

    metrics, calibration_rows = [], []
    actual = [float(row["delta_sep"]) for row in predictions]
    for model_name in MODEL_NAMES:
        predicted = [float(row[f"pred_delta_sep_{model_name}"]) for row in predictions]
        calibration, deciles = _calibration(actual, predicted)
        bootstrap = (
            {"rmse_delta_vs_ridge": 0.0, "rmse_delta_lower_95": 0.0, "rmse_delta_upper_95": 0.0}
            if model_name == "ridge_dynamic_context"
            else _paired_game_bootstrap(predictions, model_name, bootstrap_samples, seed)
        )
        metric_row = {
            "model": model_name,
            **regression_metrics(actual, predicted),
            **bootstrap,
            **{key: f"{value:.8f}" for key, value in calibration.items()},
        }
        if model_name == "xgboost_tuned":
            metric_row.update(_paired_rmse_bootstrap(
                predictions,
                "xgboost_tuned",
                "xgboost_fixed",
                bootstrap_samples,
                seed + 101,
            ))
        metrics.append(metric_row)
        calibration_rows.extend({"model": model_name, **row} for row in deciles)

    halves_by_model, reliability_rows = [], []
    for model_name in MODEL_NAMES:
        player_rows, _ = receiver_half_reliability(
            predictions,
            min_routes_per_half=min_routes_per_half,
            bootstrap_samples=0,
            seed=seed,
            model_names=(model_name,),
        )
        halves_by_model.extend({"model": model_name, **row} for row in player_rows)
    paired_players = []
    for player in sorted({row["nflId"] for row in halves_by_model}):
        model_rows = {
            model: next(row for row in halves_by_model if row["model"] == model and row["nflId"] == player)
            for model in MODEL_NAMES
        }
        out = {"nflId": player, "displayName": model_rows[MODEL_NAMES[0]]["displayName"]}
        for model, row in model_rows.items():
            out.update({key: value for key, value in row.items() if model in key})
        paired_players.append(out)
    reliability_rows = reliability_metrics(
        paired_players,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        model_names=MODEL_NAMES,
    )
    tuned_vs_ridge = reliability_metrics(
        paired_players,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        model_names=("ridge_dynamic_context", "xgboost_tuned"),
    )[-1]
    reliability_rows.append(tuned_vs_ridge)

    return (
        predictions,
        metrics,
        fold_rows,
        calibration_rows,
        search_rows,
        selected_rows,
        reliability_rows,
    )
