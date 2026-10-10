from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .models import RidgeDynamicContextModel, regression_metrics, target
from .validation import (
    assign_game_folds,
    assign_game_halves,
    pearson_correlation,
    receiver_half_reliability,
    reliability_metrics,
)
from .utils import parse_float


MODEL_NAMES = ("ridge_dynamic_context", "extra_trees", "hist_gradient_boosting")


def _features() -> tuple[list[str], list[str]]:
    return RidgeDynamicContextModel.numeric_features, RidgeDynamicContextModel.categorical_features


def _matrix(
    rows: list[dict[str, str]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                *[
                    np.nan if row.get(name, "") in {"", "NA"} else parse_float(row[name])
                    for name in numeric_features
                ],
                *[row.get(name, "") or "__missing__" for name in categorical_features],
            ]
            for row in rows
        ],
        dtype=object,
    )


def _preprocessor(n_numeric: int, n_categorical: int) -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="mean", keep_empty_features=True))]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore", max_categories=30, sparse_output=False
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(range(n_numeric))),
            ("categorical", categorical, list(range(n_numeric, n_numeric + n_categorical))),
        ],
        sparse_threshold=0,
    )


def _calibration(
    actual: list[float], predicted: list[float]
) -> tuple[dict[str, float], list[dict[str, str]]]:
    mean_actual, mean_predicted = statistics.fmean(actual), statistics.fmean(predicted)
    variance = sum((value - mean_predicted) ** 2 for value in predicted)
    slope = (
        sum(
            (p - mean_predicted) * (y - mean_actual)
            for p, y in zip(predicted, actual)
        )
        / variance
        if variance
        else 0.0
    )
    ordered = sorted(zip(predicted, actual))
    bins = []
    for i in range(10):
        group = ordered[i * len(ordered) // 10 : (i + 1) * len(ordered) // 10]
        pred = statistics.fmean(pair[0] for pair in group)
        obs = statistics.fmean(pair[1] for pair in group)
        bins.append(
            {
                "decile": str(i + 1),
                "routes": str(len(group)),
                "mean_predicted": f"{pred:.8f}",
                "mean_actual": f"{obs:.8f}",
                "mean_residual": f"{obs - pred:.8f}",
            }
        )
    return {
        "calibration_slope": slope,
        "calibration_intercept": mean_actual - slope * mean_predicted,
        "mean_absolute_decile_bias": statistics.fmean(
            abs(float(row["mean_residual"])) for row in bins
        ),
    }, bins


def _paired_game_bootstrap(
    predictions: list[dict[str, str]], model: str, iterations: int, seed: int
) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in predictions:
        y = float(row["delta_sep"])
        grouped[row["gameId"]].append(
            (
                (y - float(row[f"pred_delta_sep_{model}"])) ** 2,
                (y - float(row["pred_delta_sep_ridge_dynamic_context"])) ** 2,
            )
        )
    games = list(grouped)
    if len(games) < 2:
        raise ValueError("At least two games are required for paired game bootstrap")

    def delta(sample):
        n = sum(len(grouped[g]) for g in sample)
        se_model = sum(sum(x[0] for x in grouped[g]) for g in sample)
        se_ridge = sum(sum(x[1] for x in grouped[g]) for g in sample)
        return math.sqrt(se_model / n) - math.sqrt(se_ridge / n)

    rng = random.Random(seed)
    values = sorted(
        delta([rng.choice(games) for _ in games]) for _ in range(iterations)
    )
    return {
        "rmse_delta_vs_ridge": delta(games),
        "rmse_delta_lower_95": values[int(0.025 * (iterations - 1))],
        "rmse_delta_upper_95": values[int(0.975 * (iterations - 1))],
    }


def compare_algorithms(
    rows: list[dict[str, str]],
    n_folds: int = 5,
    min_routes_per_half: int = 20,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Compare nonlinear regressors with dynamic ridge on identical game-held-out folds."""
    rows = [row for row in rows if row.get("officialPosition") == "WR"]
    if not rows:
        raise ValueError("No WR route rows are available")
    if n_folds < 2 or min_routes_per_half < 1 or bootstrap_samples < 1:
        raise ValueError("folds, minimum routes, and bootstrap samples must be positive")
    numeric, categorical = _features()
    missing = sorted((set(numeric) | set(categorical)) - set(rows[0]))
    if missing:
        raise ValueError(f"Dynamic route table is missing features: {missing}")
    fold_by_game = assign_game_folds(rows, n_folds=n_folds, seed=seed)
    half_by_game = assign_game_halves(rows, seed=seed + 10_000)
    predictions, fold_metrics = [], []
    for fold in range(n_folds):
        train_rows = [
            r for r in rows
            if fold_by_game[(str(int(r["week"])), r["gameId"])] != fold
        ]
        test_rows = [
            r for r in rows
            if fold_by_game[(str(int(r["week"])), r["gameId"])] == fold
        ]
        ridge = RidgeDynamicContextModel(l2=25, max_levels_per_feature=30)
        ridge.fit(train_rows)
        fold_predictions = {"ridge_dynamic_context": [ridge.predict(r) for r in test_rows]}
        x_train = _matrix(train_rows, numeric, categorical)
        x_test = _matrix(test_rows, numeric, categorical)
        preprocessor = _preprocessor(len(numeric), len(categorical))
        x_train = preprocessor.fit_transform(x_train)
        x_test = preprocessor.transform(x_test)
        y_train = np.asarray([target(r) for r in train_rows])
        estimators = (
            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=250,
                    min_samples_leaf=10,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=seed + fold,
                ),
            ),
            (
                "hist_gradient_boosting",
                HistGradientBoostingRegressor(
                    max_iter=180,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    learning_rate=0.08,
                    early_stopping=False,
                    random_state=seed + fold,
                ),
            ),
        )
        for name, estimator in estimators:
            estimator.fit(x_train, y_train)
            fold_predictions[name] = estimator.predict(x_test).tolist()
        actual = [target(r) for r in test_rows]
        for name, values in fold_predictions.items():
            metrics, _ = _calibration(actual, values)
            fold_metrics.append({
                "fold": str(fold + 1), "model": name, "routes": str(len(test_rows)),
                **regression_metrics(actual, values),
                **{key: f"{value:.8f}" for key, value in metrics.items()},
            })
        for i, row in enumerate(test_rows):
            key = (str(int(row["week"])), row["gameId"])
            out = {
                "gameId": row["gameId"], "playId": row["playId"], "nflId": row["nflId"],
                "displayName": row.get("displayName", ""), "week": row["week"],
                "fold": str(fold + 1), "reliability_half": half_by_game[key],
                "delta_sep": f"{target(row):.8f}",
            }
            for name, values in fold_predictions.items():
                out[f"pred_delta_sep_{name}"] = f"{values[i]:.8f}"
                out[f"soe_route_{name}"] = f"{target(row) - values[i]:.8f}"
            predictions.append(out)

    metrics, calibration_rows = [], []
    for name in MODEL_NAMES:
        actual = [float(r["delta_sep"]) for r in predictions]
        predicted = [float(r[f"pred_delta_sep_{name}"]) for r in predictions]
        calibration, deciles = _calibration(actual, predicted)
        bootstrap = (
            {
                "rmse_delta_vs_ridge": 0.0,
                "rmse_delta_lower_95": 0.0,
                "rmse_delta_upper_95": 0.0,
            }
            if name == "ridge_dynamic_context"
            else _paired_game_bootstrap(predictions, name, bootstrap_samples, seed)
        )
        metrics.append({
            "model": name, **regression_metrics(actual, predicted), **bootstrap,
            **{key: f"{value:.8f}" for key, value in calibration.items()},
        })
        calibration_rows.extend({"model": name, **row} for row in deciles)
    halves, reliability = [], []
    for model in MODEL_NAMES:
        player_rows, _ = receiver_half_reliability(
            predictions, min_routes_per_half=min_routes_per_half,
            bootstrap_samples=0, seed=seed, model_names=(model,),
        )
        halves.extend({"model": model, **row} for row in player_rows)
    for model in MODEL_NAMES:
        if model == "ridge_dynamic_context":
            summary = reliability_metrics(
                halves_for_model(halves, model),
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                model_names=(model,),
            )
            reliability.append(
                {
                    "model": model,
                    **summary[0],
                    "pearson_delta_vs_ridge": "0",
                    "pearson_delta_lower_95": "0",
                    "pearson_delta_upper_95": "0",
                    "spearman_delta_vs_ridge": "0",
                    "spearman_delta_lower_95": "0",
                    "spearman_delta_upper_95": "0",
                }
            )
        else:
            paired = paired_player_rows(halves, "ridge_dynamic_context", model)
            summary = reliability_metrics(
                paired,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                model_names=("ridge_dynamic_context", model),
            )
            absolute = reliability_metrics(
                halves_for_model(halves, model),
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                model_names=(model,),
            )[0]
            delta = summary[-1]
            reliability.append(
                {
                    "model": model,
                    **absolute,
                    "pearson_delta_vs_ridge": delta["pearson"],
                    "pearson_delta_lower_95": delta["pearson_ci_lower"],
                    "pearson_delta_upper_95": delta["pearson_ci_upper"],
                    "spearman_delta_vs_ridge": delta["spearman"],
                    "spearman_delta_lower_95": delta["spearman_ci_lower"],
                    "spearman_delta_upper_95": delta["spearman_ci_upper"],
                }
            )
    return predictions, metrics, fold_metrics, calibration_rows, halves, reliability


def halves_for_model(rows: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key in {"nflId", "displayName"} or model in key
        }
        for row in rows
        if row["model"] == model
    ]


def paired_player_rows(
    rows: list[dict[str, str]], baseline: str, candidate: str
) -> list[dict[str, str]]:
    by_model = {
        name: {row["nflId"]: row for row in rows if row["model"] == name}
        for name in (baseline, candidate)
    }
    common = sorted(set(by_model[baseline]) & set(by_model[candidate]))
    output = []
    for nfl_id in common:
        row = {"nflId": nfl_id, "displayName": by_model[baseline][nfl_id]["displayName"]}
        for model in (baseline, candidate):
            row.update(
                {
                    key: value
                    for key, value in by_model[model][nfl_id].items()
                    if model in key
                }
            )
        output.append(row)
    return output
