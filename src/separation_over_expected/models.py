from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from .feature_schema import (
    DYNAMIC_CONTEXT_FEATURES,
    POCKET_CONTEXT_FEATURES,
    PRESSURE_CONTEXT_FEATURES,
    ROUTE_GEOMETRY_FEATURES,
)
from .utils import parse_float


def target(row: dict[str, str]) -> float:
    return parse_float(row["delta_sep"])


def split_name(row: dict[str, str]) -> str:
    if row.get("split") in {"train", "validation", "test"}:
        return row["split"]
    week = int(row["week"])
    if week <= 6:
        return "train"
    if week == 7:
        return "validation"
    return "test"


def split_rows(
    rows: list[dict[str, str]],
    strategy: str = "week",
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[dict[str, str]]]:
    if strategy in {"random", "game"}:
        validate_split_fractions(train_fraction, validation_fraction)
        if len(rows) < 3:
            raise ValueError(
                "At least 3 rows are required for a train/validation/test split"
            )
        assignments = {}
        if strategy == "random":
            indices = list(range(len(rows)))
            random.Random(seed).shuffle(indices)
            train_end = max(1, min(len(rows) - 2, int(len(rows) * train_fraction)))
            validation_end = max(
                train_end + 1,
                min(len(rows) - 1, train_end + int(len(rows) * validation_fraction)),
            )
            for index in indices[:train_end]:
                assignments[index] = "train"
            for index in indices[train_end:validation_end]:
                assignments[index] = "validation"
            for index in indices[validation_end:]:
                assignments[index] = "test"
            keys = list(range(len(rows)))
        else:
            games_by_week: dict[int, set[str]] = defaultdict(set)
            for row in rows:
                games_by_week[int(row["week"])].add(row["gameId"])
            for week, game_ids in sorted(games_by_week.items()):
                ordered_games = sorted(game_ids)
                if len(ordered_games) < 3:
                    raise ValueError(
                        f"Week {week} needs at least 3 games for a game-level split"
                    )
                random.Random(seed + week).shuffle(ordered_games)
                train_end = max(
                    1,
                    min(
                        len(ordered_games) - 2,
                        int(len(ordered_games) * train_fraction + 0.5),
                    ),
                )
                validation_count = max(
                    1, int(len(ordered_games) * validation_fraction + 0.5)
                )
                validation_end = max(
                    train_end + 1,
                    min(
                        len(ordered_games) - 1,
                        train_end + validation_count,
                    ),
                )
                for game_id in ordered_games[:train_end]:
                    assignments[(week, game_id)] = "train"
                for game_id in ordered_games[train_end:validation_end]:
                    assignments[(week, game_id)] = "validation"
                for game_id in ordered_games[validation_end:]:
                    assignments[(week, game_id)] = "test"
            keys = [(int(row["week"]), row["gameId"]) for row in rows]
        labeled_rows = []
        for row, key in zip(rows, keys):
            labeled = dict(row)
            labeled["split"] = assignments[key]
            labeled_rows.append(labeled)
        return {
            split: [row for row in labeled_rows if row["split"] == split]
            for split in ("train", "validation", "test")
        }
    if strategy != "week":
        raise ValueError(f"Unknown split strategy: {strategy}")
    return {
        "train": [row for row in rows if split_name(row) == "train"],
        "validation": [row for row in rows if split_name(row) == "validation"],
        "test": [row for row in rows if split_name(row) == "test"],
    }


def validate_split_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")


def filter_rows(
    rows: list[dict[str, str]],
    position: str | None = None,
    min_week: int | None = None,
    max_week: int | None = None,
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        week = int(row["week"])
        if position is not None and row["officialPosition"] != position:
            continue
        if min_week is not None and week < min_week:
            continue
        if max_week is not None and week > max_week:
            continue
        filtered.append(row)
    return filtered


def default_baseline_models(include_dynamic_context: bool = False) -> list[object]:
    models = [
        GlobalMeanModel(),
        SmoothedGroupMeanModel(
            keys=["officialPosition", "pff_positionLinedUp", "pff_passCoverageType"],
            alpha=25.0,
        ),
        RidgeContextModel(l2=25.0, max_levels_per_feature=30),
    ]
    if include_dynamic_context:
        models.append(RidgeDynamicContextModel(l2=25.0, max_levels_per_feature=30))
    return models


def fit_models(
    train_rows: list[dict[str, str]],
    models: list[object] | None = None,
    include_dynamic_context: bool = False,
) -> list[object]:
    fitted_models = models or default_baseline_models(include_dynamic_context)
    for model in fitted_models:
        model.fit(train_rows)
    return fitted_models


def evaluate_models(
    models: list[object], splits: dict[str, list[dict[str, str]]]
) -> list[dict[str, str]]:
    rows = []
    for split, split_data in splits.items():
        for model in models:
            actual = [target(row) for row in split_data]
            predicted = [model.predict(row) for row in split_data]
            rows.append(
                {
                    "model": model.name,
                    "split": split,
                    "rows": str(len(split_data)),
                    **regression_metrics(actual, predicted),
                }
            )
    return rows


def add_model_predictions(
    row: dict[str, str], models: list[object], scoring_model: object
) -> dict[str, str]:
    out = dict(row)
    for model in models:
        out[f"pred_delta_sep_{model.name}"] = f"{model.predict(row):.3f}"
    out["soe_route"] = f"{target(row) - scoring_model.predict(row):.3f}"
    if any(model.name == "ridge_dynamic_context" for model in models):
        static_model = next(model for model in models if model.name == "ridge_context")
        dynamic_model = next(
            model for model in models if model.name == "ridge_dynamic_context"
        )
        out["soe_route_ridge_context"] = f"{target(row) - static_model.predict(row):.3f}"
        out["soe_route_ridge_dynamic_context"] = (
            f"{target(row) - dynamic_model.predict(row):.3f}"
        )
    out["split"] = split_name(row)
    return out


class GlobalMeanModel:
    name = "global_mean"

    def fit(self, rows: list[dict[str, str]]) -> None:
        self.mean = statistics.fmean(target(row) for row in rows)

    def predict(self, row: dict[str, str]) -> float:
        return self.mean


class SmoothedGroupMeanModel:
    name = "smoothed_group_mean"

    def __init__(self, keys: list[str], alpha: float) -> None:
        self.keys = keys
        self.alpha = alpha

    def fit(self, rows: list[dict[str, str]]) -> None:
        self.global_mean = statistics.fmean(target(row) for row in rows)
        sums: dict[tuple[str, ...], float] = defaultdict(float)
        counts: dict[tuple[str, ...], int] = defaultdict(int)
        for row in rows:
            key = tuple(row[column] for column in self.keys)
            sums[key] += target(row)
            counts[key] += 1
        self.group_means = {
            key: (sums[key] + self.alpha * self.global_mean) / (counts[key] + self.alpha)
            for key in sums
        }

    def predict(self, row: dict[str, str]) -> float:
        key = tuple(row[column] for column in self.keys)
        return self.group_means.get(key, self.global_mean)


class RidgeContextModel:
    name = "ridge_context"
    impute_missing_numeric = False

    numeric_features = [
        "sep_snap",
        "snap_x_norm",
        "snap_y_norm",
        "nearest_defender_dx_snap",
        "nearest_defender_dy_snap",
        "receiver_speed_snap",
        "receiver_accel_snap",
        "time_to_throw_frames",
        "down",
        "yardsToGo",
        "absoluteYardlineNumber",
        "defendersInBox",
        "pff_playAction",
        "defender_2_dist_snap",
        "defender_3_dist_snap",
        "defenders_within_3_snap",
        "defenders_within_5_snap",
        "defenders_within_10_snap",
        "nearest_db_dist_snap",
        "nearest_lb_dist_snap",
        "nearest_defender_depth_leverage_snap",
        "nearest_defender_width_leverage_snap",
        "defender_depth_density_0_10_snap",
        "defender_inside_count_5_snap",
        "defender_outside_count_5_snap",
    ]
    categorical_features = [
        "officialPosition",
        "pff_positionLinedUp",
        "offenseFormation",
        "personnelO",
        "personnelD",
        "dropBackType",
        "pff_passCoverage",
        "pff_passCoverageType",
    ]

    def __init__(self, l2: float, max_levels_per_feature: int) -> None:
        self.l2 = l2
        self.max_levels_per_feature = max_levels_per_feature

    def fit(self, rows: list[dict[str, str]]) -> None:
        self.numeric_stats = {}
        for feature in self.numeric_features:
            if self.impute_missing_numeric:
                observed = [
                    parse_float(row[feature])
                    for row in rows
                    if row.get(feature, "") not in {"", "NA"}
                ]
                mean = statistics.fmean(observed) if observed else 0.0
                values = [
                    parse_float(row[feature])
                    if row.get(feature, "") not in {"", "NA"}
                    else mean
                    for row in rows
                ]
            else:
                values = [parse_float(row[feature]) for row in rows]
            mean = statistics.fmean(values)
            stdev = statistics.pstdev(values) or 1.0
            self.numeric_stats[feature] = (mean, stdev)

        self.categorical_levels = {}
        for feature in self.categorical_features:
            counts = defaultdict(int)
            for row in rows:
                counts[row[feature]] += 1
            self.categorical_levels[feature] = {
                level
                for level, _ in sorted(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )[: self.max_levels_per_feature]
            }

        self.feature_names = ["intercept"]
        self.feature_names.extend(f"num:{feature}" for feature in self.numeric_features)
        for feature in self.categorical_features:
            for level in sorted(self.categorical_levels[feature]):
                self.feature_names.append(f"cat:{feature}={level}")

        p = len(self.feature_names)
        xtx = [[0.0 for _ in range(p)] for _ in range(p)]
        xty = [0.0 for _ in range(p)]
        for row in rows:
            features = self.features(row)
            y = target(row)
            for i, xi in features:
                xty[i] += xi * y
                for j, xj in features:
                    xtx[i][j] += xi * xj

        for i in range(1, p):
            xtx[i][i] += self.l2

        self.coefficients = solve_linear_system(xtx, xty)

    def predict(self, row: dict[str, str]) -> float:
        return sum(self.coefficients[i] * value for i, value in self.features(row))

    def features(self, row: dict[str, str]) -> list[tuple[int, float]]:
        values = [(0, 1.0)]
        index = 1
        for feature in self.numeric_features:
            mean, stdev = self.numeric_stats[feature]
            value = row.get(feature, "")
            numeric_value = (
                mean
                if self.impute_missing_numeric and value in {"", "NA"}
                else parse_float(value)
            )
            values.append((index, (numeric_value - mean) / stdev))
            index += 1
        for feature in self.categorical_features:
            levels = self.categorical_levels[feature]
            for level in sorted(levels):
                if row[feature] == level:
                    values.append((index, 1.0))
                index += 1
        return values


class RidgeDynamicContextModel(RidgeContextModel):
    name = "ridge_dynamic_context"
    impute_missing_numeric = True
    numeric_features = [*RidgeContextModel.numeric_features, *DYNAMIC_CONTEXT_FEATURES]


class RidgePocketContextModel(RidgeDynamicContextModel):
    name = "ridge_pocket_context"
    numeric_features = [
        *RidgeDynamicContextModel.numeric_features,
        *POCKET_CONTEXT_FEATURES,
    ]


class RidgePressureContextModel(RidgeDynamicContextModel):
    name = "ridge_pressure_context"
    numeric_features = [
        *RidgeDynamicContextModel.numeric_features,
        *PRESSURE_CONTEXT_FEATURES,
    ]


class RidgeRouteGeometryModel(RidgeDynamicContextModel):
    name = "ridge_dynamic_geometry_context"
    numeric_features = [
        *RidgeDynamicContextModel.numeric_features,
        *ROUTE_GEOMETRY_FEATURES,
    ]


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, str]:
    n = len(actual)
    residuals = [y - yhat for y, yhat in zip(actual, predicted)]
    mae = statistics.fmean(abs(e) for e in residuals)
    rmse = math.sqrt(statistics.fmean(e * e for e in residuals))
    ybar = statistics.fmean(actual)
    sse = sum(e * e for e in residuals)
    sst = sum((y - ybar) ** 2 for y in actual)
    r2 = 1.0 - sse / sst if sst else 0.0
    bias = statistics.fmean(residuals)
    return {
        "mae": f"{mae:.3f}",
        "rmse": f"{rmse:.3f}",
        "r2": f"{r2:.3f}",
        "bias": f"{bias:.3f}",
        "mean_actual": f"{ybar:.3f}",
        "mean_predicted": f"{statistics.fmean(predicted):.3f}",
        "n": str(n),
    }


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    matrix = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            continue
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        pivot_value = matrix[col][col]
        for j in range(col, n + 1):
            matrix[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                matrix[row][j] -= factor * matrix[col][j]
    return [matrix[i][n] for i in range(n)]
