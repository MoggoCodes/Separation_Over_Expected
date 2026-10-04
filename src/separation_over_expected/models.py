from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .utils import parse_float


def target(row: dict[str, str]) -> float:
    return parse_float(row["delta_sep"])


def split_name(row: dict[str, str]) -> str:
    week = int(row["week"])
    if week <= 6:
        return "train"
    if week == 7:
        return "validation"
    return "test"


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {
        "train": [row for row in rows if split_name(row) == "train"],
        "validation": [row for row in rows if split_name(row) == "validation"],
        "test": [row for row in rows if split_name(row) == "test"],
    }


def default_baseline_models() -> list[object]:
    return [
        GlobalMeanModel(),
        SmoothedGroupMeanModel(
            keys=["officialPosition", "pff_positionLinedUp", "pff_passCoverageType"],
            alpha=25.0,
        ),
        RidgeContextModel(l2=25.0, max_levels_per_feature=30),
    ]


def fit_models(
    train_rows: list[dict[str, str]], models: list[object] | None = None
) -> list[object]:
    fitted_models = models or default_baseline_models()
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
            values.append((index, (parse_float(row[feature]) - mean) / stdev))
            index += 1
        for feature in self.categorical_features:
            levels = self.categorical_levels[feature]
            for level in sorted(levels):
                if row[feature] == level:
                    values.append((index, 1.0))
                index += 1
        return values


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
