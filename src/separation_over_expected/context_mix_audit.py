from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

from .validation import (
    assign_game_halves,
    average_ranks,
    pearson_correlation,
    percentile,
)


DEFAULT_COHORTS = ((10, 3), (20, 5), (30, 5))
CONTEXT_FEATURES = (
    "down",
    "yards_to_go_bin",
    "yardline_bin",
    "start_separation_bin",
    "start_width_bin",
    "defender_depth_leverage_bin",
    "defender_width_leverage_bin",
    "receiver_start_speed_bin",
)


def _number(row: dict[str, str], name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bin(value: float | None, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if value is None:
        return "Unknown"
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def context_categories(row: dict[str, str]) -> dict[str, str]:
    """Coarse pre-release context categories; excludes route outcomes and labels."""
    ytg = _number(row, "yardsToGo")
    yardline = _number(row, "yardline_100")
    separation = _number(row, "sep_snap")
    width = _number(row, "snap_y_norm")
    depth_leverage = _number(row, "nearest_defender_dx_snap")
    width_leverage = _number(row, "nearest_defender_dy_snap")
    speed = _number(row, "receiver_speed_snap")
    try:
        down = str(int(float(row["down"])))
    except (KeyError, TypeError, ValueError):
        down = "Unknown"
    return {
        "down": down,
        "yards_to_go_bin": _bin(ytg, (4, 7, 11, 16), ("1-3", "4-6", "7-10", "11-15", "16+")),
        "yardline_bin": _bin(yardline, (10, 20, 40, 60, 80, 90), ("0-9", "10-19", "20-39", "40-59", "60-79", "80-89", "90-100")),
        "start_separation_bin": _bin(separation, (2, 4, 6, 10), ("<2", "2-3.9", "4-5.9", "6-9.9", "10+")),
        "start_width_bin": _bin(width, (5, 12, 20, 33, 41, 48), ("0-4.9", "5-11.9", "12-19.9", "20-32.9", "33-40.9", "41-47.9", "48+")),
        "defender_depth_leverage_bin": _bin(depth_leverage, (-6, -3, -1, 1, 3, 6), ("<-6", "-6--3", "-3--1", "-1-1", "1-3", "3-6", "6+")),
        "defender_width_leverage_bin": _bin(width_leverage, (-6, -3, -1, 1, 3, 6), ("<-6", "-6--3", "-3--1", "-1-1", "1-3", "3-6", "6+")),
        "receiver_start_speed_bin": _bin(speed, (0.5, 2, 4), ("<0.5", "0.5-1.9", "2-3.9", "4+")),
    }


class _ContextResidualRidge:
    """Fixed-penalty additive context model for residual diagnostics only."""

    def __init__(self, alpha: float = 25.0):
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = Ridge(alpha=alpha)

    def fit(self, rows: list[dict[str, str]], centered_residuals: list[float]) -> None:
        x = self.vectorizer.fit_transform([context_categories(row) for row in rows])
        self.model.fit(x, centered_residuals)

    def predict(self, rows: list[dict[str, str]]) -> list[float]:
        x = self.vectorizer.transform([context_categories(row) for row in rows])
        return [float(value) for value in self.model.predict(x)]


def _corr(xs: list[float], ys: list[float], metric: str) -> float:
    if metric == "pearson":
        return pearson_correlation(xs, ys)
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def audit_context_mix(
    context_rows: list[dict[str, str]],
    predictions: list[dict[str, str]],
    cohorts: tuple[tuple[int, int], ...] = DEFAULT_COHORTS,
    split_seeds: int = 100,
    bootstrap_samples: int = 300,
    seed: int = 42,
    ridge_alpha: float = 25.0,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Cross-fit a pre-release context correction and compare receiver repeatability.

    The base dynamic Ridge predictions are never changed. The context learner is
    trained on the route residuals from one set of games and predicts residual
    context patterns in the other set, then vice versa. It contains no receiver
    identity, route depth, route labels, or post-release data.
    """
    if split_seeds < 1 or bootstrap_samples < 1 or ridge_alpha <= 0:
        raise ValueError("split_seeds, bootstrap_samples, and ridge_alpha must be positive")
    if not cohorts or any(route_count < 1 or game_count < 1 for route_count, game_count in cohorts):
        raise ValueError("cohorts must contain positive route and game thresholds")
    required_context = {
        "gameId", "playId", "nflId", "week", "down", "yardsToGo", "yardline_100",
        "sep_snap", "snap_y_norm", "nearest_defender_dx_snap",
        "nearest_defender_dy_snap", "receiver_speed_snap",
    }
    required_prediction = {"season", "gameId", "playId", "nflId", "week", "residual_dynamic"}
    wr_context_rows = [row for row in context_rows if row.get("officialPosition", "WR") == "WR"]
    context_keys = [(row["gameId"], row["playId"], row["nflId"]) for row in wr_context_rows]
    context_by_key = dict(zip(context_keys, wr_context_rows))
    if not context_by_key:
        raise ValueError("No WR context rows were provided")
    if len(context_by_key) != len(wr_context_rows):
        raise ValueError("2023 WR context rows contain duplicate route keys")
    if any(not required_context.issubset(row) for row in context_by_key.values()):
        raise ValueError(f"Context rows must contain {sorted(required_context)}")
    predictions_2023 = [row for row in predictions if row.get("season") == "2023"]
    if not predictions_2023:
        raise ValueError("No 2023 route predictions were provided")
    if any(not required_prediction.issubset(row) for row in predictions_2023):
        raise ValueError(f"2023 predictions must contain {sorted(required_prediction)}")
    prediction_by_key = {
        (row["gameId"], row["playId"], row["nflId"]): row
        for row in predictions_2023
    }
    if len(prediction_by_key) != len(predictions_2023):
        raise ValueError("2023 predictions contain duplicate route keys")
    if set(context_by_key) != set(prediction_by_key):
        raise ValueError(
            "2023 WR context rows and frozen predictions have different route keys "
            f"(context-only={len(set(context_by_key)-set(prediction_by_key))}, "
            f"prediction-only={len(set(prediction_by_key)-set(context_by_key))})"
        )

    rows: list[dict[str, str]] = []
    for key, context in context_by_key.items():
        prediction = prediction_by_key[key]
        if context["week"] != prediction["week"]:
            raise ValueError(f"Week mismatch for route {key}")
        residual = float(prediction["residual_dynamic"])
        if not math.isfinite(residual):
            raise ValueError(f"Invalid dynamic residual for route {key}")
        rows.append({**context, **prediction})

    split_rows: list[dict[str, str]] = []
    fit_rows: list[dict[str, str]] = []
    values: dict[tuple[int, int], dict[str, list[float]]] = {
        cohort: {
            f"{metric}_{version}": []
            for metric in ("pearson", "spearman")
            for version in ("raw", "adjusted", "change", "bootstrap_raw", "bootstrap_adjusted", "bootstrap_change")
        } | {"receivers": [], "raw_rmse": [], "adjusted_rmse": []}
        for cohort in cohorts
    }
    for split_index in range(split_seeds):
        split_seed = seed + 10_000 + split_index * 37
        rng = random.Random(split_seed + 1)
        halves = assign_game_halves(rows, seed=split_seed)
        split_rows_by_half = {
            half: [row for row in rows if halves[(str(int(row["week"])), row["gameId"])] == half]
            for half in ("A", "B")
        }
        adjusted_rows: dict[str, list[dict[str, str | float]]] = {"A": [], "B": []}
        raw_rmse_sums = 0.0
        adjusted_rmse_sums = 0.0
        n_routes = 0
        for train_half, test_half in (("A", "B"), ("B", "A")):
            train_rows = split_rows_by_half[train_half]
            test_rows = split_rows_by_half[test_half]
            train_residuals = [float(row["residual_dynamic"]) for row in train_rows]
            train_mean = statistics.fmean(train_residuals)
            centered_train = [value - train_mean for value in train_residuals]
            adjuster = _ContextResidualRidge(ridge_alpha)
            adjuster.fit(train_rows, centered_train)
            predicted_context = adjuster.predict(test_rows)
            test_residuals = [float(row["residual_dynamic"]) for row in test_rows]
            test_mean = statistics.fmean(test_residuals)
            centered_test = [value - test_mean for value in test_residuals]
            prediction_mean = statistics.fmean(predicted_context)
            centered_prediction = [value - prediction_mean for value in predicted_context]
            for row, raw, correction in zip(test_rows, centered_test, centered_prediction):
                adjusted_rows[test_half].append({
                    **row,
                    "raw_centered_residual": raw,
                    "context_adjusted_residual": raw - correction,
                })
                raw_rmse_sums += raw * raw
                adjusted_rmse_sums += (raw - correction) ** 2
                n_routes += 1
        raw_rmse = math.sqrt(raw_rmse_sums / n_routes)
        adjusted_rmse = math.sqrt(adjusted_rmse_sums / n_routes)
        fit_rows.append({
            "seed": str(split_seed),
            "routes": str(n_routes),
            "raw_centered_residual_rmse": f"{raw_rmse:.8f}",
            "adjusted_centered_residual_rmse": f"{adjusted_rmse:.8f}",
            "rmse_change_adjusted_minus_raw": f"{adjusted_rmse - raw_rmse:.8f}",
        })

        per_player: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"raw": [], "adjusted": []}
        )
        games_per_player_half: dict[tuple[str, str], set[str]] = defaultdict(set)
        for half in ("A", "B"):
            for row in adjusted_rows[half]:
                key = (str(row["nflId"]), half)
                per_player[key]["raw"].append(float(row["raw_centered_residual"]))
                per_player[key]["adjusted"].append(float(row["context_adjusted_residual"]))
                games_per_player_half[key].add(str(row["gameId"]))
        player_ids = sorted({player for player, _ in per_player})
        player_means = {
            (player, half, version): statistics.fmean(per_player[(player, half)][version])
            for player in player_ids
            for half in ("A", "B")
            for version in ("raw", "adjusted")
            if per_player[(player, half)][version]
        }
        for min_routes, min_games in cohorts:
            eligible = [
                player for player in player_ids
                if all(
                    len(per_player[(player, half)]["raw"]) >= min_routes
                    and len(games_per_player_half[(player, half)]) >= min_games
                    for half in ("A", "B")
                )
            ]
            metric_values: dict[str, float] = {}
            bootstrap_values: dict[str, list[float]] = defaultdict(list)
            if len(eligible) >= 3:
                for metric in ("pearson", "spearman"):
                    for version in ("raw", "adjusted"):
                        xs = [player_means[(player, "A", version)] for player in eligible]
                        ys = [player_means[(player, "B", version)] for player in eligible]
                        metric_values[f"{metric}_{version}"] = _corr(xs, ys, metric)
                    metric_values[f"{metric}_change"] = metric_values[f"{metric}_adjusted"] - metric_values[f"{metric}_raw"]
                for _ in range(bootstrap_samples):
                    indices = [rng.randrange(len(eligible)) for _ in eligible]
                    for metric in ("pearson", "spearman"):
                        corr_by_version = {}
                        for version in ("raw", "adjusted"):
                            xs = [player_means[(eligible[i], "A", version)] for i in indices]
                            ys = [player_means[(eligible[i], "B", version)] for i in indices]
                            corr_by_version[version] = _corr(xs, ys, metric)
                            bootstrap_values[f"{metric}_bootstrap_{version}"].append(corr_by_version[version])
                        bootstrap_values[f"{metric}_bootstrap_change"].append(
                            corr_by_version["adjusted"] - corr_by_version["raw"]
                        )
            cohort_values = values[(min_routes, min_games)]
            cohort_values["receivers"].append(float(len(eligible)))
            cohort_values["raw_rmse"].append(raw_rmse)
            cohort_values["adjusted_rmse"].append(adjusted_rmse)
            row: dict[str, str] = {
                "seed": str(split_seed),
                "min_routes_per_half": str(min_routes),
                "min_games_per_half": str(min_games),
                "eligible_receivers": str(len(eligible)),
            }
            for metric in ("pearson", "spearman"):
                for version in ("raw", "adjusted", "change"):
                    value = metric_values.get(f"{metric}_{version}")
                    row[f"{metric}_{version}"] = "" if value is None else f"{value:.8f}"
                    if value is not None:
                        cohort_values[f"{metric}_{version}"].append(value)
                for version in ("raw", "adjusted", "change"):
                    cohort_values[f"{metric}_bootstrap_{version}"].extend(
                        bootstrap_values[f"{metric}_bootstrap_{version}"]
                    )
            split_rows.append(row)

    summary_rows: list[dict[str, str]] = []
    for min_routes, min_games in cohorts:
        cohort_values = values[(min_routes, min_games)]
        summary: dict[str, str] = {
            "min_routes_per_half": str(min_routes),
            "min_games_per_half": str(min_games),
            "valid_splits": str(len(cohort_values["pearson_raw"])),
        }
        for metric in ("pearson", "spearman"):
            for version in ("raw", "adjusted", "change"):
                key = f"{metric}_{version}"
                ordered = sorted(cohort_values[key])
                summary[f"{key}_median"] = f"{statistics.median(ordered):.8f}" if ordered else ""
                summary[f"{key}_p10"] = f"{percentile(ordered, .10):.8f}" if ordered else ""
                summary[f"{key}_p90"] = f"{percentile(ordered, .90):.8f}" if ordered else ""
                boot = sorted(cohort_values[f"{metric}_bootstrap_{version}"])
                summary[f"{key}_bootstrap_ci_lower"] = f"{percentile(boot, .025):.8f}" if boot else ""
                summary[f"{key}_bootstrap_ci_upper"] = f"{percentile(boot, .975):.8f}" if boot else ""
        for metric in ("receivers", "raw_rmse", "adjusted_rmse"):
            ordered = sorted(cohort_values[metric])
            summary[f"{metric}_median"] = f"{statistics.median(ordered):.8f}" if ordered else ""
            summary[f"{metric}_p10"] = f"{percentile(ordered, .10):.8f}" if ordered else ""
            summary[f"{metric}_p90"] = f"{percentile(ordered, .90):.8f}" if ordered else ""
        summary_rows.append(summary)
    return split_rows, fit_rows, summary_rows
