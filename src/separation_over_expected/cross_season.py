from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from .feature_schema import DYNAMIC_CONTEXT_FEATURES
from .models import (
    GlobalMeanModel,
    RidgeContextModel,
    regression_metrics,
    split_rows,
    target,
)
from .reports import read_csv_rows


class SharedFeatureRidgeModel(RidgeContextModel):
    """Ridge model limited to fields available in both tracking eras."""

    name = "shared_feature_ridge"
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
        "yardline_100",
    ]
    categorical_features: list[str] = []


class SharedDynamicFeatureRidgeModel(SharedFeatureRidgeModel):
    """Common static features plus the project's three-defender motion summaries."""

    name = "shared_dynamic_feature_ridge"
    impute_missing_numeric = True
    numeric_features = [*SharedFeatureRidgeModel.numeric_features, *DYNAMIC_CONTEXT_FEATURES]


def yardline_100_from_tracking(absolute_yardline: float, play_direction: str) -> float:
    """Convert the absolute field x coordinate to offense-relative yards to goal."""
    if play_direction == "right":
        return 110.0 - absolute_yardline
    if play_direction == "left":
        return absolute_yardline - 10.0
    raise ValueError(f"Unknown play direction: {play_direction!r}")


def load_cross_season_wr_rows(
    legacy_route_path: Path,
    bdb2026_route_path: Path,
    nflverse_pbp_path: Path,
    require_dynamic: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int | float]]:
    """Map 2021 and 2023 route tables onto the shared pre-throw feature schema.

    The 2023 play context is joined using BDB `gameId`/`playId` and nflverse
    `old_game_id`/`play_id`. The join uses CSV and the Python standard library.
    """

    old_rows = [
        dict(row)
        for row in read_csv_rows(legacy_route_path)
        if row.get("officialPosition") == "WR"
    ]
    if require_dynamic:
        missing = [
            feature for feature in DYNAMIC_CONTEXT_FEATURES
            if not old_rows or feature not in old_rows[0]
        ]
        if missing:
            raise ValueError(
                "2021 route table lacks dynamic features; rebuild with "
                "build-route-table --include-dynamic-features."
            )
    old_common = []
    for row in old_rows:
        absolute = row.get("absoluteYardlineNumber", "")
        if absolute in {"", "NA"}:
            continue
        row["yardline_100"] = f"{yardline_100_from_tracking(float(absolute), row['playDirection']):.3f}"
        old_common.append(row)

    pbp_context: dict[tuple[str, str], dict[str, str]] = {}
    with nflverse_pbp_path.open(newline="") as source:
        for row in csv.DictReader(source):
            if not row.get("old_game_id") or not row.get("play_id"):
                continue
            key = (row["old_game_id"], str(int(float(row["play_id"]))))
            pbp_context[key] = {
                field: row[field]
                for field in ("down", "ydstogo", "yardline_100")
            }

    new_common = []
    missing_context = 0
    yardline_errors = []
    with bdb2026_route_path.open(newline="") as source:
        route_rows = csv.DictReader(source)
        if require_dynamic:
            missing = [
                feature for feature in DYNAMIC_CONTEXT_FEATURES
                if feature not in (route_rows.fieldnames or [])
            ]
            if missing:
                raise ValueError(
                    "2023 route table lacks dynamic features; build it with "
                    "build-bdb2026-route-table --include-dynamic-features."
                )
        for row in route_rows:
            if row.get("officialPosition") != "WR":
                continue
            key = (row["gameId"], row["playId"])
            context = pbp_context.get(key)
            if context is None:
                missing_context += 1
                continue
            derived = yardline_100_from_tracking(
                float(row["absoluteYardlineNumber"]), row["playDirection"]
            )
            yardline_errors.append(abs(derived - float(context["yardline_100"])))
            new_common.append(
                {
                    "gameId": row["gameId"],
                    "playId": row["playId"],
                    "week": row["week"],
                    "nflId": row["nflId"],
                    "displayName": row.get("displayName", ""),
                    "officialPosition": "WR",
                    "delta_sep": row["delta_sep_input_window"],
                    "sep_snap": row["sep_first_input"],
                    "snap_x_norm": row["first_x_norm"],
                    "snap_y_norm": row["first_y_norm"],
                    "nearest_defender_dx_snap": row["nearest_defender_dx_first"],
                    "nearest_defender_dy_snap": row["nearest_defender_dy_first"],
                    "receiver_speed_snap": row["receiver_speed_first"],
                    "receiver_accel_snap": row["receiver_accel_first"],
                    "time_to_throw_frames": row["input_window_frames"],
                    "down": context["down"],
                    "yardsToGo": context["ydstogo"],
                    "yardline_100": f"{float(context['yardline_100']):.3f}",
                }
            )
            if require_dynamic:
                new_common[-1].update(
                    {feature: row[feature] for feature in DYNAMIC_CONTEXT_FEATURES}
                )
    if missing_context:
        raise ValueError(f"{missing_context} BDB WR route rows lack matched play context")
    yardline_error = max(yardline_errors, default=0.0)
    if yardline_error > 1e-9:
        raise ValueError(
            f"Tracking/pbp yardline conventions differ (max absolute error {yardline_error})"
        )
    diagnostics = {
        "legacy_wr_routes": len(old_common),
        "bdb2023_wr_routes": len(new_common),
        "bdb2023_plays_without_context": missing_context,
        "yardline_max_absolute_error": yardline_error,
    }
    return old_common, new_common, diagnostics


def _score(model, rows: list[dict[str, str]], evaluation: str) -> dict[str, str]:
    actual = [target(row) for row in rows]
    predicted = [model.predict(row) for row in rows]
    return {
        "evaluation": evaluation,
        "model": model.name,
        "rows": str(len(rows)),
        **regression_metrics(actual, predicted),
    }


def evaluate_cross_season_transfer(
    legacy_rows: list[dict[str, str]],
    bdb2023_rows: list[dict[str, str]],
    seed: int = 42,
    include_dynamic: bool = False,
) -> list[dict[str, str]]:
    """Compare common-feature static/dynamic models across seasons and game holdouts."""
    old_splits = split_rows(legacy_rows, strategy="game", seed=seed)
    new_splits = split_rows(bdb2023_rows, strategy="game", seed=seed)
    results = []
    model_types = [SharedFeatureRidgeModel]
    if include_dynamic:
        model_types.append(SharedDynamicFeatureRidgeModel)
    evaluations = (
        ("2021 games held out", old_splits["train"], old_splits["test"]),
        ("2021 train → all 2023 WR routes", legacy_rows, bdb2023_rows),
        ("2023 game holdout", new_splits["train"], new_splits["test"]),
    )
    for label, training_rows, scoring_rows in evaluations:
        mean_model = GlobalMeanModel()
        mean_model.fit(training_rows)
        results.append(_score(mean_model, scoring_rows, label))
        for model_type in model_types:
            model = model_type(l2=25.0, max_levels_per_feature=0)
            model.fit(training_rows)
            results.append(_score(model, scoring_rows, label))
    return results


def paired_game_bootstrap_rmse_delta(
    actual: list[float],
    static_predictions: list[float],
    dynamic_predictions: list[float],
    game_ids: list[str],
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired dynamic-minus-static RMSE while resampling whole games."""
    if not (len(actual) == len(static_predictions) == len(dynamic_predictions) == len(game_ids)):
        raise ValueError("Actuals, predictions, and game IDs must have equal lengths")
    if not actual or iterations < 1:
        raise ValueError("At least one route and one bootstrap iteration are required")

    by_game: dict[str, list[int]] = {}
    for index, game_id in enumerate(game_ids):
        by_game.setdefault(game_id, []).append(index)
    games = list(by_game)
    if len(games) < 2:
        raise ValueError("At least two games are required for a game-cluster bootstrap")

    def rmse_delta(indices: list[int]) -> float:
        static_sse = sum((actual[i] - static_predictions[i]) ** 2 for i in indices)
        dynamic_sse = sum((actual[i] - dynamic_predictions[i]) ** 2 for i in indices)
        return math.sqrt(dynamic_sse / len(indices)) - math.sqrt(static_sse / len(indices))

    point = rmse_delta(list(range(len(actual))))
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        sampled_games = [rng.choice(games) for _ in games]
        sample_indices = [
            index for game_id in sampled_games for index in by_game[game_id]
        ]
        deltas.append(rmse_delta(sample_indices))
    deltas.sort()
    return {
        "point_delta": point,
        "lower_95": deltas[int(0.025 * (iterations - 1))],
        "upper_95": deltas[int(0.975 * (iterations - 1))],
        "bootstrap_share_dynamic_better": sum(value < 0 for value in deltas) / iterations,
    }
