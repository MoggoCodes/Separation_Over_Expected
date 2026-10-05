from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .models import add_model_predictions, split_rows, target


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "model",
        "split",
        "rows",
        "mae",
        "rmse",
        "r2",
        "bias",
        "mean_actual",
        "mean_predicted",
        "n",
    ]
    write_csv(path, fieldnames, rows)


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    models: list[object],
    scoring_model: object,
) -> None:
    fieldnames = [field for field in rows[0].keys() if field != "split"]
    for model in models:
        fieldnames.append(f"pred_delta_sep_{model.name}")
    fieldnames.extend(["soe_route", "split"])
    write_csv(
        path,
        fieldnames,
        [add_model_predictions(row, models, scoring_model) for row in rows],
    )


def receiver_summary_rows(
    rows: list[dict[str, str]],
    scoring_model: object,
    min_routes: int = 25,
    position: str | None = None,
) -> list[dict[str, str]]:
    by_player: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if position is not None and row["officialPosition"] != position:
            continue
        key = (row["nflId"], row["displayName"], row["officialPosition"])
        by_player[key].append(target(row) - scoring_model.predict(row))

    summaries = []
    for (nfl_id, name, official_position), values in by_player.items():
        if len(values) < min_routes:
            continue
        summaries.append(
            {
                "nflId": nfl_id,
                "displayName": name,
                "officialPosition": official_position,
                "routes": str(len(values)),
                "mean_soe": f"{statistics.fmean(values):.3f}",
                "median_soe": f"{statistics.median(values):.3f}",
                "std_soe": f"{sample_stdev(values):.3f}",
                "se_soe": f"{standard_error(values):.3f}",
                "lower_95_soe": f"{confidence_interval(values)[0]:.3f}",
                "upper_95_soe": f"{confidence_interval(values)[1]:.3f}",
                "total_soe": f"{sum(values):.3f}",
            }
        )
    summaries.sort(key=lambda row: float(row["lower_95_soe"]), reverse=True)
    return summaries


def write_receiver_summary(
    path: Path,
    rows: list[dict[str, str]],
    scoring_model: object,
    min_routes: int = 25,
    position: str | None = None,
) -> None:
    fieldnames = [
        "nflId",
        "displayName",
        "officialPosition",
        "routes",
        "mean_soe",
        "median_soe",
        "std_soe",
        "se_soe",
        "lower_95_soe",
        "upper_95_soe",
        "total_soe",
    ]
    write_csv(
        path,
        fieldnames,
        receiver_summary_rows(rows, scoring_model, min_routes, position),
    )


def split_half_stability_rows(
    rows: list[dict[str, str]],
    position: str | None = None,
    early_weeks: set[int] | None = None,
    late_weeks: set[int] | None = None,
    min_routes_per_half: int = 20,
) -> list[dict[str, str]]:
    early_weeks = early_weeks or {1, 2, 3, 4}
    late_weeks = late_weeks or {5, 6, 7, 8}
    by_player: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"early": [], "late": []}
    )

    for row in rows:
        if position is not None and row["officialPosition"] != position:
            continue
        week = int(row["week"])
        split = "early" if week in early_weeks else "late" if week in late_weeks else None
        if split is None:
            continue
        key = (row["nflId"], row["displayName"], row["officialPosition"])
        by_player[key][split].append(float(row["soe_route"]))

    summaries = []
    for (nfl_id, name, official_position), values in by_player.items():
        early = values["early"]
        late = values["late"]
        if len(early) < min_routes_per_half or len(late) < min_routes_per_half:
            continue
        early_mean = statistics.fmean(early)
        late_mean = statistics.fmean(late)
        summaries.append(
            {
                "nflId": nfl_id,
                "displayName": name,
                "officialPosition": official_position,
                "early_routes": str(len(early)),
                "late_routes": str(len(late)),
                "early_mean_soe": f"{early_mean:.3f}",
                "late_mean_soe": f"{late_mean:.3f}",
                "late_minus_early": f"{late_mean - early_mean:.3f}",
            }
        )
    summaries.sort(key=lambda row: float(row["early_mean_soe"]), reverse=True)
    return summaries


def write_split_half_stability(
    path: Path,
    rows: list[dict[str, str]],
    position: str | None = None,
    min_routes_per_half: int = 20,
) -> None:
    fieldnames = [
        "nflId",
        "displayName",
        "officialPosition",
        "early_routes",
        "late_routes",
        "early_mean_soe",
        "late_mean_soe",
        "late_minus_early",
    ]
    write_csv(
        path,
        fieldnames,
        split_half_stability_rows(
            rows,
            position=position,
            min_routes_per_half=min_routes_per_half,
        ),
    )


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom == 0.0 or y_denom == 0.0:
        return 0.0
    return numerator / (x_denom * y_denom)


def dataset_overview(rows: list[dict[str, str]]) -> dict[str, object]:
    splits = split_rows(rows)
    return {
        "routes": len(rows),
        "plays": len({(row["gameId"], row["playId"]) for row in rows}),
        "players": len({row["nflId"] for row in rows}),
        "weeks": sorted({int(row["week"]) for row in rows}),
        "split_rows": {name: len(split) for name, split in splits.items()},
    }


def sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def standard_error(values: list[float]) -> float:
    if not values:
        return 0.0
    return sample_stdev(values) / math.sqrt(len(values))


def confidence_interval(values: list[float], z: float = 1.96) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mean = statistics.fmean(values)
    margin = z * standard_error(values)
    return mean - margin, mean + margin


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
