from __future__ import annotations

import csv
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
    fieldnames = list(rows[0].keys())
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
                "total_soe": f"{sum(values):.3f}",
            }
        )
    summaries.sort(key=lambda row: float(row["mean_soe"]), reverse=True)
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
        "total_soe",
    ]
    write_csv(
        path,
        fieldnames,
        receiver_summary_rows(rows, scoring_model, min_routes, position),
    )


def dataset_overview(rows: list[dict[str, str]]) -> dict[str, object]:
    splits = split_rows(rows)
    return {
        "routes": len(rows),
        "plays": len({(row["gameId"], row["playId"]) for row in rows}),
        "players": len({row["nflId"] for row in rows}),
        "weeks": sorted({int(row["week"]) for row in rows}),
        "split_rows": {name: len(split) for name, split in splits.items()},
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
