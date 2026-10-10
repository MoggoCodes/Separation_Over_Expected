from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path

from .algorithm_comparison import MODEL_NAMES, compare_algorithms
from .reports import write_csv
from .validation import assign_game_halves, correlation_stats, percentile


TARGETS = {
    "endpoint_nearest": "delta_sep",
    "snap_anchor": "delta_sep_snap_anchor",
    "snap_top3": "delta_sep_snap_top3",
}
DEPTH_BINS = ((float("-inf"), 2.0, "<2 yd"), (2.0, 5.0, "2–5 yd"), (5.0, 10.0, "5–10 yd"), (10.0, float("inf"), "10+ yd"))


def _write(output_dir: Path, filename: str, rows: list[dict[str, str]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output_dir / filename, fields, rows)


def _target_distribution(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for target_name, column in TARGETS.items():
        values = sorted(float(row[column]) for row in rows)
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values)
        output.append({
            "target": target_name,
            "routes": str(len(values)),
            "mean": f"{mean:.6f}",
            "sd": f"{sd:.6f}",
            "median": f"{percentile(values, .5):.6f}",
            "p05": f"{percentile(values, .05):.6f}",
            "p95": f"{percentile(values, .95):.6f}",
            "positive_fraction": f"{sum(value > 0 for value in values)/len(values):.6f}",
        })
    return output


def _reliability(
    predictions_by_target: dict[str, list[dict[str, str]]],
    split_seeds: int,
    seed: int,
    thresholds: tuple[int, ...],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    observations = []
    for target_name, predictions in predictions_by_target.items():
        seeds = [seed + 10_000 + i * 37 for i in range(split_seeds)]
        for split_seed in seeds:
            halves = assign_game_halves(predictions, seed=split_seed)
            for model in MODEL_NAMES:
                by_player_half: dict[tuple[str, str], list[float]] = defaultdict(list)
                names = {}
                for row in predictions:
                    half = halves[(str(int(row["week"])), row["gameId"])]
                    residual = float(row["delta_sep"]) - float(row[f"pred_delta_sep_{model}"])
                    by_player_half[(row["nflId"], half)].append(residual)
                    names[row["nflId"]] = row["displayName"]
                player_ids = sorted({player for player, _ in by_player_half})
                for threshold in thresholds:
                    eligible = [
                        player for player in player_ids
                        if all(len(by_player_half[(player, half)]) >= threshold for half in ("A", "B"))
                    ]
                    a = [statistics.fmean(by_player_half[(player, "A")]) for player in eligible]
                    b = [statistics.fmean(by_player_half[(player, "B")]) for player in eligible]
                    correlations = correlation_stats(a, b)
                    observations.append({
                        "target": target_name, "model": model, "seed": str(split_seed),
                        "min_routes_per_half": str(threshold), "receivers": str(len(eligible)),
                        "pearson": f"{correlations['pearson']:.8f}",
                        "spearman": f"{correlations['spearman']:.8f}",
                    })
    summaries = []
    for target_name in TARGETS:
        for model in MODEL_NAMES:
            for threshold in thresholds:
                selected = [
                    row for row in observations
                    if row["target"] == target_name and row["model"] == model
                    and int(row["min_routes_per_half"]) == threshold
                ]
                for metric in ("pearson", "spearman", "receivers"):
                    values = sorted(float(row[metric]) for row in selected)
                    summaries.append({
                        "target": target_name, "model": model,
                        "min_routes_per_half": str(threshold), "metric": metric,
                        "median": f"{statistics.median(values):.8f}",
                        "p10": f"{percentile(values, .10):.8f}",
                        "p90": f"{percentile(values, .90):.8f}",
                        "split_seeds": str(len(values)),
                    })
    return observations, summaries


def _diagnostics(
    route_rows: list[dict[str, str]],
    predictions_by_target: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    route_by_key = {(row["gameId"], row["playId"], row["nflId"]): row for row in route_rows}
    output = []
    for target_name, predictions in predictions_by_target.items():
        for model in MODEL_NAMES:
            groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
            for prediction in predictions:
                key = (prediction["gameId"], prediction["playId"], prediction["nflId"])
                route = route_by_key[key]
                depth = float(route["route_depth"])
                depth_label = next(label for lo, hi, label in DEPTH_BINS if lo <= depth < hi)
                residual = float(prediction["delta_sep"]) - float(prediction[f"pred_delta_sep_{model}"])
                groups[("route_depth", depth_label)].append((float(prediction["delta_sep"]), residual))
                groups[("defender_switch", "switched" if route["nearest_defender_switched"] == "1" else "stable")].append((float(prediction["delta_sep"]), residual))
                groups[("alignment", route["pff_positionLinedUp"] or "Unknown")].append((float(prediction["delta_sep"]), residual))
            for (group_type, group), values in groups.items():
                actual = [value for value, _ in values]
                residuals = [residual for _, residual in values]
                output.append({
                    "target": target_name, "model": model,
                    "group_type": group_type, "group": group,
                    "routes": str(len(values)),
                    "mean_target": f"{statistics.fmean(actual):.6f}",
                    "mean_residual": f"{statistics.fmean(residuals):.6f}",
                    "rmse": f"{math.sqrt(statistics.fmean(value*value for value in residuals)):.6f}",
                })
    return output


def compare_separation_targets(
    rows: list[dict[str, str]],
    output_dir: Path,
    n_folds: int = 5,
    split_seeds: int = 100,
    min_routes_per_half: tuple[int, ...] = (10, 20, 30, 40, 50),
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Compare separation target definitions with matched route rows and grouped folds."""
    wr_rows = [row for row in rows if row.get("officialPosition") == "WR"]
    required = set(TARGETS.values()) | {"snap_top3_release_count", "nearest_defender_switched", "route_depth", "pff_positionLinedUp"}
    missing = sorted(required - set(wr_rows[0] if wr_rows else {}))
    if not wr_rows:
        raise ValueError("No WR route rows available")
    if missing:
        raise ValueError(f"Route table is missing target audit fields: {missing}")
    complete = [
        row for row in wr_rows
        if row["delta_sep_snap_anchor"] not in {"", "NA"}
        and row["delta_sep_snap_top3"] not in {"", "NA"}
        and int(row["snap_top3_release_count"]) == 3
    ]
    if not complete:
        raise ValueError("No routes have all three separation target definitions")

    predictions_by_target: dict[str, list[dict[str, str]]] = {}
    metrics, folds, calibration = [], [], []
    for target_name, target_column in TARGETS.items():
        target_rows = []
        for row in complete:
            clone = dict(row)
            clone["delta_sep"] = row[target_column]
            target_rows.append(clone)
        predictions, target_metrics, target_folds, target_calibration, _, _ = compare_algorithms(
            target_rows,
            n_folds=n_folds,
            min_routes_per_half=20,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        predictions_by_target[target_name] = predictions
        metrics.extend({"target": target_name, **row} for row in target_metrics)
        folds.extend({"target": target_name, **row} for row in target_folds)
        calibration.extend({"target": target_name, **row} for row in target_calibration)

    reliability_seeds, reliability_summary = _reliability(
        predictions_by_target, split_seeds, seed, min_routes_per_half
    )
    diagnostics = _diagnostics(complete, predictions_by_target)
    distributions = _target_distribution(complete)
    sd_by_target = {row["target"]: float(row["sd"]) for row in distributions}
    for row in metrics:
        row["target_sd"] = f"{sd_by_target[row['target']]:.6f}"
        row["normalized_rmse"] = f"{float(row['rmse'])/sd_by_target[row['target']]:.6f}"
    switch_rates = [{
        "routes": str(len(complete)),
        "defender_switches": str(sum(row["nearest_defender_switched"] == "1" for row in complete)),
        "switch_rate": f"{sum(row['nearest_defender_switched'] == '1' for row in complete)/len(complete):.8f}",
        "excluded_missing_target_routes": str(len(wr_rows)-len(complete)),
    }]
    pair_correlations = []
    for i, first in enumerate(TARGETS):
        for second in list(TARGETS)[i+1:]:
            stats = correlation_stats(
                [float(row[TARGETS[first]]) for row in complete],
                [float(row[TARGETS[second]]) for row in complete],
            )
            pair_correlations.append({"target_a": first, "target_b": second, **{key: f"{value:.8f}" for key, value in stats.items()}})

    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "target_distributions": distributions,
        "switch_rates": switch_rates,
        "target_correlations": pair_correlations,
        "model_metrics": metrics,
        "fold_metrics": folds,
        "calibration_deciles": calibration,
        "reliability_seeds": reliability_seeds,
        "reliability_summary": reliability_summary,
        "route_diagnostics": diagnostics,
        "oof_predictions": [
            {"target": target_name, **row}
            for target_name, predictions in predictions_by_target.items()
            for row in predictions
        ],
    }
    for name, table in output.items():
        _write(output_dir, f"{name}.csv", table)
    return output
