from __future__ import annotations

import argparse
from pathlib import Path

from .features import build_route_table
from .models import evaluate_models, fit_models, split_rows
from .reports import (
    read_csv_rows,
    write_metrics,
    write_predictions,
    write_receiver_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and model NFL Big Data Bowl separation-over-expected data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-route-table",
        help="Create one row per pass route with snap-to-release separation features.",
    )
    build.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/big_data_bowl_2023"),
        help="Directory containing games.csv, plays.csv, players.csv, pffScoutingData.csv, and week*.csv.",
    )
    build.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/route_level_snap_to_release.csv"),
        help="Output CSV path.",
    )
    build.add_argument(
        "--weeks",
        nargs="*",
        type=int,
        default=list(range(1, 9)),
        help="Week numbers to process.",
    )

    baseline = subparsers.add_parser(
        "fit-baselines",
        help="Fit transparent baseline models for route-level separation over expected.",
    )
    baseline.add_argument(
        "--route-table",
        type=Path,
        default=Path("data/processed/route_level_snap_to_release.csv"),
        help="Route-level CSV created by build-route-table.",
    )
    baseline.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/route_level_baseline_predictions.csv"),
        help="Output CSV with route-level predictions and residuals.",
    )
    baseline.add_argument(
        "--receiver-summary",
        type=Path,
        default=Path("data/processed/receiver_baseline_summary.csv"),
        help="Output CSV with receiver-level SOE aggregates.",
    )
    baseline.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/processed/baseline_metrics.csv"),
        help="Output CSV with split-level model metrics.",
    )

    args = parser.parse_args()
    if args.command == "build-route-table":
        build_route_table(args.data_dir, args.output, args.weeks)
    elif args.command == "fit-baselines":
        fit_baselines(
            args.route_table,
            args.predictions,
            args.receiver_summary,
            args.metrics,
        )


def fit_baselines(
    route_table: Path,
    predictions_path: Path,
    receiver_summary_path: Path,
    metrics_path: Path,
) -> None:
    rows = read_csv_rows(route_table)
    splits = split_rows(rows)
    models = fit_models(splits["train"])
    metrics_rows = evaluate_models(models, splits)
    scoring_model = next(model for model in models if model.name == "ridge_context")

    write_metrics(metrics_path, metrics_rows)
    write_predictions(predictions_path, rows, models, scoring_model)
    write_receiver_summary(receiver_summary_path, rows, scoring_model)

    print("baseline metrics")
    for row in metrics_rows:
        if row["split"] in {"validation", "test"}:
            print(
                f"{row['model']:>20} {row['split']:>10} "
                f"rmse={row['rmse']} mae={row['mae']} r2={row['r2']}"
            )
    print(f"predictions: {predictions_path}")
    print(f"receiver summary: {receiver_summary_path}")
    print(f"metrics: {metrics_path}")
