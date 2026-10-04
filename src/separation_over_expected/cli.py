from __future__ import annotations

import argparse
from pathlib import Path

from .features import build_route_table
from .models import evaluate_models, filter_rows, fit_models, split_rows
from .reports import (
    read_csv_rows,
    write_metrics,
    write_predictions,
    write_receiver_summary,
    write_split_half_stability,
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
    baseline.add_argument(
        "--position",
        choices=["WR", "TE", "RB", "FB"],
        help="Optional official position filter for model fitting and scoring.",
    )
    baseline.add_argument(
        "--min-routes",
        type=int,
        default=25,
        help="Minimum player routes required in the receiver summary.",
    )

    position_baselines = subparsers.add_parser(
        "fit-position-baselines",
        help="Fit pooled and position-specific baseline models for WR, TE, and RB.",
    )
    position_baselines.add_argument(
        "--route-table",
        type=Path,
        default=Path("data/processed/route_level_snap_to_release.csv"),
        help="Route-level CSV created by build-route-table.",
    )
    position_baselines.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/position_baselines"),
        help="Directory for position-specific predictions, summaries, and metrics.",
    )
    position_baselines.add_argument(
        "--positions",
        nargs="*",
        choices=["WR", "TE", "RB", "FB"],
        default=["WR", "TE", "RB"],
        help="Official positions to model separately.",
    )
    position_baselines.add_argument(
        "--min-routes",
        type=int,
        default=25,
        help="Minimum player routes required in each receiver summary.",
    )

    stability = subparsers.add_parser(
        "split-half-stability",
        help="Create a player-level early/late SOE stability table.",
    )
    stability.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/position_baselines/route_level_baseline_predictions_wr.csv"),
        help="Route-level predictions CSV containing soe_route.",
    )
    stability.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stability/wr_split_half_stability.csv"),
        help="Output CSV path for player-level split-half stability.",
    )
    stability.add_argument(
        "--position",
        choices=["WR", "TE", "RB", "FB"],
        default="WR",
        help="Official position to include.",
    )
    stability.add_argument(
        "--min-routes-per-half",
        type=int,
        default=20,
        help="Minimum routes required in weeks 1-4 and weeks 5-8.",
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
            args.position,
            args.min_routes,
        )
    elif args.command == "fit-position-baselines":
        fit_position_baselines(
            args.route_table,
            args.output_dir,
            args.positions,
            args.min_routes,
        )
    elif args.command == "split-half-stability":
        rows = read_csv_rows(args.predictions)
        write_split_half_stability(
            args.output,
            rows,
            position=args.position,
            min_routes_per_half=args.min_routes_per_half,
        )
        print(f"stability table: {args.output}")


def fit_baselines(
    route_table: Path,
    predictions_path: Path,
    receiver_summary_path: Path,
    metrics_path: Path,
    position: str | None = None,
    min_routes: int = 25,
) -> None:
    rows = read_csv_rows(route_table)
    rows = filter_rows(rows, position=position)
    if not rows:
        raise ValueError(f"No route rows available for position={position!r}")
    splits = split_rows(rows)
    models = fit_models(splits["train"])
    metrics_rows = evaluate_models(models, splits)
    scoring_model = next(model for model in models if model.name == "ridge_context")

    write_metrics(metrics_path, metrics_rows)
    write_predictions(predictions_path, rows, models, scoring_model)
    write_receiver_summary(
        receiver_summary_path,
        rows,
        scoring_model,
        min_routes=min_routes,
        position=position,
    )

    label = position or "ALL"
    print(f"baseline metrics ({label})")
    for row in metrics_rows:
        if row["split"] in {"validation", "test"}:
            print(
                f"{row['model']:>20} {row['split']:>10} "
                f"rmse={row['rmse']} mae={row['mae']} r2={row['r2']}"
            )
    print(f"predictions: {predictions_path}")
    print(f"receiver summary: {receiver_summary_path}")
    print(f"metrics: {metrics_path}")


def fit_position_baselines(
    route_table: Path,
    output_dir: Path,
    positions: list[str],
    min_routes: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_baselines(
        route_table,
        output_dir / "route_level_baseline_predictions_all.csv",
        output_dir / "receiver_baseline_summary_all.csv",
        output_dir / "baseline_metrics_all.csv",
        position=None,
        min_routes=min_routes,
    )
    for position in positions:
        suffix = position.lower()
        fit_baselines(
            route_table,
            output_dir / f"route_level_baseline_predictions_{suffix}.csv",
            output_dir / f"receiver_baseline_summary_{suffix}.csv",
            output_dir / f"baseline_metrics_{suffix}.csv",
            position=position,
            min_routes=min_routes,
        )
