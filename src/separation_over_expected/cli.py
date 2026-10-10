from __future__ import annotations

import argparse
from pathlib import Path

from .features import build_route_table
from .models import evaluate_models, filter_rows, fit_models, split_rows
from .reports import (
    read_csv_rows,
    write_csv,
    write_metrics,
    write_predictions,
    write_receiver_summary,
    write_split_half_stability,
)
from .validation import cross_validate_position, write_oof_outputs
from .bdb2026 import build_bdb2026_dynamic_route_table, build_bdb2026_route_table
from .cross_season import evaluate_cross_season_transfer, load_cross_season_wr_rows
from .receiver_reliability import (
    cross_season_receiver_predictions,
    summarize_cross_season_receivers,
)
from .ridge_comparison import DEFAULT_L2_VALUES, compare_ridge_specifications
from .algorithm_comparison import compare_algorithms
from .player_audit import run_player_validity_audit


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
    build.add_argument(
        "--include-dynamic-features",
        action="store_true",
        help="Add pre-release summaries of the three coverage defenders nearest at the snap.",
    )
    build.add_argument(
        "--include-pocket-features",
        action="store_true",
        help="Also add quarterback movement and PFF pass-rush pressure summaries before release.",
    )

    build_2026 = subparsers.add_parser(
        "build-bdb2026-route-table",
        help="Build a route-level table from the 2023 BDB 2026 pre-throw inputs.",
    )
    build_2026.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data/big_data_bowl_2026"),
        help="Directory containing train/input_2023_wXX.csv.",
    )
    build_2026.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/bdb2026/route_level_input_window_2023.csv"),
    )
    build_2026.add_argument(
        "--weeks",
        nargs="*",
        type=int,
        help="Optional weeks to process; defaults to all available training weeks.",
    )
    build_2026.add_argument(
        "--include-dynamic-features",
        action="store_true",
        help="Also summarize motion of the three nearest tagged coverage defenders.",
    )

    cross_season = subparsers.add_parser(
        "compare-cross-season",
        help="Evaluate a shared-feature WR ridge across the 2021 and 2023 tracking datasets.",
    )
    cross_season.add_argument(
        "--legacy-route-table",
        type=Path,
        default=Path("data/processed/route_level_snap_to_release.csv"),
    )
    cross_season.add_argument(
        "--bdb2026-route-table",
        type=Path,
        default=Path("data/processed/bdb2026/route_level_input_window_2023.csv"),
    )
    cross_season.add_argument(
        "--nflverse-pbp",
        type=Path,
        default=Path("../data/big_data_bowl_2026/nflverse_play_by_play_2023.csv"),
    )
    cross_season.add_argument("--seed", type=int, default=42)

    cross_season_dynamic = subparsers.add_parser(
        "compare-cross-season-dynamic",
        help="Compare common-feature static and dynamic WR ridge models across 2021 and 2023.",
    )
    cross_season_dynamic.add_argument(
        "--legacy-route-table",
        type=Path,
        default=Path("data/processed/dynamic_features/route_level_snap_to_release_dynamic.csv"),
    )
    cross_season_dynamic.add_argument(
        "--bdb2026-route-table",
        type=Path,
        default=Path("data/processed/bdb2026/route_level_input_window_2023_dynamic.csv"),
    )
    cross_season_dynamic.add_argument(
        "--nflverse-pbp",
        type=Path,
        default=Path("../data/big_data_bowl_2026/nflverse_play_by_play_2023.csv"),
    )
    cross_season_dynamic.add_argument("--seed", type=int, default=42)

    receiver_reliability = subparsers.add_parser(
        "cross-season-receiver-reliability",
        help="Compare receiver residual performance from 2021 to 2023.",
    )
    receiver_reliability.add_argument(
        "--legacy-route-table",
        type=Path,
        default=Path("data/processed/dynamic_features/route_level_snap_to_release_dynamic.csv"),
    )
    receiver_reliability.add_argument(
        "--bdb2026-route-table",
        type=Path,
        default=Path("data/processed/bdb2026/route_level_input_window_2023_dynamic.csv"),
    )
    receiver_reliability.add_argument(
        "--nflverse-pbp",
        type=Path,
        default=Path("../data/big_data_bowl_2026/nflverse_play_by_play_2023.csv"),
    )
    receiver_reliability.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cross_season_receiver_reliability"),
    )
    receiver_reliability.add_argument("--folds", type=int, default=5)
    receiver_reliability.add_argument("--min-routes", type=int, default=20)
    receiver_reliability.add_argument("--min-games", type=int, default=5)
    receiver_reliability.add_argument("--bootstrap-samples", type=int, default=2000)
    receiver_reliability.add_argument("--seed", type=int, default=42)

    ridge_comparison = subparsers.add_parser(
        "compare-ridge-specifications",
        help="Compare ridge feature blocks and l2 penalties on grouped OOF routes.",
    )
    ridge_comparison.add_argument(
        "--route-table",
        type=Path,
        default=Path("data/processed/dynamic_features/pocket_context/route_level_snap_to_release_pocket.csv"),
        help="Dynamic route table containing both pocket and pressure features.",
    )
    ridge_comparison.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/ridge_comparison"),
    )
    ridge_comparison.add_argument(
        "--l2-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_L2_VALUES),
        help="Ridge penalties to compare; include 25 for the fixed reference candidate.",
    )
    ridge_comparison.add_argument("--folds", type=int, default=5)
    ridge_comparison.add_argument("--min-routes-per-half", type=int, default=20)
    ridge_comparison.add_argument("--bootstrap-samples", type=int, default=2000)
    ridge_comparison.add_argument("--seed", type=int, default=42)

    algorithm_comparison = subparsers.add_parser(
        "compare-algorithms",
        help="Compare dynamic ridge with Extra Trees and histogram gradient boosting.",
    )
    algorithm_comparison.add_argument(
        "--route-table", type=Path,
        default=Path("data/processed/dynamic_features/pocket_context/route_level_snap_to_release_pocket.csv"),
    )
    algorithm_comparison.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/algorithm_comparison"),
    )
    algorithm_comparison.add_argument("--folds", type=int, default=5)
    algorithm_comparison.add_argument("--min-routes-per-half", type=int, default=20)
    algorithm_comparison.add_argument("--bootstrap-samples", type=int, default=1000)
    algorithm_comparison.add_argument("--seed", type=int, default=42)

    player_audit = subparsers.add_parser(
        "audit-receiver-validity",
        help="Audit receiver split-half reliability, usage sensitivity, and route-depth residuals.",
    )
    player_audit.add_argument("--route-table", type=Path, default=Path("data/processed/dynamic_features/pocket_context/route_level_snap_to_release_pocket.csv"))
    player_audit.add_argument("--comparison-dir", type=Path, default=Path("data/processed/algorithm_comparison"))
    player_audit.add_argument("--output-dir", type=Path, default=Path("data/processed/player_validity_audit"))
    player_audit.add_argument("--folds", type=int, default=5)
    player_audit.add_argument("--split-seeds", type=int, default=100)
    player_audit.add_argument("--seed", type=int, default=42)

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
    baseline.add_argument(
        "--dynamic-receiver-summary",
        type=Path,
        default=Path("data/processed/receiver_dynamic_context_summary.csv"),
        help="Output path for dynamic-model receiver summaries when enabled.",
    )
    baseline.add_argument("--include-dynamic-features", action="store_true")
    add_split_arguments(baseline)

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
    position_baselines.add_argument("--include-dynamic-features", action="store_true")
    add_split_arguments(position_baselines)

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

    cross_validation = subparsers.add_parser(
        "cross-validate-position",
        help="Create game-grouped out-of-fold predictions and receiver stability diagnostics.",
    )
    cross_validation.add_argument(
        "--route-table",
        type=Path,
        default=Path("data/processed/dynamic_features/route_level_snap_to_release_dynamic.csv"),
        help="Dynamic route-level CSV created by build-route-table.",
    )
    cross_validation.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/dynamic_features/cross_validation"),
        help="Output directory for out-of-fold predictions and reliability reports.",
    )
    cross_validation.add_argument(
        "--position", choices=["WR", "TE", "RB", "FB"], default="WR"
    )
    cross_validation.add_argument("--folds", type=int, default=5)
    cross_validation.add_argument("--seed", type=int, default=42)
    cross_validation.add_argument("--min-routes-per-half", type=int, default=20)
    cross_validation.add_argument("--bootstrap-samples", type=int, default=2000)
    cross_validation.add_argument(
        "--include-pocket-features",
        action="store_true",
        help="Fit an additional ridge model with quarterback and pressure context.",
    )
    cross_validation.add_argument(
        "--include-pressure-context",
        action="store_true",
        help="Fit an additional ridge model using pass-rusher proximity and closing features only.",
    )

    args = parser.parse_args()
    if args.command == "build-route-table":
        build_route_table(
            args.data_dir,
            args.output,
            args.weeks,
            include_dynamic_features=args.include_dynamic_features,
            include_pocket_features=args.include_pocket_features,
        )
    elif args.command == "build-bdb2026-route-table":
        builder = (
            build_bdb2026_dynamic_route_table
            if args.include_dynamic_features
            else build_bdb2026_route_table
        )
        diagnostics = builder(args.data_dir, args.output, args.weeks)
        print(f"BDB 2026 route table: {args.output}")
        for name, value in diagnostics.items():
            print(f"{name}: {value:,}")
    elif args.command == "compare-cross-season":
        old_rows, new_rows, diagnostics = load_cross_season_wr_rows(
            args.legacy_route_table,
            args.bdb2026_route_table,
            args.nflverse_pbp,
        )
        print("cross-season data checks")
        for name, value in diagnostics.items():
            print(f"{name}: {value:,}")
        print("cross-season model results")
        for row in evaluate_cross_season_transfer(old_rows, new_rows, args.seed):
            print(
                f"{row['evaluation']}: {row['model']} n={row['n']} "
                f"RMSE={row['rmse']} R2={row['r2']} MAE={row['mae']} "
                f"bias={row['bias']}"
            )
    elif args.command == "compare-cross-season-dynamic":
        old_rows, new_rows, diagnostics = load_cross_season_wr_rows(
            args.legacy_route_table,
            args.bdb2026_route_table,
            args.nflverse_pbp,
            require_dynamic=True,
        )
        print("dynamic cross-season data checks")
        for name, value in diagnostics.items():
            print(f"{name}: {value:,}")
        print("dynamic cross-season model results")
        for row in evaluate_cross_season_transfer(
            old_rows, new_rows, args.seed, include_dynamic=True
        ):
            print(
                f"{row['evaluation']}: {row['model']} n={row['n']} "
                f"RMSE={row['rmse']} R2={row['r2']} MAE={row['mae']} "
                f"bias={row['bias']}"
            )
    elif args.command == "cross-season-receiver-reliability":
        old_rows, new_rows, diagnostics = load_cross_season_wr_rows(
            args.legacy_route_table,
            args.bdb2026_route_table,
            args.nflverse_pbp,
            require_dynamic=True,
        )
        predictions = cross_season_receiver_predictions(
            old_rows, new_rows, n_folds=args.folds, seed=args.seed
        )
        receiver_rows, reliability, receiver_diagnostics = summarize_cross_season_receivers(
            predictions,
            min_routes=args.min_routes,
            min_games=args.min_games,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        if not receiver_rows or not reliability:
            raise ValueError(
                "At least three shared receivers must meet the route and game thresholds"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "route_predictions.csv", list(predictions[0]), predictions)
        write_csv(args.output_dir / "receiver_summaries.csv", list(receiver_rows[0]), receiver_rows)
        write_csv(args.output_dir / "reliability_metrics.csv", list(reliability[0]), reliability)
        print("cross-season join checks")
        for name, value in diagnostics.items():
            print(f"{name}: {value}")
        print("receiver cohort")
        for name, value in receiver_diagnostics.items():
            print(f"{name}: {value}")
        print("cross-season receiver correlations")
        for row in reliability:
            print(row)
        print(f"outputs: {args.output_dir}")
    elif args.command == "compare-ridge-specifications":
        outputs = compare_ridge_specifications(
            read_csv_rows(args.route_table),
            l2_values=tuple(args.l2_values),
            n_folds=args.folds,
            min_routes_per_half=args.min_routes_per_half,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        predictions, metrics, folds, deciles, player_halves, reliability = outputs
        if not reliability:
            raise ValueError(
                "Fewer than three receivers met the split-half route threshold"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, table in (
            ("route_oof_predictions.csv", predictions),
            ("candidate_metrics.csv", metrics),
            ("fold_metrics.csv", folds),
            ("calibration_deciles.csv", deciles),
            ("receiver_half_scores.csv", player_halves),
            ("receiver_reliability.csv", reliability),
        ):
            columns = list(dict.fromkeys(key for row in table for key in row))
            write_csv(args.output_dir / filename, columns, table)
        print("grouped OOF ridge candidate scores (lower RMSE is better)")
        for row in sorted(metrics, key=lambda item: float(item["rmse"])):
            print(
                f"{row['candidate']}: RMSE={row['rmse']} MAE={row['mae']} "
                f"R2={row['r2']} slope={row['calibration_slope']} "
                f"decile_bias={row['mean_absolute_decile_bias']} "
                f"ΔRMSE={row['rmse_delta_vs_dynamic_l2_25']} "
                f"[{row['rmse_delta_lower_95']}, {row['rmse_delta_upper_95']}]"
            )
        print("receiver split-half reliability")
        for row in sorted(reliability, key=lambda item: float(item["pearson"]), reverse=True):
            print(
                f"{row['candidate']}: Pearson={row['pearson']} "
                f"[{row['pearson_ci_lower']}, {row['pearson_ci_upper']}], "
                f"Δ vs dynamic_l2_25={row['pearson_delta_vs_dynamic_l2_25']} "
                f"[{row['pearson_delta_lower_95']}, {row['pearson_delta_upper_95']}]"
            )
        print(f"outputs: {args.output_dir}")
    elif args.command == "compare-algorithms":
        outputs = compare_algorithms(
            read_csv_rows(args.route_table), n_folds=args.folds,
            min_routes_per_half=args.min_routes_per_half,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        )
        predictions, metrics, folds, deciles, player_halves, reliability = outputs
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, table in (
            ("route_oof_predictions.csv", predictions),
            ("candidate_metrics.csv", metrics),
            ("fold_metrics.csv", folds),
            ("calibration_deciles.csv", deciles),
            ("receiver_half_scores.csv", player_halves),
            ("receiver_reliability.csv", reliability),
        ):
            columns = list(dict.fromkeys(key for row in table for key in row))
            write_csv(args.output_dir / filename, columns, table)
        print("game-held-out algorithm comparison (lower RMSE is better)")
        for row in sorted(metrics, key=lambda item: float(item["rmse"])):
            print(
                f"{row['model']}: RMSE={row['rmse']} MAE={row['mae']} R2={row['r2']} "
                f"ΔRMSE vs ridge={row['rmse_delta_vs_ridge']} "
                f"[{row['rmse_delta_lower_95']}, {row['rmse_delta_upper_95']}], "
                f"calibration slope={row['calibration_slope']}"
            )
        print("receiver split-half reliability")
        for row in reliability:
            print(row)
        print(f"outputs: {args.output_dir}")
    elif args.command == "audit-receiver-validity":
        outputs = run_player_validity_audit(
            read_csv_rows(args.route_table),
            read_csv_rows(args.comparison_dir / "route_oof_predictions.csv"),
            args.output_dir,
            read_csv_rows(Path("data/processed/cross_season_receiver_reliability/receiver_summaries.csv")),
            folds=args.folds, split_seeds=args.split_seeds, seed=args.seed,
        )
        print("median split-half reliability across random game-half assignments")
        for row in outputs["reliability_summary"]:
            if row["metric"] in {"pearson", "spearman"} and row["min_routes_per_half"] == "20":
                print(f"{row['model']} {row['metric']}: {row['median']} (P10–P90 {row['p10']}–{row['p90']}; n={row['split_seeds']} splits)")
        print(f"outputs: {args.output_dir}")
    elif args.command == "fit-baselines":
        fit_baselines(
            args.route_table,
            args.predictions,
            args.receiver_summary,
            args.metrics,
            args.position,
            args.min_routes,
            args.split_strategy,
            args.seed,
            args.train_fraction,
            args.validation_fraction,
            include_dynamic_features=args.include_dynamic_features,
            dynamic_receiver_summary_path=args.dynamic_receiver_summary,
        )
    elif args.command == "fit-position-baselines":
        fit_position_baselines(
            args.route_table,
            args.output_dir,
            args.positions,
            args.min_routes,
            args.split_strategy,
            args.seed,
            args.train_fraction,
            args.validation_fraction,
            args.include_dynamic_features,
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
    elif args.command == "cross-validate-position":
        rows = read_csv_rows(args.route_table)
        predictions, metrics, player_rows, reliability = cross_validate_position(
            rows,
            position=args.position,
            n_folds=args.folds,
            seed=args.seed,
            min_routes_per_half=args.min_routes_per_half,
            bootstrap_samples=args.bootstrap_samples,
            include_pocket_context=args.include_pocket_features,
            include_pressure_context=args.include_pressure_context,
        )
        write_oof_outputs(
            args.output_dir,
            args.position,
            predictions,
            metrics,
            player_rows,
            reliability,
        )
        print(f"out-of-fold routes: {len(predictions):,}")
        print(f"metrics: {args.output_dir / f'oof_metrics_{args.position.lower()}.csv'}")
        print(f"receiver reliability: {args.output_dir / f'receiver_oof_reliability_{args.position.lower()}.csv'}")


def fit_baselines(
    route_table: Path,
    predictions_path: Path,
    receiver_summary_path: Path,
    metrics_path: Path,
    position: str | None = None,
    min_routes: int = 25,
    split_strategy: str = "week",
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    precomputed_splits: dict[str, list[dict[str, str]]] | None = None,
    include_dynamic_features: bool = False,
    dynamic_receiver_summary_path: Path | None = None,
) -> None:
    rows = read_csv_rows(route_table)
    rows = filter_rows(rows, position=position)
    if not rows:
        raise ValueError(f"No route rows available for position={position!r}")
    if include_dynamic_features:
        from .feature_schema import DYNAMIC_CONTEXT_FEATURES

        missing = [feature for feature in DYNAMIC_CONTEXT_FEATURES if feature not in rows[0]]
        if missing:
            raise ValueError(
                "Dynamic model requested, but the route table is missing dynamic features. "
                "Rebuild it with build-route-table --include-dynamic-features."
            )
    if precomputed_splits is None:
        splits = split_rows(
            rows,
            strategy=split_strategy,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    else:
        splits = {
            split: filter_rows(split_rows_data, position=position)
            for split, split_rows_data in precomputed_splits.items()
        }
    models = fit_models(
        splits["train"], include_dynamic_context=include_dynamic_features
    )
    metrics_rows = evaluate_models(models, splits)
    scoring_model = next(model for model in models if model.name == "ridge_context")

    write_metrics(metrics_path, metrics_rows)
    prediction_rows = [row for split_rows_data in splits.values() for row in split_rows_data]
    write_predictions(predictions_path, prediction_rows, models, scoring_model)
    summary_rows = splits["test"] if split_strategy in {"random", "game"} else rows
    write_receiver_summary(
        receiver_summary_path,
        summary_rows,
        scoring_model,
        min_routes=min_routes,
        position=position,
    )
    if include_dynamic_features:
        dynamic_model = next(
            model for model in models if model.name == "ridge_dynamic_context"
        )
        dynamic_summary_path = dynamic_receiver_summary_path or receiver_summary_path.with_name(
            receiver_summary_path.stem + "_dynamic_context.csv"
        )
        write_receiver_summary(
            dynamic_summary_path,
            summary_rows,
            dynamic_model,
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
    if include_dynamic_features:
        print(f"dynamic receiver summary: {dynamic_summary_path}")
    print(f"metrics: {metrics_path}")


def fit_position_baselines(
    route_table: Path,
    output_dir: Path,
    positions: list[str],
    min_routes: int,
    split_strategy: str = "week",
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    include_dynamic_features: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(route_table)
    all_splits = split_rows(
        rows,
        strategy=split_strategy,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    fit_baselines(
        route_table,
        output_dir / "route_level_baseline_predictions_all.csv",
        output_dir / "receiver_baseline_summary_all.csv",
        output_dir / "baseline_metrics_all.csv",
        position=None,
        min_routes=min_routes,
        split_strategy=split_strategy,
        precomputed_splits=all_splits,
        include_dynamic_features=include_dynamic_features,
        dynamic_receiver_summary_path=(
            output_dir / "receiver_dynamic_context_summary_all.csv"
        ),
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
            split_strategy=split_strategy,
            precomputed_splits=all_splits,
            include_dynamic_features=include_dynamic_features,
            dynamic_receiver_summary_path=(
                output_dir / f"receiver_dynamic_context_summary_{suffix}.csv"
            ),
        )


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-strategy",
        choices=["week", "random", "game"],
        default="week",
        help="Use a week holdout, random route split, or game-grouped split stratified by week.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
