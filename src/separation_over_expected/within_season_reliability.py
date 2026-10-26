from __future__ import annotations

import random
import statistics
from collections import defaultdict

from .validation import assign_game_halves, average_ranks, pearson_correlation, percentile


DEFAULT_COHORTS = ((10, 3), (20, 5), (30, 5))


def evaluate_within_season_receiver_reliability(
    predictions: list[dict[str, str]],
    cohorts: tuple[tuple[int, int], ...] = DEFAULT_COHORTS,
    split_seeds: int = 100,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Measure 2023 receiver residual repeatability across repeated game splits.

    Each route's dynamic Ridge residual is held fixed. Whole games are assigned
    to halves, balanced within week; eligibility is assessed separately in
    each half by route and game counts. Summary percentiles describe split
    assignment sensitivity, not population confidence intervals.
    """
    if split_seeds < 1 or bootstrap_samples < 1:
        raise ValueError("split_seeds and bootstrap_samples must be positive")
    if not cohorts or any(routes < 1 or games < 1 for routes, games in cohorts):
        raise ValueError("cohorts must contain positive route and game thresholds")
    required = {"season", "week", "gameId", "playId", "nflId", "residual_dynamic"}
    rows = [row for row in predictions if row.get("season") == "2023"]
    if not rows:
        raise ValueError("No 2023 route predictions were provided")
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"2023 rows must contain {sorted(required)}")
    route_keys = [(row["gameId"], row["playId"], row["nflId"]) for row in rows]
    if len(set(route_keys)) != len(route_keys):
        raise ValueError("2023 route predictions contain duplicate game/play/receiver keys")

    for row in rows:
        try:
            float(row["residual_dynamic"])
            int(row["week"])
        except (TypeError, ValueError) as exc:
            raise ValueError("2023 predictions contain invalid week or dynamic residual values") from exc

    split_rows: list[dict[str, str]] = []
    split_values: dict[tuple[int, int], dict[str, list[float]]] = {
        cohort: {
            "pearson": [], "spearman": [], "receivers": [],
            "pearson_bootstrap": [], "spearman_bootstrap": [],
        }
        for cohort in cohorts
    }
    for split_index in range(split_seeds):
        split_seed = seed + 10_000 + split_index * 37
        rng = random.Random(split_seed + 1)
        halves = assign_game_halves(rows, seed=split_seed)
        by_player_half: dict[tuple[str, str], list[float]] = defaultdict(list)
        games_by_player_half: dict[tuple[str, str], set[str]] = defaultdict(set)
        half_totals: dict[str, list[float]] = {"A": [], "B": []}
        names: dict[str, str] = {}
        for row in rows:
            half = halves[(str(int(row["week"])), row["gameId"])]
            residual = float(row["residual_dynamic"])
            key = (row["nflId"], half)
            by_player_half[key].append(residual)
            games_by_player_half[key].add(row["gameId"])
            half_totals[half].append(residual)
            names[row["nflId"]] = row.get("displayName", "")

        league_mean = {
            half: statistics.fmean(values) for half, values in half_totals.items()
        }
        players = sorted(names)
        player_means = {
            (player, half): statistics.fmean(by_player_half[(player, half)]) - league_mean[half]
            for player in players
            for half in ("A", "B")
            if by_player_half[(player, half)]
        }
        for min_routes, min_games in cohorts:
            eligible = [
                player for player in players
                if all(
                    len(by_player_half[(player, half)]) >= min_routes
                    and len(games_by_player_half[(player, half)]) >= min_games
                    for half in ("A", "B")
                )
            ]
            pearson = spearman = ""
            pearson_ci_lower = pearson_ci_upper = ""
            spearman_ci_lower = spearman_ci_upper = ""
            if len(eligible) >= 3:
                xs = [player_means[(player, "A")] for player in eligible]
                ys = [player_means[(player, "B")] for player in eligible]
                pearson_value = pearson_correlation(xs, ys)
                spearman_value = pearson_correlation(average_ranks(xs), average_ranks(ys))
                pearson, spearman = f"{pearson_value:.8f}", f"{spearman_value:.8f}"
                pearson_bootstrap = []
                spearman_bootstrap = []
                for _ in range(bootstrap_samples):
                    indices = [rng.randrange(len(eligible)) for _ in eligible]
                    sample_x = [xs[index] for index in indices]
                    sample_y = [ys[index] for index in indices]
                    pearson_bootstrap.append(pearson_correlation(sample_x, sample_y))
                    spearman_bootstrap.append(
                        pearson_correlation(average_ranks(sample_x), average_ranks(sample_y))
                    )
                pearson_bootstrap.sort()
                spearman_bootstrap.sort()
                pearson_ci_lower = f"{percentile(pearson_bootstrap, .025):.8f}"
                pearson_ci_upper = f"{percentile(pearson_bootstrap, .975):.8f}"
                spearman_ci_lower = f"{percentile(spearman_bootstrap, .025):.8f}"
                spearman_ci_upper = f"{percentile(spearman_bootstrap, .975):.8f}"
                cohort_values = split_values[(min_routes, min_games)]
                cohort_values["pearson"].append(pearson_value)
                cohort_values["spearman"].append(spearman_value)
                cohort_values["pearson_bootstrap"].extend(pearson_bootstrap)
                cohort_values["spearman_bootstrap"].extend(spearman_bootstrap)
            split_values[(min_routes, min_games)]["receivers"].append(float(len(eligible)))
            split_rows.append({
                "season": "2023",
                "model": "frozen_2021_dynamic_ridge",
                "seed": str(split_seed),
                "min_routes_per_half": str(min_routes),
                "min_games_per_half": str(min_games),
                "eligible_receivers": str(len(eligible)),
                "pearson": pearson,
                "spearman": spearman,
                "pearson_ci_lower": pearson_ci_lower,
                "pearson_ci_upper": pearson_ci_upper,
                "spearman_ci_lower": spearman_ci_lower,
                "spearman_ci_upper": spearman_ci_upper,
            })

    summary_rows: list[dict[str, str]] = []
    for min_routes, min_games in cohorts:
        values = split_values[(min_routes, min_games)]
        valid_splits = len(values["pearson"])
        row = {
            "season": "2023",
            "model": "frozen_2021_dynamic_ridge",
            "min_routes_per_half": str(min_routes),
            "min_games_per_half": str(min_games),
            "valid_splits": str(valid_splits),
        }
        for metric in ("pearson", "spearman", "receivers"):
            metric_values = values[metric]
            if not metric_values:
                row.update({f"{metric}_median": "", f"{metric}_p10": "", f"{metric}_p90": ""})
                continue
            ordered = sorted(metric_values)
            row.update({
                f"{metric}_median": f"{statistics.median(ordered):.8f}",
                f"{metric}_p10": f"{percentile(ordered, .10):.8f}",
                f"{metric}_p90": f"{percentile(ordered, .90):.8f}",
            })
        for metric in ("pearson", "spearman"):
            bootstrap_values = sorted(values[f"{metric}_bootstrap"])
            row[f"{metric}_bootstrap_ci_lower"] = (
                f"{percentile(bootstrap_values, .025):.8f}" if bootstrap_values else ""
            )
            row[f"{metric}_bootstrap_ci_upper"] = (
                f"{percentile(bootstrap_values, .975):.8f}" if bootstrap_values else ""
            )
        summary_rows.append(row)
    return split_rows, summary_rows
