from __future__ import annotations

from separation_over_expected.receiver_reliability import summarize_cross_season_receivers
from separation_over_expected.validation import pearson_correlation


DEFAULT_COHORTS = ((10, 5), (20, 5), (30, 5), (20, 8))
INFLUENTIAL_RECEIVERS = ("Isaiah McKenzie", "Rondale Moore")


def audit_cross_season_receiver_validity(
    predictions: list[dict[str, str]],
    cohorts: tuple[tuple[int, int], ...] = DEFAULT_COHORTS,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Measure receiver-repeatability sensitivity to sample cutoffs and players."""
    cohort_rows: list[dict[str, str]] = []
    receiver_rows: list[dict[str, str]] = []
    baseline_receivers: list[dict[str, str]] = []
    for min_routes, min_games in cohorts:
        summaries, metrics, diagnostics = summarize_cross_season_receivers(
            predictions,
            min_routes=min_routes,
            min_games=min_games,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        eligible = [row for row in summaries if row["eligible"] == "true"]
        if min_routes == 20 and min_games == 5:
            baseline_receivers = eligible
        for metric in metrics:
            cohort_rows.append({
                "min_routes": str(min_routes),
                "min_games": str(min_games),
                "eligible_receivers": str(diagnostics["eligible_shared_players"]),
                **metric,
            })
        for receiver in eligible:
            receiver_rows.append({
                "min_routes": str(min_routes),
                "min_games": str(min_games),
                **receiver,
            })

    loo_rows: list[dict[str, str]] = []
    for model in ("static", "dynamic"):
        x_key, y_key = f"centered_residual_2021_{model}", f"centered_residual_2023_{model}"
        xs = [float(row[x_key]) for row in baseline_receivers]
        ys = [float(row[y_key]) for row in baseline_receivers]
        if len(xs) < 3:
            continue
        full = pearson_correlation(xs, ys)
        for omitted in baseline_receivers:
            reduced = [row for row in baseline_receivers if row["nflId"] != omitted["nflId"]]
            corr = pearson_correlation(
                [float(row[x_key]) for row in reduced],
                [float(row[y_key]) for row in reduced],
            )
            loo_rows.append({
                "model": model,
                "omitted_nflId": omitted["nflId"],
                "omitted_receiver": omitted["displayName"],
                "receivers_remaining": str(len(reduced)),
                "pearson": f"{corr:.8f}",
                "change_vs_full": f"{corr - full:.8f}",
            })
        pair = [row for row in baseline_receivers if row["displayName"] in INFLUENTIAL_RECEIVERS]
        reduced = [row for row in baseline_receivers if row["displayName"] not in INFLUENTIAL_RECEIVERS]
        if len(pair) == len(INFLUENTIAL_RECEIVERS) and len(reduced) >= 3:
            corr = pearson_correlation(
                [float(row[x_key]) for row in reduced],
                [float(row[y_key]) for row in reduced],
            )
            loo_rows.append({
                "model": model,
                "omitted_nflId": ",".join(row["nflId"] for row in pair),
                "omitted_receiver": "Isaiah McKenzie + Rondale Moore",
                "receivers_remaining": str(len(reduced)),
                "pearson": f"{corr:.8f}",
                "change_vs_full": f"{corr - full:.8f}",
            })
        loo_rows.append({
            "model": model,
            "omitted_nflId": "",
            "omitted_receiver": "FULL COHORT",
            "receivers_remaining": str(len(baseline_receivers)),
            "pearson": f"{full:.8f}",
            "change_vs_full": "0.00000000",
        })
    return cohort_rows, receiver_rows, loo_rows
