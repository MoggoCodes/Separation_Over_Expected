from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

from .feature_schema import ROUTE_SHAPE_COLUMN


def identify_route_families(
    rows: list[dict[str, str]],
    position: str = "WR",
    min_clusters: int = 3,
    max_clusters: int = 10,
    seed: int = 42,
    silhouette_sample_size: int = 1500,
    stability_repeats: int = 4,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Cluster resampled, start-relative route paths for visualization and inspection."""
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Route-family discovery requires the development dependency scikit-learn."
        ) from exc

    if min_clusters < 2 or max_clusters < min_clusters:
        raise ValueError("Require 2 <= min_clusters <= max_clusters")
    if stability_repeats < 2:
        raise ValueError("stability_repeats must be at least 2")
    selected = [row for row in rows if row.get("officialPosition") == position]
    if not selected:
        raise ValueError(f"No route rows available for position={position!r}")

    route_shapes = []
    for row in selected:
        encoded = row.get(ROUTE_SHAPE_COLUMN, "")
        if not encoded:
            raise ValueError(
                "Route shape coordinates are missing. Rebuild the table with "
                "build-route-table --include-route-geometry."
            )
        points = json.loads(encoded)
        if len(points) != 11 or any(len(point) != 2 for point in points):
            raise ValueError("Each route shape must contain 11 two-dimensional points")
        route_shapes.append(points)

    shape_array = np.asarray(route_shapes, dtype=float)
    # The first point is always the route origin, so omit its constant coordinates.
    flattened = shape_array[:, 1:, :].reshape(len(shape_array), -1)
    scaled = StandardScaler().fit_transform(flattened)
    sample_count = min(len(selected), silhouette_sample_size)
    sample_indices = np.random.default_rng(seed).choice(
        len(selected), size=sample_count, replace=False
    )
    sample_scaled = scaled[sample_indices]
    candidate_results = []
    candidate_fits = {}
    for cluster_count in range(min_clusters, max_clusters + 1):
        if cluster_count >= len(selected):
            continue
        runs = []
        for repeat in range(stability_repeats):
            model = KMeans(
                n_clusters=cluster_count,
                random_state=seed + repeat,
                n_init=5,
                max_iter=100,
                algorithm="lloyd",
            ).fit(scaled)
            labels = model.labels_
            runs.append((model, labels))
        run_silhouettes = [
            float(
                silhouette_score(
                    sample_scaled,
                    labels[sample_indices],
                    metric="euclidean",
                )
            )
            for _, labels in runs
        ]
        stability_values = [
            float(adjusted_rand_score(left[sample_indices], right[sample_indices]))
            for (_, left), (_, right) in combinations(runs, 2)
        ]
        best_run_index = max(
            range(len(runs)), key=lambda index: (run_silhouettes[index], -index)
        )
        candidate_fits[cluster_count] = runs[best_run_index]
        candidate_results.append(
            {
                "n_clusters": cluster_count,
                "silhouette": sum(run_silhouettes) / len(run_silhouettes),
                "silhouette_sd": _population_sd(run_silhouettes),
                "mean_pairwise_ari": sum(stability_values) / len(stability_values),
                "best_run_silhouette": run_silhouettes[best_run_index],
            }
        )
    if not candidate_results:
        raise ValueError("Not enough route rows to evaluate the requested cluster counts")

    chosen = max(
        candidate_results,
        key=lambda result: (result["silhouette"], -result["n_clusters"]),
    )
    chosen_k = int(chosen["n_clusters"])
    _, labels = candidate_fits[chosen_k]
    metrics = [
        {
            "position": position,
            "n_routes": str(len(selected)),
            "n_clusters": str(result["n_clusters"]),
            "mean_silhouette": f'{result["silhouette"]:.6f}',
            "silhouette_sd": f'{result["silhouette_sd"]:.6f}',
            "mean_pairwise_ari": f'{result["mean_pairwise_ari"]:.6f}',
            "selected": str(result["n_clusters"] == chosen_k).lower(),
        }
        for result in candidate_results
    ]

    assignments = []
    summaries = []
    for cluster_id in range(chosen_k):
        member_indices = np.flatnonzero(labels == cluster_id)
        if len(member_indices) == 0:
            continue
        members = shape_array[member_indices]
        mean_shape = members.mean(axis=0)
        medoid_index = int(
            member_indices[
                np.square(members - mean_shape).sum(axis=(1, 2)).argmin()
            ]
        )
        medoid_row = selected[medoid_index]
        summaries.append(
            {
                "cluster_id": str(cluster_id),
                "route_count": str(len(member_indices)),
                "route_share": f"{len(member_indices) / len(selected):.6f}",
                "representative_gameId": medoid_row["gameId"],
                "representative_playId": medoid_row["playId"],
                "representative_nflId": medoid_row["nflId"],
                "representative_name": medoid_row.get("displayName", ""),
                "representative_path": json.dumps(
                    [[round(float(x), 4), round(float(y), 4)] for x, y in shape_array[medoid_index]],
                    separators=(",", ":"),
                ),
                "mean_path": json.dumps(
                    [[round(float(x), 4), round(float(y), 4)] for x, y in mean_shape],
                    separators=(",", ":"),
                ),
            }
        )
    for row, label in zip(selected, labels):
        assignments.append(
            {
                "gameId": row["gameId"],
                "playId": row["playId"],
                "nflId": row["nflId"],
                "displayName": row.get("displayName", ""),
                "week": row.get("week", ""),
                "cluster_id": str(int(label)),
            }
        )
    return assignments, metrics, summaries


def write_route_family_outputs(
    output_dir: Path,
    position: str,
    assignments: list[dict[str, str]],
    metrics: list[dict[str, str]],
    summaries: list[dict[str, str]],
) -> None:
    suffix = position.lower()
    _write_rows(
        output_dir / f"route_family_assignments_{suffix}.csv",
        assignments,
        ["gameId", "playId", "nflId", "displayName", "week", "cluster_id"],
    )
    _write_rows(
        output_dir / f"route_family_metrics_{suffix}.csv",
        metrics,
        ["position", "n_routes", "n_clusters", "mean_silhouette", "silhouette_sd", "mean_pairwise_ari", "selected"],
    )
    _write_rows(
        output_dir / f"route_family_summaries_{suffix}.csv",
        summaries,
        ["cluster_id", "route_count", "route_share", "representative_gameId", "representative_playId", "representative_nflId", "representative_name", "representative_path", "mean_path"],
    )


def _write_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _population_sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
