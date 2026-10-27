import json
import unittest

from separation_over_expected.feature_schema import ROUTE_SHAPE_COLUMN
from separation_over_expected.route_clustering import identify_route_families


class RouteClusteringTests(unittest.TestCase):
    def test_clustering_is_reproducible_and_returns_inspectable_representatives(self):
        rows = []
        for index in range(16):
            lateral_sign = -1 if index < 8 else 1
            drift = (index % 4) * 0.03
            points = [[round(step * 0.5, 4), round(lateral_sign * step * 0.25 + drift, 4)] for step in range(11)]
            rows.append(
                {
                    "gameId": str(index),
                    "playId": "1",
                    "nflId": str(index),
                    "displayName": f"WR {index}",
                    "week": "1",
                    "officialPosition": "WR",
                    ROUTE_SHAPE_COLUMN: json.dumps(points),
                }
            )
        result = identify_route_families(
            rows,
            position="WR",
            min_clusters=2,
            max_clusters=2,
            seed=17,
            silhouette_sample_size=16,
            stability_repeats=2,
        )
        repeat = identify_route_families(
            rows,
            position="WR",
            min_clusters=2,
            max_clusters=2,
            seed=17,
            silhouette_sample_size=16,
            stability_repeats=2,
        )
        assignments, metrics, summaries = result
        self.assertEqual(assignments, repeat[0])
        self.assertEqual(len(assignments), len(rows))
        self.assertEqual({row["cluster_id"] for row in assignments}, {"0", "1"})
        self.assertEqual(metrics[0]["selected"], "true")
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(json.loads(summaries[0]["representative_path"])), 11)


if __name__ == "__main__":
    unittest.main()
