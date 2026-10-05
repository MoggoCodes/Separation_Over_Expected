import json
import unittest

from separation_over_expected.feature_schema import ROUTE_GEOMETRY_FEATURES, ROUTE_SHAPE_COLUMN
from separation_over_expected.features import resample_route_xy, route_geometry_features


class RouteGeometryTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = {
            10: {"wr": {"x": 1.0, "y": 10.0}},
            11: {"wr": {"x": 2.0, "y": 10.0}},
            12: {"wr": {"x": 2.0, "y": 12.0}},
            13: {"wr": {"x": 30.0, "y": 50.0}},
        }

    def test_route_features_include_release_and_ignore_later_frames(self):
        result = route_geometry_features(
            self.trajectory,
            receiver_id="wr",
            snap_frame=10,
            release_frame=12,
            play_direction="right",
        )
        self.assertEqual(set(result), set(ROUTE_GEOMETRY_FEATURES) | {ROUTE_SHAPE_COLUMN})
        self.assertEqual(result["route_path_length_pre_release"], "3.000")
        self.assertEqual(result["route_chord_length_pre_release"], "2.236")
        self.assertEqual(result["route_directness_pre_release"], "0.745")
        self.assertEqual(result["route_lateral_excursion_max_pre_release"], "2.000")
        self.assertEqual(len(json.loads(result[ROUTE_SHAPE_COLUMN])), 11)

    def test_geometry_is_consistent_after_field_direction_normalization(self):
        right = route_geometry_features(
            self.trajectory,
            receiver_id="wr",
            snap_frame=10,
            release_frame=12,
            play_direction="right",
        )
        mirrored = {
            frame: {
                "wr": {"x": 120.0 - row["wr"]["x"], "y": 53.3 - row["wr"]["y"]}
            }
            for frame, row in self.trajectory.items()
        }
        left = route_geometry_features(
            mirrored,
            receiver_id="wr",
            snap_frame=10,
            release_frame=12,
            play_direction="left",
        )
        for feature in ROUTE_GEOMETRY_FEATURES:
            self.assertEqual(right[feature], left[feature], feature)
        self.assertEqual(
            json.loads(right[ROUTE_SHAPE_COLUMN]),
            json.loads(left[ROUTE_SHAPE_COLUMN]),
        )

    def test_missing_or_single_sample_route_returns_blank_geometry(self):
        result = route_geometry_features(
            {10: {"other": {"x": 1.0, "y": 2.0}}},
            receiver_id="wr",
            snap_frame=10,
            release_frame=10,
            play_direction="right",
        )
        self.assertTrue(all(value == "" for value in result.values()))

    def test_resample_interpolates_equal_time_points(self):
        points = resample_route_xy([(0.0, (0.0, 0.0)), (2.0, (2.0, 4.0))], 5)
        self.assertEqual(points, [(0.0, 0.0), (0.5, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0)])


if __name__ == "__main__":
    unittest.main()
