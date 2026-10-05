import unittest

from separation_over_expected.feature_schema import POCKET_CONTEXT_FEATURES
from separation_over_expected.features import pocket_context_features


class PocketFeatureTests(unittest.TestCase):
    def setUp(self):
        self.snap_positions = {
            "qb": {"x": 10.0, "y": 20.0, "s": 1.0, "a": 0.5},
            "rusher": {"x": 15.0, "y": 20.0, "s": 2.0, "a": 0.5},
        }

    def test_pre_release_qb_and_pressure_features_exclude_release_frame(self):
        trajectory = {
            10: {
                "qb": {"x": 10.0, "y": 20.0, "s": 1.0, "a": 0.5},
                "rusher": {"x": 15.0, "y": 20.0, "s": 2.0, "a": 0.5},
            },
            11: {
                "qb": {"x": 8.0, "y": 21.0, "s": 3.0, "a": 1.5},
                "rusher": {"x": 11.0, "y": 21.0, "s": 4.0, "a": 1.5},
            },
            # This deliberately extreme release-frame row must not affect features.
            12: {
                "qb": {"x": 0.0, "y": 50.0, "s": 9.0, "a": 9.0},
                "rusher": {"x": 0.0, "y": 50.0, "s": 9.0, "a": 9.0},
            },
        }
        result = pocket_context_features(
            trajectory,
            snap_frame=10,
            release_frame=12,
            snap_positions=self.snap_positions,
            passer_ids={"qb"},
            rusher_ids={"rusher"},
            play_direction="right",
        )
        self.assertEqual(set(result), set(POCKET_CONTEXT_FEATURES))
        self.assertEqual(result["qb_depth_drop_pre_release"], "-2.000")
        self.assertEqual(result["qb_lateral_drift_pre_release"], "1.000")
        self.assertEqual(result["qb_mean_speed_pre_release"], "2.000")
        self.assertEqual(result["qb_nearest_rusher_min_dist_pre_release"], "3.000")
        self.assertEqual(result["qb_rusher_closing_rate_pre_release"], "20.000")
        self.assertEqual(result["qb_pressure_observed_fraction_pre_release"], "1.000")

    def test_missing_passer_returns_missing_features(self):
        result = pocket_context_features(
            trajectory={},
            snap_frame=1,
            release_frame=3,
            snap_positions={},
            passer_ids=set(),
            rusher_ids=set(),
            play_direction="right",
        )
        self.assertEqual(set(result), set(POCKET_CONTEXT_FEATURES))
        self.assertTrue(all(value == "" for value in result.values()))


if __name__ == "__main__":
    unittest.main()
