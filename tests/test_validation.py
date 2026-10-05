import unittest

from separation_over_expected.validation import (
    assign_game_folds,
    assign_game_halves,
    average_ranks,
    reliability_metrics,
)


class GameGroupingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"week": str(week), "gameId": f"{week}-{game}"}
            for week in (1, 2)
            for game in range(10)
        ]

    def test_folds_are_deterministic_and_games_stay_together(self):
        first = assign_game_folds(self.rows, n_folds=5, seed=42)
        second = assign_game_folds(self.rows, n_folds=5, seed=42)
        self.assertEqual(first, second)
        for week in (1, 2):
            counts = [
                sum(fold == value for (assigned_week, _), value in first.items() if assigned_week == str(week))
                for fold in range(5)
            ]
            self.assertEqual(counts, [2, 2, 2, 2, 2])

    def test_halves_are_week_balanced(self):
        halves = assign_game_halves(self.rows, seed=7)
        for week in (1, 2):
            counts = [
                sum(half == label for (assigned_week, _), half in halves.items() if assigned_week == str(week))
                for label in ("A", "B")
            ]
            self.assertLessEqual(abs(counts[0] - counts[1]), 1)

    def test_average_ranks_assigns_midrank_to_ties(self):
        self.assertEqual(average_ranks([3.0, 1.0, 1.0, 4.0]), [3.0, 1.5, 1.5, 4.0])

    def test_reliability_reports_same_player_pairs_for_both_models(self):
        players = []
        for index in range(5):
            players.append(
                {
                    "nflId": str(index),
                    "displayName": f"Player {index}",
                    "mean_soe_ridge_context_A": str(index),
                    "mean_soe_ridge_context_B": str(index + 0.1),
                    "mean_soe_ridge_dynamic_context_A": str(index * 2),
                    "mean_soe_ridge_dynamic_context_B": str(index * 2 + 0.1),
                }
            )
        metrics = reliability_metrics(players, bootstrap_samples=100, seed=4)
        self.assertEqual(len(metrics), 3)
        self.assertAlmostEqual(float(metrics[0]["pearson"]), 1.0)
        self.assertAlmostEqual(float(metrics[1]["pearson"]), 1.0)
        self.assertEqual(metrics[2]["comparison"], "dynamic_minus_static")


if __name__ == "__main__":
    unittest.main()
