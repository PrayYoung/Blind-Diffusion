import unittest

from blind_diffusion.utils.metrics import summarize_episode_metrics


class MetricsTests(unittest.TestCase):
    def test_summarize_episode_metrics_counts_and_averages(self):
        metrics = summarize_episode_metrics(
            [
                {"success": 1.0, "collision": 0.0, "return": 3.0, "length": 5},
                {"success": 0.0, "collision": 1.0, "return": 1.0, "length": 7},
            ]
        )

        self.assertEqual(metrics["num_episodes"], 2)
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["collision_count"], 1)
        self.assertAlmostEqual(metrics["success_rate"], 0.5)
        self.assertAlmostEqual(metrics["collision_rate"], 0.5)
        self.assertAlmostEqual(metrics["avg_return"], 2.0)
        self.assertAlmostEqual(metrics["avg_length"], 6.0)


if __name__ == "__main__":
    unittest.main()
