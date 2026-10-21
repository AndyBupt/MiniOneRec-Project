import math
import unittest
from collections import Counter

from scripts.score_recommendations import score_predictions


class RecommendationMetricsTest(unittest.TestCase):
    def test_ranks_invalid_after_hit_duplicates_and_collisions(self):
        rows = [
            {"output": "A\n", "predict": ["A", "bad", "A", "B", "C"]},
            {"output": "C", "predict": ["B", "A", "bad", "C", "bad"]},
            {"output": "B", "predict": ["A", "C", "A", "C", "A"]},
        ]
        result = score_predictions(rows, Counter({"A": 1, "B": 2, "C": 1}))
        self.assertAlmostEqual(result["HR@5"], 2 / 3)
        self.assertAlmostEqual(result["NDCG@5"], (1 + 1 / math.log2(5)) / 3)
        self.assertAlmostEqual(result["invalid_sid@5"], 3 / 15)
        self.assertAlmostEqual(result["duplicate_slots@5"], 5 / 15)
        self.assertAlmostEqual(result["missing_slots@10"], 15 / 30)
        self.assertAlmostEqual(result["invalid_sid@10"], 18 / 30)
        self.assertAlmostEqual(result["target_collision_query_fraction"], 1 / 3)
        self.assertEqual(result["unique_target_sid_subset"]["queries"], 2)
        self.assertEqual(result["unique_target_sid_subset"]["HR@5"], 1.0)

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            score_predictions([{"output": "unknown", "predict": ["A"]}], Counter({"A": 1}))


if __name__ == "__main__":
    unittest.main()
