"""Check D's k-means, cluster scores and score fusion on small synthetic inputs."""

import unittest

import numpy as np

from scripts.models.clustering import aggregate, cluster_similarities, normalize_rows, weighted_kmeans
from scripts.models.fusion import Metrics, component_weights, fuse, normalize, rank


class ClusteringTest(unittest.TestCase):
    def test_kmeans_separates_two_directions(self):
        rng = np.random.default_rng(0)
        a = np.array([1.0, 0, 0]) + 0.05 * rng.standard_normal((40, 3))
        b = np.array([0, 1.0, 0]) + 0.05 * rng.standard_normal((20, 3))
        points = normalize_rows(np.vstack([a, b]))
        weights = np.r_[np.ones(40), 2 * np.ones(20)]
        centres, labels, shares = weighted_kmeans(points, weights, 2, np.random.default_rng(42))
        self.assertEqual(len(set(labels[:40])), 1)
        self.assertEqual(len(set(labels[40:])), 1)
        self.assertNotEqual(labels[0], labels[40])
        self.assertAlmostEqual(shares.sum(), 1.0)
        self.assertAlmostEqual(sorted(shares)[0], 0.5)          # 20 points × weight 2 versus 40 × 1

    def test_k_is_capped_and_scores_cover_all_users(self):
        users = normalize_rows(np.random.default_rng(1).standard_normal((50, 4)).astype(np.float32))
        candidates = np.array([3, 7])
        similarities, shares = cluster_similarities(users, candidates, np.array([1.0, 3.0]), 5,
                                                    np.random.default_rng(42))
        self.assertEqual(similarities.shape, (50, 2))
        self.assertEqual(aggregate(similarities, shares, "wsum").shape, (50,))
        self.assertTrue(np.allclose(aggregate(similarities, shares, "max")[candidates], 1.0, atol=1e-5))


class FusionTest(unittest.TestCase):
    def test_normalisation_and_weights(self):
        self.assertTrue(np.allclose(normalize(np.array([2.0, 4.0, 6.0]), "minmax"), [0, 0.5, 1]))
        self.assertTrue(np.allclose(normalize(np.ones(4), "zscore"), 0))
        self.assertEqual(component_weights(0.25, 0.8), (0.2, 0.6, 0.2))
        self.assertEqual(component_weights(1.0, 0.0), (0.0, 0.0, 1.0))

    def test_ties_go_to_active_users_and_scores_win(self):
        activity_rank = np.arange(1500)[::-1].copy()       # user 1499 is the most active
        score = fuse({"pool": np.zeros(1500)}, (1, 0, 0))
        score[[3, 4]] = [0.5, 1e-9]                         # a tiny real difference must not be overridden
        ranked = rank(score, activity_rank)
        self.assertEqual(ranked[:4].tolist(), [3, 4, 1499, 1498])

    def test_metrics_match_definition(self):
        metrics = Metrics(2000)
        ranked = np.arange(1000)
        values = metrics.track(ranked, np.array([0, 2, 1500]))
        self.assertAlmostEqual(values[0], 2 / 3)                 # Recall@100
        ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
        self.assertAlmostEqual(values[3], (1 + 1 / np.log2(4)) / ideal)   # NDCG@100


if __name__ == "__main__":
    unittest.main()
