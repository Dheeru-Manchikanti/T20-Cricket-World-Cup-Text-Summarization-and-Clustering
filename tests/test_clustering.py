"""Unit tests for document clustering."""

import unittest
import pandas as pd
from src.clustering import CricketNewsClusterer, compute_elbow_curve


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "India won the T20 World Cup final against South Africa with Bumrah taking key wickets.",
            "Virat Kohli scored 76 runs in the T20 World Cup final at Kensington Oval Barbados.",
            "Australia defeated England in Barbados with Adam Zampa taking two crucial wickets.",
            "Pat Cummins took consecutive hat-tricks against Bangladesh and Afghanistan in T20 World Cup.",
            "United States stunned Pakistan in Dallas Super Over with Saurabh Netravalkar bowling composedly.",
            "Afghanistan defeated Australia in St Vincent with Gulbadin Naib claiming four wickets.",
            "Nassau County New York cricket stadium hosted low scoring matches between India and Pakistan.",
            "Brian Lara Cricket Academy in Trinidad hosted the first semifinal between South Africa and Afghanistan.",
        ]

    def test_fit_and_assign_clusters(self):
        clusterer = CricketNewsClusterer(n_clusters=3, random_state=42)
        clusterer.fit(self.texts)
        self.assertTrue(clusterer.is_fitted)
        self.assertEqual(len(clusterer.cluster_keywords), 3)

        cluster_ids, cluster_labels = clusterer.get_cluster_assignments(self.texts)
        self.assertEqual(len(cluster_ids), len(self.texts))
        self.assertEqual(len(cluster_labels), len(self.texts))
        self.assertTrue(all(0 <= cid < 3 for cid in cluster_ids))

    def test_predict_custom_article(self):
        clusterer = CricketNewsClusterer(n_clusters=3, random_state=42)
        clusterer.fit(self.texts)

        custom_text = "Jasprit Bumrah bowled Yorker deliveries in the final over to win the World Cup for India."
        pred = clusterer.predict_custom_article(custom_text)
        self.assertIn("predicted_cluster_id", pred)
        self.assertIn("cluster_label", pred)
        self.assertIn("cluster_keywords", pred)
        self.assertIn("centroid_distance", pred)
        self.assertTrue(0 <= pred["predicted_cluster_id"] < 3)

    def test_elbow_curve_computation(self):
        res = compute_elbow_curve(self.texts, max_k=4)
        self.assertIn("k_values", res)
        self.assertIn("inertias", res)
        self.assertIn("silhouettes", res)
        self.assertGreaterEqual(len(res["k_values"]), 2)

    def test_small_dataset_clamping(self):
        short_texts = [
            "Cricket match one between India and Pakistan.",
            "Cricket match two between Australia and England."
        ]
        clusterer = CricketNewsClusterer(n_clusters=5)
        # Should gracefully clamp clusters to usable docs
        clusterer.fit(short_texts)
        self.assertEqual(clusterer.n_clusters, 2)


if __name__ == "__main__":
    unittest.main()
