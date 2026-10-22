"""Numerical ItemCF examples and shared ranking-protocol checks."""

from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.evaluate import evaluate_ranking, rank_candidates, ranking_metrics
from src.models.item_cf import ItemCF
from src.recommend import recommendation_table


class ItemCFTests(unittest.TestCase):
    def setUp(self):
        self.interactions = csr_matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype="float32")

    def test_cosine_values_and_history_scoring(self):
        model = ItemCF(n_neighbors=2).fit(self.interactions)
        expected = np.array([[0, 0.5, 1 / np.sqrt(2)], [0.5, 0, 0], [1 / np.sqrt(2), 0, 0]])
        np.testing.assert_allclose(model.similarity_.toarray(), expected, rtol=1e-6)
        np.testing.assert_allclose(model.score([0]), expected[0], rtol=1e-6)
        np.testing.assert_allclose(model.score([0, 1]), expected[0] + expected[1], rtol=1e-6)
        np.testing.assert_allclose(model.score([0, 0]), model.score([0]))

    def test_neighbor_pruning_removes_weaker_edge(self):
        model = ItemCF(n_neighbors=1).fit(self.interactions)
        self.assertEqual(model.similarity_[0, 1], 0)
        self.assertAlmostEqual(float(model.similarity_[0, 2]), 1 / np.sqrt(2), places=6)
        self.assertTrue((np.diff(model.similarity_.indptr) <= 1).all())
        self.assertTrue((model.similarity_.diagonal() == 0).all())

    def test_shrinkage_downweights_small_overlap(self):
        plain = ItemCF(2, 0).fit(self.interactions)
        shrunk = ItemCF(2, 1).fit(self.interactions)
        np.testing.assert_allclose(shrunk.similarity_.toarray(), plain.similarity_.toarray() / 2)

    def test_zero_support_item_and_empty_history_remain_finite(self):
        model = ItemCF(2).fit(csr_matrix([[1, 0, 1], [1, 0, 0]], dtype="float32"))
        self.assertEqual(model.similarity_[1].nnz, 0)
        self.assertTrue(np.isfinite(model.score([0])).all())
        np.testing.assert_array_equal(model.score([]), np.zeros(3))

    def test_binary_input_and_index_validation(self):
        with self.assertRaises(ValueError):
            ItemCF().fit(csr_matrix([[5, 2]]))
        with self.assertRaises(ValueError):
            ItemCF().fit(csr_matrix((2, 3)))
        model = ItemCF().fit(self.interactions)
        for history in [[-1], [3], [[0, 1]]]:
            with self.assertRaises(ValueError):
                model.score(history)
        with self.assertRaises(RuntimeError):
            ItemCF().score([0])

    def test_training_input_is_unchanged_and_padding_is_ignored(self):
        original = self.interactions.copy()
        model = ItemCF().fit(self.interactions)
        self.assertEqual((self.interactions != original).nnz, 0)
        np.testing.assert_allclose(model.score_tokens(1, [0, 1, 2]), model.score([0, 1]))

    def test_saved_model_produces_the_same_scores(self):
        model = ItemCF(2, 10).fit(self.interactions)
        with TemporaryDirectory() as folder:
            model.save(folder)
            restored = ItemCF.load(folder)
            np.testing.assert_array_equal(restored.score([0, 2]), model.score([0, 2]))
            self.assertEqual(restored.n_neighbors, 2)
            self.assertEqual(restored.shrinkage, 10)

    def test_explanation_identifies_a_contributing_history_movie(self):
        model = ItemCF(2).fit(self.interactions)
        movies = pd.DataFrame({"movie_id": [10, 20, 30], "title": ["A", "B", "C"],
                               "genres": ["Comedy", "Drama", "Action"]})
        context = {"positive_movie_tokens": [1], "seen_movie_ids": [10, 20]}
        result = recommendation_table(model, context, [10, 20, 30], movies, k=1)
        self.assertEqual(result.iloc[0]["movie_id"], 30)
        self.assertEqual(result.iloc[0]["because_movie_id"], 10)
        self.assertEqual(result.iloc[0]["title"], "C")


class RankingTests(unittest.TestCase):
    def test_seen_movies_are_masked_and_score_ties_use_raw_ids(self):
        ranked, scores = rank_candidates([9, 0.5, 0.5, 0.2], [40, 30, 10, 20], [40, 999], k=3)
        self.assertEqual(ranked.tolist(), [10, 30, 20])
        np.testing.assert_allclose(scores, [0.5, 0.5, 0.2])

    def test_zero_score_ties_and_nonfinite_scores(self):
        ranked, _ = rank_candidates([0, 0, 0], [30, 10, 20], [10], k=5)
        self.assertEqual(ranked.tolist(), [20, 30])
        with self.assertRaises(ValueError):
            rank_candidates([np.nan, 1], [10, 20], [], k=1)

    def test_recall_and_discounted_gain_match_hand_calculation(self):
        result = ranking_metrics([30, 20, 10], {10, 30}, k_values=(1, 3))
        self.assertEqual(result["Recall@1"], 0.5)
        self.assertEqual(result["NDCG@1"], 1)
        self.assertEqual(result["Recall@3"], 1)
        self.assertAlmostEqual(result["NDCG@3"], (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
        with self.assertRaises(ValueError):
            ranking_metrics([10, 10], {10}, k_values=(2,))

    def test_evaluation_uses_shared_cohort_and_never_passes_targets_to_scorer(self):
        contexts = pd.DataFrame([
            {"user_id": 1, "eligible": True, "positive_movie_tokens": [1], "seen_movie_ids": [10],
             "relevant_movie_ids": [30], "candidate_count": 2},
            {"user_id": 2, "eligible": False, "positive_movie_tokens": [], "seen_movie_ids": [],
             "relevant_movie_ids": [], "candidate_count": 3},
        ])
        calls = []

        def score(user_id, tokens):
            calls.append((user_id, list(tokens)))
            return np.array([100, 0.1, 0.9])

        aggregate, per_user = evaluate_ranking(score, contexts, [10, 20, 30], k_values=(1,))
        self.assertEqual(calls, [(1, [1])])
        self.assertEqual(aggregate["users"], 1)
        self.assertEqual(aggregate["Recall@1"], 1)
        self.assertEqual(per_user["user_id"].tolist(), [1])
        self.assertAlmostEqual(aggregate["CatalogCoverage@1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
