"""Check stage-local history weighting and isolation in the production helpers."""

import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.itemcf_history import add_context_weights, decay_histories, decay_scorer
from src.lightfm_features import ContentFeatures, feature_variant
from src.models.item_cf import ItemCF


class HistoryWeightTests(unittest.TestCase):
    def test_time_weights_use_only_known_history_and_exclude_low_ratings(self):
        history = pd.DataFrame({"user_id": [10, 10, 10], "movie_id": [1, 2, 3],
                                "timestamp": [0, 86400, 172800], "is_relevant": [1, 1, 0]})
        original = history.copy(deep=True)
        indexes, weights = decay_histories(history, {1: 0, 2: 1, 3: 2}, 1)[10]
        np.testing.assert_array_equal(indexes, [0, 1])
        np.testing.assert_allclose(weights, [.25, .5])
        pd.testing.assert_frame_equal(history, original)
        with self.assertRaises(ValueError):
            decay_histories(history, {1: 0}, 0)

    def test_weighted_scores_follow_source_item_direction_and_check_context(self):
        model = ItemCF(half_life_days=90)
        model.similarity_ = csr_matrix([[0, .5, .8], [.2, 0, .6], [.1, .3, 0]])
        model.n_items_ = 3
        score = decay_scorer(model, {10: (np.array([0, 1]), np.array([.25, 1]))})
        np.testing.assert_allclose(score(10, [1, 2]), [.2, .125, .8])
        with self.assertRaises(ValueError):
            score(10, [1, 3])

    def test_static_ablation_removes_all_history_columns_and_renormalizes(self):
        content = ContentFeatures(csr_matrix([[.2, .3, .5]]), csr_matrix([[.2, .2, .6]]),
                                  ["gender_F", "age_18", "preferred_genre:Drama"],
                                  ["genre:Drama", "release_decade_1990", "history_log_popularity"])
        static = feature_variant(content, [10], [1], "id_static", identity_features=False)
        self.assertEqual(static.user_names, ["gender_F", "age_18"])
        self.assertEqual(static.item_names, ["genre:Drama", "release_decade_1990"])
        np.testing.assert_allclose(static.users.toarray(), [[.4, .6]])
        np.testing.assert_allclose(static.items.toarray(), [[.5, .5]])
        self.assertEqual(content.users.shape[1], 3)

    def test_context_weights_follow_token_order_and_handle_empty_positive_history(self):
        histories = {10: (np.array([0, 1]), np.array([.25, 1]))}
        context = {"user_id": 10, "positive_movie_tokens": [2, 1]}
        weighted = add_context_weights(context, histories)
        self.assertEqual(weighted["positive_history_weights"], [1, .25])
        self.assertNotIn("positive_history_weights", context)
        empty = add_context_weights({"user_id": 99, "positive_movie_tokens": []}, histories)
        self.assertEqual(empty["positive_history_weights"], [])


if __name__ == "__main__":
    unittest.main()
