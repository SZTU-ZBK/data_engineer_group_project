"""Test temporal feature-history isolation and content/history representations."""

import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.lightfm_features import build_enriched_features, split_feature_history
from src.models.lightfm import LightFMRecommender


class EnrichedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.users = pd.DataFrame({"user_id": [10, 30], "user_index": [0, 1]})
        self.user_meta = pd.DataFrame({"user_id": [30, 10], "gender": ["F", "F"],
                                       "age": [25, 25], "occupation": [4, 4]})
        self.movies = pd.DataFrame({"movie_id": [11, 22, 33], "movie_index": [0, 1, 2]})
        self.movie_meta = pd.DataFrame({"movie_id": [33, 22, 11, 99],
                                        "title": ["C (1985)", "B (1995)", "A (1980)", "Unknown (2050)"]})
        self.genres = csr_matrix([[1, 0], [0, 1], [1, 0]], dtype="float32")
        self.history = pd.DataFrame({"user_id": [10, 10, 30, 30], "movie_id": [11, 33, 22, 33],
                                     "rating": [5, 1, 4, 2]})

    def build(self, history=None):
        return build_enriched_features(self.users, self.user_meta, self.movies, self.movie_meta,
                                        self.genres, ["Action", "Drama"],
                                        self.history if history is None else history)

    def test_timestamp_groups_keep_feature_history_disjoint_from_fit_targets(self):
        train = pd.DataFrame({"event_id": range(7), "user_id": [10, 10, 10, 10, 30, 30, 80],
                              "movie_id": range(7), "timestamp": [1, 2, 2, 3, 1, 2, 1]})
        original = train.copy(deep=True)
        history, fit = split_feature_history(train)
        self.assertFalse(set(history.event_id) & set(fit.event_id))
        self.assertEqual(set(history.event_id) | set(fit.event_id), set(train.event_id))
        for user in fit.user_id.unique():
            self.assertLess(history.loc[history.user_id.eq(user), "timestamp"].max(),
                            fit.loc[fit.user_id.eq(user), "timestamp"].min())
        self.assertNotIn(80, fit.user_id.tolist())
        pd.testing.assert_frame_equal(train, original)

    def test_preferences_distinguish_identical_demographics_and_content_stats_distinguish_movies(self):
        features = self.build()
        self.assertEqual(features.users.shape, (2, 7))  # 3 demographics + 2 genres + 2 decades.
        self.assertEqual(features.items.shape, (3, 7))  # 2 genres + 2 decades + 3 numeric values.
        u, m = features.users.toarray(), features.items.toarray()
        np.testing.assert_allclose(u.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(m.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(u[0, :3], u[1, :3])
        self.assertGreater(u[0, features.user_names.index("preferred_genre:Action")], 0)
        self.assertEqual(u[1, features.user_names.index("preferred_genre:Action")], 0)
        self.assertFalse(np.array_equal(u[0], u[1]))
        self.assertFalse(np.array_equal(m[0], m[2]))  # Same genre/decade, different history statistics.
        self.assertFalse(any("2050" in name or "user_id" in name or "movie_id" in name
                             for name in features.user_names + features.item_names))

    def test_zero_history_movies_get_finite_smoothed_features_and_histories_are_not_mutated(self):
        history = self.history.loc[self.history.movie_id.ne(33)].copy()
        original = history.copy(deep=True)
        features = self.build(history)
        values = features.items.toarray()
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[2, features.item_names.index("history_log_popularity")], 0)
        self.assertGreater(values[2, features.item_names.index("history_smoothed_mean_rating")], 0)
        pd.testing.assert_frame_equal(history, original)

    def test_native_prediction_matches_export_with_inference_history_features(self):
        from lightfm import LightFM

        training = self.build(self.history.loc[self.history.movie_id.ne(33)])
        scoring = self.build()
        self.assertEqual(training.user_names, scoring.user_names)
        self.assertEqual(training.item_names, scoring.item_names)
        engine = LightFM(no_components=4, loss="logistic", random_state=42)
        interactions = csr_matrix([[0, -1, 0], [1, 0, 0]], dtype="float32")
        engine.fit(interactions, user_features=training.users, item_features=training.items,
                   epochs=2, num_threads=1)
        model = LightFMRecommender.from_fitted(engine, scoring.items, self.users.user_id,
                                               user_features=scoring.users)
        self.assertEqual(engine.user_embeddings.shape, (7, 4))
        self.assertEqual(engine.item_embeddings.shape, (7, 4))
        for index, user in enumerate(self.users.user_id):
            expected = engine.predict(index, np.arange(3), user_features=scoring.users,
                                      item_features=scoring.items, num_threads=1)
            np.testing.assert_allclose(model.score_tokens(user, []), expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
