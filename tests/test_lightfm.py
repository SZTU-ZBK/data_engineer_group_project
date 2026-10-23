"""Test LightFM encoding, native logistic training, inference and WARP guard."""

from tempfile import TemporaryDirectory
import unittest

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.lightfm import (LightFMRecommender, build_item_features, build_user_features, build_logistic_interactions,
                               check_training_backend, WINDOWS_RNG_CAST)
from src.recommend import recommendation_table


class LightFMTests(unittest.TestCase):
    def setUp(self):
        self.genres = csr_matrix([[1, 0], [0, 1], [1, 1], [0, 1]], dtype="float32")
        self.user_ids = [10, 30, 80]

    def snapshot(self):
        users = np.array([[1, 0], [0, 1], [1, 1]], dtype="float32")
        items = np.array([[1, 2], [2, 1], [3, 0], [0, 3]], dtype="float32")
        fitted = SimpleNamespace(
            get_user_representations=lambda: (np.array([0.1, 0.2, 0.3]), users),
            get_item_representations=lambda features: (np.array([0, 0.1, 0.2, 0.3]), items),
        )
        return users, items, LightFMRecommender.from_fitted(fitted, None, self.user_ids)

    def test_genre_features_have_no_movie_identity_columns(self):
        features = build_item_features(self.genres).toarray()
        self.assertEqual(features.shape, (4, 2))
        np.testing.assert_allclose(features.sum(axis=1), 1)
        np.testing.assert_allclose(features[2], [0.5, 0.5])
        np.testing.assert_array_equal(features[1], features[3])
        with self.assertRaises(ValueError):
            build_item_features(csr_matrix([[0, 0]]))

    def user_metadata(self):
        return pd.DataFrame({"user_id": [80, 10, 30, 999], "gender": ["F", "F", "M", "X"],
                             "age": [1, 1, 18, 99], "occupation": [0, 0, 1, 99]})

    def test_user_categories_fit_train_users_and_ignore_identity(self):
        mapping = pd.DataFrame({"user_id": self.user_ids})
        features, names = build_user_features(mapping, self.user_metadata())
        self.assertEqual(features.shape, (3, 6))
        self.assertFalse(any("99" in name or "user_id" in name for name in names))
        np.testing.assert_allclose(features.toarray().sum(axis=1), 1)
        np.testing.assert_array_equal(features.toarray()[0], features.toarray()[2])
        renamed = self.user_metadata().replace({"user_id": {10: 11, 30: 31, 80: 81}})
        renamed_features, renamed_names = build_user_features(pd.DataFrame({"user_id": [11, 31, 81]}), renamed)
        np.testing.assert_array_equal(features.toarray(), renamed_features.toarray())
        self.assertEqual(names, renamed_names)
        with self.assertRaises(ValueError):
            build_user_features(mapping, self.user_metadata().query("user_id != 30"))

    def test_scores_use_correct_raw_user_id_and_both_biases(self):
        _, _, model = self.snapshot()
        np.testing.assert_allclose(model.score_tokens(30, []), [2.2, 1.3, 0.4, 3.5], atol=1e-6)
        with self.assertRaises(ValueError):
            model.score_tokens(999, [])

    def test_selected_snapshot_is_independent_and_survives_round_trip(self):
        users, items, selected = self.snapshot()
        before = selected.score_tokens(30, []).copy()
        users[:] = 100
        items[:] = 200
        np.testing.assert_array_equal(before, selected.score_tokens(30, []))
        with TemporaryDirectory() as directory:
            selected.save(directory)
            np.testing.assert_array_equal(before, LightFMRecommender.load(directory).score_tokens(30, []))

    def test_shared_recommendation_filter_includes_low_rated_history(self):
        _, _, model = self.snapshot()
        movies = pd.DataFrame({"movie_id": [11, 22, 33, 44], "title": list("ABCD"),
                               "genres": ["Comedy", "Drama", "Action", "Drama"]})
        context = {"user_id": 10, "positive_movie_tokens": [1], "seen_movie_ids": [11, 22]}
        result = recommendation_table(model, context, movies.movie_id, movies, k=2)
        self.assertEqual(set(result.movie_id), {33, 44})
        self.assertEqual(set(result.title), {"C", "D"})

    def test_windows_rng_guard_rejects_the_signed_cast(self):
        unsafe = "__pyx_r = (((long)__pyx_f_lightfm_temper((__pyx_v_seed[0]))) / 2);"
        safe = WINDOWS_RNG_CAST.sub(lambda match: f'({match["call"]} >> 1)', unsafe)
        self.assertNotEqual(safe, unsafe)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "_lightfm_fast_no_openmp.c"
            with patch("src.models.lightfm.sys.platform", "win32"), patch(
                "src.models.lightfm.importlib.util.find_spec",
                return_value=SimpleNamespace(origin=str(Path(directory) / "__init__.py")),
            ):
                source.write_text(unsafe)
                check_training_backend("logistic")
                with self.assertRaisesRegex(RuntimeError, "repair_lightfm_windows.py"):
                    check_training_backend()
                source.write_text(safe)
                check_training_backend()

    def test_logistic_labels_keep_unknown_and_neutral_unlabelled(self):
        ratings = csr_matrix([[5, 4, 3, 2, 1, 0]], dtype="float32")
        original = ratings.copy()
        signed = build_logistic_interactions(ratings)
        np.testing.assert_array_equal(signed.toarray(), [[1, 1, 0, -1, -1, 0]])
        np.testing.assert_array_equal(ratings.toarray(), original.toarray())
        with self.assertRaises(ValueError):
            build_logistic_interactions(csr_matrix([[5, 4, 3, 0]]))

    def test_native_logistic_training_and_exported_score_parity(self):
        from lightfm import LightFM

        ratings = csr_matrix([[5, 1, 0, 3], [4, 0, 2, 0], [0, 1, 5, 4]], dtype="float32")
        signed = build_logistic_interactions(ratings)
        features = build_item_features(self.genres)
        user_features, names = build_user_features(pd.DataFrame({"user_id": self.user_ids}), self.user_metadata())
        engine = LightFM(no_components=4, loss="logistic", random_state=42)
        engine.fit(signed, user_features=user_features, item_features=features, epochs=5, num_threads=1)
        self.assertEqual(engine.user_embeddings.shape, (len(names), 4))
        self.assertEqual(engine.item_embeddings.shape, (2, 4))
        exported = LightFMRecommender.from_fitted(engine, features, self.user_ids, user_features=user_features)
        snapshot = exported.score_tokens(30, []).copy()
        for index, user in enumerate(self.user_ids):
            expected = engine.predict(index, np.arange(4), user_features=user_features,
                                      item_features=features, num_threads=1)
            np.testing.assert_allclose(exported.score_tokens(user, []), expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(exported.score_tokens(10, []), exported.score_tokens(80, []))
        self.assertEqual(exported.score_tokens(10, [])[1], exported.score_tokens(10, [])[3])
        engine.fit_partial(signed, user_features=user_features, item_features=features, epochs=3, num_threads=1)
        np.testing.assert_array_equal(exported.score_tokens(30, []), snapshot)


if __name__ == "__main__":
    unittest.main()
