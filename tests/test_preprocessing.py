"""Behavioral checks for temporal boundaries and held-out-data isolation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from scipy.sparse import load_npz

from src.data import add_relevance_labels, chronological_split, clean_ratings, validate_splits
from src.features import (
    build_evaluation_context, build_model_inputs, candidate_movie_ids,
    evaluation_summary, save_preprocessed_artifacts,
)


def events(rows):
    frame = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    frame.insert(0, "event_id", range(len(frame)))
    frame["rated_at"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    return frame


class TemporalSplitTests(unittest.TestCase):
    def setUp(self):
        self.frame = events([(1, movie, 5, 100) for movie in range(1, 9)] + [
            (1, 9, 3, 200), (1, 10, 4, 300), (2, 11, 2, 100), (2, 12, 5, 200),
        ])

    def test_ties_stay_together_and_short_history_stays_in_train(self):
        splits = chronological_split(self.frame)
        self.assertEqual(splits["train"]["event_id"].tolist(), list(range(8)) + [10, 11])
        self.assertEqual(splits["validation"]["event_id"].tolist(), [8])
        self.assertEqual(splits["test"]["event_id"].tolist(), [9])
        self.assertTrue((validate_splits(splits, self.frame)["violations"] == 0).all())

    def test_result_is_independent_of_input_row_order(self):
        expected = chronological_split(self.frame)
        shuffled = chronological_split(self.frame.sample(frac=1, random_state=42))
        for name in expected:
            pd.testing.assert_frame_equal(expected[name], shuffled[name])

    def test_labels_preserve_original_scores_and_low_ratings(self):
        splits = add_relevance_labels(chronological_split(self.frame))
        self.assertEqual(splits["validation"]["rating"].tolist(), [3])
        self.assertEqual(splits["validation"]["is_relevant"].tolist(), [0])
        self.assertEqual(splits["test"]["rating"].tolist(), [4])
        self.assertEqual(splits["test"]["is_relevant"].tolist(), [1])

    def test_validator_rejects_time_and_event_overlap(self):
        splits = chronological_split(self.frame)
        splits["test"] = pd.concat([splits["test"], splits["train"].iloc[:1]])
        with self.assertRaises(ValueError):
            validate_splits(splits, self.frame)

    def test_invalid_ratios_fail(self):
        for ratios in [(0.8, 0.1, 0.2), (1.0, 0.0, 0.0)]:
            with self.assertRaises(ValueError):
                chronological_split(self.frame, ratios)


class CleaningTests(unittest.TestCase):
    def test_invalid_rows_and_exact_duplicates_are_audited(self):
        raw = pd.DataFrame([
            (1, 1, 5, 100), (1, 1, 5, 100), (1, 2, 0, 200),
            (1, 3, 4, -1), (999, 2, 4, 300), (1, 999, 4, 300),
            (1, 2, None, 400),
        ], columns=["user_id", "movie_id", "rating", "timestamp"])
        tables = {"ratings": raw, "users": pd.DataFrame({"user_id": [1]}),
                  "movies": pd.DataFrame({"movie_id": [1, 2, 3]})}
        cleaned, audit = clean_ratings(tables)
        self.assertEqual(cleaned["event_id"].tolist(), [0])
        self.assertEqual(int(audit["removed_rows"].sum()), 6)
        self.assertEqual(str(cleaned["rated_at"].dt.tz), "UTC")
        self.assertEqual(len(raw), 7)

    def test_conflicting_repeat_ratings_require_a_policy(self):
        tables = {"ratings": pd.DataFrame([(1, 1, 5, 100), (1, 1, 2, 200)],
                  columns=["user_id", "movie_id", "rating", "timestamp"]),
                  "users": pd.DataFrame({"user_id": [1]}),
                  "movies": pd.DataFrame({"movie_id": [1]})}
        with self.assertRaises(ValueError):
            clean_ratings(tables)


class FeatureIsolationTests(unittest.TestCase):
    def setUp(self):
        frame = events([
            (1, 1, 5, 10), (1, 2, 2, 20), (2, 3, 4, 10), (2, 8, 5, 11),
            (1, 4, 5, 30), (1, 3, 1, 31),
            (1, 5, 5, 40), (1, 6, 5, 41), (1, 8, 5, 42),
        ])
        self.splits = add_relevance_labels({
            "train": frame.iloc[:4].copy(), "validation": frame.iloc[4:6].copy(),
            "test": frame.iloc[6:].copy(),
        })
        self.movies = pd.DataFrame({"movie_id": [1, 2, 3, 4, 5, 6, 8],
                                    "genres": ["Comedy", "Drama", "Comedy|Action",
                                               "FutureGenre", "FutureGenre", "Drama", "Action"]})
        self.inputs = build_model_inputs(self.splits["train"], self.movies)

    def test_mappings_features_and_sequences_only_use_train(self):
        self.assertEqual(self.inputs.movie_mapping["movie_id"].tolist(), [1, 2, 3, 8])
        self.assertNotIn("FutureGenre", self.inputs.genre_names)
        self.assertEqual(self.inputs.explicit_ratings.shape, (2, 4))
        self.assertEqual(self.inputs.explicit_ratings.nnz, 4)
        self.assertEqual(self.inputs.positive_interactions.nnz, 3)
        self.assertEqual(self.inputs.explicit_ratings[0, 1], 2)
        self.assertEqual(self.inputs.positive_interactions[0, 1], 0)
        sequence = self.inputs.train_sequences.set_index("user_id").loc[1]
        self.assertEqual(sequence["movie_tokens"], [1])
        self.assertEqual(sequence["event_ids"], [0])
        self.assertNotIn(0, sequence["movie_tokens"])

    def test_test_context_includes_validation_but_never_test(self):
        context = build_evaluation_context(self.splits, self.inputs, "test", k=1).iloc[0]
        self.assertEqual(context["seen_movie_ids"], [1, 2, 3, 4])
        self.assertEqual(context["positive_movie_tokens"], [1])
        self.assertEqual(context["relevant_movie_ids"], [8])
        self.assertEqual(context["cold_relevant_count"], 2)
        self.assertTrue(context["eligible"])
        self.assertEqual(candidate_movie_ids(self.inputs, context["seen_movie_ids"]), [8])

    def test_validation_context_masks_low_ratings_and_counts_cold_targets(self):
        frame = build_evaluation_context(self.splits, self.inputs, "validation", k=1)
        context = frame.iloc[0]
        self.assertEqual(context["seen_movie_ids"], [1, 2])
        self.assertEqual(candidate_movie_ids(self.inputs, context["seen_movie_ids"]), [3, 8])
        self.assertFalse(context["eligible"])
        self.assertEqual(context["exclusion_reason"], "no_warm_relevant_items")
        self.assertEqual(evaluation_summary({"validation": frame}).loc["validation", "positive_pairs_with_cold_movie"], 1)

    def test_artifacts_round_trip_without_dense_matrices(self):
        contexts = {"test": build_evaluation_context(self.splits, self.inputs, "test", k=1)}
        with TemporaryDirectory() as folder:
            save_preprocessed_artifacts(self.splits, self.inputs, contexts, folder, {"threshold": 4})
            directory = Path(folder)
            restored = load_npz(directory / "positive_interactions.npz")
            self.assertEqual((restored != self.inputs.positive_interactions).nnz, 0)
            self.assertEqual(json.loads((directory / "config.json").read_text())["threshold"], 4)
            saved = pd.read_json(directory / "test_contexts.jsonl", lines=True)
            self.assertEqual(saved.iloc[0]["relevant_movie_ids"], [8])

    def test_empty_holdout_has_an_empty_reportable_cohort(self):
        splits = dict(self.splits)
        splits["validation"] = splits["validation"].iloc[:0]
        context = build_evaluation_context(splits, self.inputs, "validation")
        self.assertTrue(context.empty)
        self.assertEqual(evaluation_summary({"validation": context}).loc["validation", "eligible_users"], 0)

    def test_no_positive_training_history_is_reported(self):
        splits = dict(self.splits)
        splits["train"] = splits["train"].copy()
        splits["train"].loc[splits["train"]["user_id"].eq(1), "is_relevant"] = 0
        inputs = build_model_inputs(splits["train"], self.movies)
        context = build_evaluation_context(splits, inputs, "test", k=1).iloc[0]
        self.assertFalse(context["eligible"])
        self.assertEqual(context["exclusion_reason"], "no_train_positive_history")


if __name__ == "__main__":
    unittest.main()
