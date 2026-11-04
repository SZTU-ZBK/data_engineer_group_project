"""Check loss semantics, stable logits, and batching independently of training."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.lightfm_diagnostics import observed_logistic_loss
from src.models.lightfm import LightFMRecommender


class LogisticLossTests(unittest.TestCase):
    def model(self, scores):
        return LightFMRecommender([1], [[0]], [0], np.zeros((len(scores), 1)), scores)

    def test_known_binary_cross_entropy_and_unknown_exclusion(self):
        model = self.model([0, 2, -2, 100])
        labels = csr_matrix([[1, 1, -1, 0]], dtype="float32")
        result = observed_logistic_loss(model, labels, batch_size=2)
        expected = (np.log(2) + 2 * np.log1p(np.exp(-2))) / 3
        self.assertAlmostEqual(result["loss"], expected)
        self.assertEqual(result["positive_count"], 2)
        self.assertEqual(result["negative_count"], 1)
        self.assertAlmostEqual(result["loss"], observed_logistic_loss(model, labels, batch_size=1)["loss"])

    def test_saturated_logits_are_finite_and_invalid_labels_are_rejected(self):
        model = self.model([1000, -1000, -1000])
        result = observed_logistic_loss(model, csr_matrix([[1, -1, 1]], dtype="float32"))
        self.assertAlmostEqual(result["loss"], 1000 / 3)
        for labels in (csr_matrix([[0, 0, 0]]), csr_matrix([[5, 0, 0]]), csr_matrix([[1, -1]])):
            with self.assertRaises(ValueError):
                observed_logistic_loss(model, labels)


if __name__ == "__main__":
    unittest.main()
