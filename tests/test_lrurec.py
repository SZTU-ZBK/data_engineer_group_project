"""Check next-item supervision, recurrence, causality and shared evaluation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_ranking, write_comparison
from src.sequences import pad_history, training_windows, validation_windows

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from src.models.lrurec import LRURec, parallel_recurrence
    from src.train_lrurec import evaluate_batched, evaluate_loss, next_item_loss


class SequenceTests(unittest.TestCase):
    def test_validation_labels_exclude_train_and_future_history(self):
        train = pd.DataFrame({"user_id": [1], "movie_tokens": [[1, 2, 3]], "timestamps": [[1, 2, 3]]})
        heldout = pd.DataFrame({"user_id": [1], "movie_tokens": [[4, 5, 6]], "timestamps": [[4, 4, 5]]})
        inputs, targets = validation_windows(train, heldout, 4, 6)
        np.testing.assert_array_equal(inputs, [[1, 2, 3, 4], [2, 3, 4, 5]])
        np.testing.assert_array_equal(targets, [[0, 0, 4, 0], [0, 0, 0, 6]])
        self.assertEqual(train.movie_tokens.iloc[0], [1, 2, 3])
        for x, y in zip(inputs, targets):
            for position in np.flatnonzero(y):
                self.assertNotIn(y[position], x[:position + 1])

    def test_windows_cover_train_transitions_and_ignore_timestamp_ties(self):
        sequences = pd.DataFrame({"movie_tokens": [[1, 2, 3, 4, 5, 6], [7]],
                                  "timestamps": [[1, 1, 2, 3, 3, 4], [1]]})
        original = sequences.copy(deep=True)
        inputs, targets = training_windows(sequences, 3, 7)
        np.testing.assert_array_equal(inputs, [[1, 2, 3], [0, 4, 5]])
        np.testing.assert_array_equal(targets, [[0, 3, 4], [0, 0, 6]])
        pd.testing.assert_frame_equal(sequences, original)

    def test_padding_truncation_and_invalid_history(self):
        np.testing.assert_array_equal(pad_history([1, 2, 3, 4], 3, 4), [2, 3, 4])
        np.testing.assert_array_equal(pad_history([], 3, 4), [0, 0, 0])
        with self.assertRaises(ValueError):
            pad_history([0, 2], 3, 4)
        with self.assertRaises(ValueError):
            training_windows(pd.DataFrame({"movie_tokens": [[1, 2]], "timestamps": [[2, 1]]}), 3, 4)

    def test_comparison_preserves_three_models_and_rejects_stale_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("itemcf", "lightfm", "lrurec"):
                run = {"input_sha256": {"config.json": "same"},
                       "validation_metrics": {"users": 2, "NDCG@10": 0.1},
                       "test_metrics": {"users": 3, "NDCG@10": 0.2}}
                (directory / f"{name}_run.json").write_text(json.dumps(run), encoding="utf-8")
            write_comparison(directory)
            comparison = pd.read_csv(directory / "comparison_metrics.csv")
            self.assertEqual(len(comparison), 6)
            self.assertEqual(set(comparison.algorithm), {"ItemCF", "LightFM", "LRURec"})
            run["input_sha256"]["config.json"] = "changed"
            (directory / "lrurec_run.json").write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaises(ValueError):
                write_comparison(directory)


@unittest.skipUnless(HAS_TORCH, "Install CPU PyTorch to run LRURec model tests")
class LRURecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_parallel_scan_matches_sequential_values_and_gradients(self):
        for length in (1, 3, 8):
            updates = torch.randn(2, length, 4, dtype=torch.complex64, requires_grad=True)
            eigenvalues = (0.8 * torch.rand(4, dtype=torch.complex64)).requires_grad_()
            expected, h = [], torch.zeros_like(updates[:, 0])
            for t in range(length):
                h = eigenvalues * h + updates[:, t]
                expected.append(h)
            expected = torch.stack(expected, dim=1)
            actual = parallel_recurrence(updates, eigenvalues)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            actual_grad = torch.autograd.grad(actual.abs().square().sum(), (updates, eigenvalues),
                                              retain_graph=True, allow_unused=True)
            expected_grad = torch.autograd.grad(expected.abs().square().sum(), (updates, eigenvalues),
                                                allow_unused=True)
            for a, b in zip(actual_grad, expected_grad):
                # With one step lambda is unused; autograd may report None or zero.
                if a is None:
                    a = torch.zeros_like(eigenvalues)
                if b is None:
                    b = torch.zeros_like(eigenvalues)
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_causality_and_padding_invariance(self):
        model = LRURec(8, hidden_size=4, max_length=6, dropout=0).eval()
        a = model(torch.tensor([[1, 2, 3, 4]]))
        b = model(torch.tensor([[1, 2, 7, 8]]))
        torch.testing.assert_close(a[:, :2], b[:, :2])
        padded = model(torch.tensor([[0, 0, 1, 2, 3, 4]]))
        torch.testing.assert_close(a, padded[:, -4:], atol=1e-6, rtol=1e-5)

    def test_validation_loss_averages_targets_without_updating_parameters(self):
        model = LRURec(6, hidden_size=4, num_blocks=1, max_length=3, dropout=0)
        inputs = torch.tensor([[1, 2, 3], [2, 3, 4], [1, 3, 5]])
        targets = torch.tensor([[0, 3, 4], [0, 0, 5], [0, 5, 6]])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, targets), batch_size=2)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        actual = evaluate_loss(model, loader, "cpu")
        expected = next_item_loss(model, inputs, targets).item()
        self.assertAlmostEqual(actual, expected, places=5)
        for name, p in model.named_parameters():
            torch.testing.assert_close(before[name], p)
            self.assertIsNone(p.grad)

    def test_training_loss_falls_and_ignored_labels_have_no_gradient(self):
        model = LRURec(6, hidden_size=8, num_blocks=1, max_length=3, dropout=0)
        inputs = torch.tensor([[1, 2, 3], [1, 2, 3]])
        targets = torch.tensor([[0, 3, 4], [0, 3, 4]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = next_item_loss(model, inputs, targets).item()
        for _ in range(20):
            optimizer.zero_grad()
            loss = next_item_loss(model, inputs, targets)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertLess(next_item_loss(model, inputs, targets).item(), initial * 0.5)
        logits = model(inputs)
        expected = torch.nn.functional.cross_entropy(logits[:, 1:].reshape(-1, 6),
                                                      (targets[:, 1:] - 1).reshape(-1))
        torch.testing.assert_close(next_item_loss(model, inputs, targets), expected)

    def test_batch_scores_and_save_load(self):
        model = LRURec(8, hidden_size=4, max_length=3, dropout=0).eval()
        histories = [[1, 2], [4, 5, 6, 7]]
        batched = model.score_histories(histories)
        for i, history in enumerate(histories):
            np.testing.assert_allclose(batched[i], model.score_tokens(99, history), atol=1e-6)
        with tempfile.TemporaryDirectory() as temporary:
            model.save(temporary)
            restored = LRURec.load(temporary)
            np.testing.assert_array_equal(batched, restored.score_histories(histories))

    def test_batch_evaluation_matches_shared_evaluator_and_hides_targets(self):
        model = LRURec(12, hidden_size=4, max_length=3, dropout=0).eval()
        contexts = pd.DataFrame([
            {"user_id": 1, "positive_movie_tokens": [1], "seen_movie_ids": [1],
             "relevant_movie_ids": [4, 5], "eligible": True, "candidate_count": 11},
            {"user_id": 2, "positive_movie_tokens": [2], "seen_movie_ids": [2],
             "relevant_movie_ids": [6], "eligible": True, "candidate_count": 11},
        ])
        ids = np.arange(1, 13)
        a, users_a = evaluate_batched(model, contexts, ids, 2)
        b, users_b = evaluate_ranking(model.score_tokens, contexts, ids)
        self.assertEqual(a, b)
        pd.testing.assert_frame_equal(users_a, users_b)


if __name__ == "__main__":
    unittest.main()
