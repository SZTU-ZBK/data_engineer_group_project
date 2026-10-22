"""Train LRURec on train sequences and choose the checkpoint on validation only."""

import json
import platform
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate_ranking, write_comparison
from src.models.lrurec import LRURec
from src.recommend import file_sha256, load_movie_metadata, recommendation_table
from src.sequences import training_windows


def next_item_loss(model, inputs, targets):
    valid = targets.ne(0)
    hidden = model.encode(inputs)[valid]
    logits = hidden @ model.embedding.weight[1:].T + model.item_bias
    return nn.functional.cross_entropy(logits, targets[valid] - 1)


def evaluate_batched(model, contexts, movie_ids, batch_size):
    """Encode known histories in batches, then use the shared ranking evaluator."""
    cohort = contexts.loc[contexts.eligible, ["user_id", "positive_movie_tokens"]]
    scores = {}
    for start in range(0, len(cohort), batch_size):
        batch = cohort.iloc[start:start + batch_size]
        predictions = model.score_histories(batch.positive_movie_tokens.tolist())
        scores.update(zip(batch.user_id.astype(int), predictions))
    return evaluate_ranking(lambda user_id, history: scores[user_id], contexts, movie_ids)


def run_lrurec(processed_dir, output_dir, epochs=(1, 3), hidden_size=32, max_length=50,
               batch_size=64, num_blocks=2, learning_rate=0.001, weight_decay=0.01,
               dropout=0.2, seed=42, device="cpu", num_threads=2):
    epochs = sorted(set(epochs))
    if (not epochs or min(epochs) < 1 or min(batch_size, num_threads) < 1
            or learning_rate <= 0 or weight_decay < 0):
        raise ValueError("Use positive epochs/batch size/threads/rate and nonnegative weight decay.")
    processed_dir, output_dir = processed_dir.resolve(), output_dir.resolve()
    required = ["config.json", "movie_mapping.csv", "positive_interactions.npz",
                "train_sequences.jsonl", "validation_contexts.jsonl", "test_contexts.jsonl"]
    checksums = {name: file_sha256(processed_dir / name) for name in required}
    config = json.loads((processed_dir / "config.json").read_text(encoding="utf-8"))
    mapping = pd.read_csv(processed_dir / "movie_mapping.csv").sort_values("movie_index")
    movie_ids = mapping.movie_id.to_numpy()
    if (mapping.movie_index.tolist() != list(range(len(mapping)))
            or not np.all(np.diff(movie_ids) > 0) or config["evaluation_k"] < 10):
        raise ValueError("Expected the shared training catalog and Top-10 evaluation cohort.")
    table_dir, model_dir = output_dir / "tables", output_dir / "models" / "lrurec"
    # Detect incompatible baselines before doing expensive training.
    for name in ("itemcf", "lightfm"):
        path = table_dir / f"{name}_run.json"
        if path.is_file():
            other = json.loads(path.read_text(encoding="utf-8"))["input_sha256"]
            if any(checksums[key] != other[key] for key in checksums.keys() & other.keys()):
                raise ValueError(f"{name} used different inputs. Rerun it before comparing models.")
    sequences = pd.read_json(processed_dir / "train_sequences.jsonl", lines=True)
    inputs, targets = training_windows(sequences, max_length, len(mapping))
    validation = pd.read_json(processed_dir / "validation_contexts.jsonl", lines=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    model = LRURec(len(mapping), hidden_size, num_blocks, max_length, dropout).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
                        batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    table_dir.mkdir(parents=True, exist_ok=True)
    started, trials, best_key = perf_counter(), [], None
    print(f"LRURec: {len(inputs):,} train windows, {int((targets > 0).sum()):,} targets; "
          f"device={device}, threads={num_threads}.", flush=True)
    for epoch in range(1, max(epochs) + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = next_item_loss(model, x, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; no checkpoint was accepted.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = int(y.ne(0).sum())
            total_loss += loss.item() * n
            count += n
        print(f"Epoch {epoch}: train loss={total_loss / count:.4f}", flush=True)
        if epoch not in epochs:
            continue
        metrics, _ = evaluate_batched(model, validation, movie_ids, batch_size)
        trial = {"epochs": epoch, "train_loss": total_loss / count,
                 "elapsed_seconds": perf_counter() - started, **metrics}
        trials.append(trial)
        key = (metrics["NDCG@10"], metrics["Recall@10"], -epoch)
        if best_key is None or key > best_key:
            best_key, selected = key, trial.copy()
            model.save(model_dir / "search_best")
            (model_dir / "search_best" / "validation.json").write_text(
                json.dumps(selected, indent=2), encoding="utf-8")
        pd.DataFrame(trials).to_csv(table_dir / "lrurec_validation_search.csv", index=False)
        print(f'Validation: NDCG@10={metrics["NDCG@10"]:.6f}, '
              f'Precision@10={metrics["Precision@10"]:.6f}', flush=True)

    # Fix the winner before loading test labels. Never refit on validation.
    model = LRURec.load(model_dir / "search_best", device=device)
    test = pd.read_json(processed_dir / "test_contexts.jsonl", lines=True)
    test_metrics, per_user = evaluate_batched(model, test, movie_ids, batch_size)
    validation_metrics = {name: selected[name] for name in test_metrics}
    model.save(model_dir)
    per_user.to_csv(model_dir / "test_per_user.csv", index=False)
    search = pd.DataFrame(trials)
    search["selected"] = search.epochs.eq(selected["epochs"])
    search.to_csv(table_dir / "lrurec_validation_search.csv", index=False)
    results = pd.DataFrame([{"algorithm": "LRURec", "split": stage, **metrics}
                            for stage, metrics in [("validation", validation_metrics), ("test", test_metrics)]])
    results.to_csv(table_dir / "lrurec_metrics.csv", index=False)
    examples, movies = [], load_movie_metadata()
    for context in test.loc[test.eligible].sort_values("user_id").head(3).to_dict("records"):
        example = recommendation_table(model, context, movie_ids, movies, k=10)
        example.insert(0, "user_id", int(context["user_id"]))
        examples.append(example)
    pd.concat(examples, ignore_index=True).to_csv(table_dir / "lrurec_recommendation_examples.csv", index=False)
    run = {
        "algorithm": "LRURec", "reference": "https://arxiv.org/abs/2310.02367",
        "selected_parameters": {**model.config, "epochs": int(selected["epochs"])},
        "epoch_checkpoints": epochs, "selection_metric": "NDCG@10 on validation",
        "tie_break": "Recall@10, then earlier epoch", "random_seed": seed,
        "device": device, "num_threads": num_threads, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "optimizer": "AdamW", "gradient_clip_norm": 5.0,
        "loss": "Full-catalog next-item cross entropy; padding excluded from classifier",
        "training_windows": len(inputs), "training_targets": int((targets > 0).sum()),
        "window_policy": "Nonoverlapping train input chunks; each adjacent next-item transition once",
        "timestamp_ties": "Equal-timestamp adjacent targets ignored; history tie order is deterministic movie ID",
        "model_fit_partitions": ["train"], "refit_on_validation": False,
        "validation_history": config["validation_history"], "test_history": config["test_history"],
        "history_use": f"Last {max_length} known positive movie tokens; recompute sequence state at inference",
        "candidate_protocol": "Shared full unseen training catalog and eligible users; no sampled evaluation negatives",
        "k_values": [5, 10], "metric_aggregation": "macro-average over eligible users; warm relevant targets",
        "implementation": "Complex stable diagonal LRU, 2x state width, residual FFN, tied embeddings; prefix doubling scan",
        "paper_differences": "Shared temporal 80/10/10 split and positive-only sequences; timestamp-tie masking; "
                             "compact model/training schedule; project validation selection; not a paper-number replication",
        "input_sha256": checksums, "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
        "total_seconds": perf_counter() - started,
    }
    payload = json.dumps(run, indent=2)
    (model_dir / "run.json").write_text(payload, encoding="utf-8")
    (table_dir / "lrurec_run.json").write_text(payload, encoding="utf-8")
    write_comparison(table_dir)
    print(results.to_string(index=False), flush=True)
