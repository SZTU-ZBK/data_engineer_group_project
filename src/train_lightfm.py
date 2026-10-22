"""Train official LightFM; select features and checkpoints on validation."""

import itertools
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import load_npz

from src.evaluate import evaluate_ranking, write_comparison
from src.models.lightfm import (LightFMRecommender, build_item_features,
                               build_logistic_interactions, check_training_backend)
from src.recommend import file_sha256, load_movie_metadata, recommendation_table


def run_lightfm(processed_dir, output_dir, components, epochs, alphas, seed=42, loss="logistic"):
    check_training_backend(loss)
    import lightfm
    from lightfm import LightFM

    components, epochs, alphas = sorted(set(components)), sorted(set(epochs)), sorted(set(alphas))
    if not components or not epochs or not alphas or min(components + epochs) < 1 or min(alphas) < 0:
        raise ValueError("Use positive dimensions/epochs and nonnegative regularization.")
    processed_dir, output_dir = processed_dir.resolve(), output_dir.resolve()
    required = ["config.json", "user_mapping.csv", "movie_mapping.csv", "positive_interactions.npz",
                "genre_features.npz", "validation_contexts.jsonl", "test_contexts.jsonl"]
    if loss == "logistic":
        required.append("explicit_ratings.npz")
    missing = [name for name in required if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run 01_data_analysis.ipynb first: {missing}")
    checksums = {name: file_sha256(processed_dir / name) for name in required}
    config = json.loads((processed_dir / "config.json").read_text(encoding="utf-8"))
    interactions = load_npz(processed_dir / "positive_interactions.npz").tocsr()
    users = pd.read_csv(processed_dir / "user_mapping.csv").sort_values("user_index")
    movies = pd.read_csv(processed_dir / "movie_mapping.csv").sort_values("movie_index")
    if (users.user_index.tolist() != list(range(len(users)))
            or movies.movie_index.tolist() != list(range(len(movies)))
            or interactions.shape != (len(users), len(movies))
            or not np.all(np.diff(movies.movie_id) > 0)
            or not np.all(interactions.data == 1) or config["evaluation_k"] < 10):
        raise ValueError("Expected aligned binary training inputs and the shared Top-10 cohort.")
    if loss == "logistic":
        signed = build_logistic_interactions(load_npz(processed_dir / "explicit_ratings.npz"),
                                            config["relevance_threshold"])
        if signed.shape != interactions.shape or (signed.maximum(0) != interactions).nnz:
            raise ValueError("Explicit training scores and positive interactions do not align.")
        interactions = signed
    print(f'LightFM {loss}: {int((interactions.data > 0).sum()):,} positives, '
          f'{int((interactions.data < 0).sum()):,} negatives; training data only.', flush=True)
    genres = load_npz(processed_dir / "genre_features.npz")
    movie_ids, user_ids = movies.movie_id.to_numpy(), users.user_id.to_numpy()
    validation = pd.read_json(processed_dir / "validation_contexts.jsonl", lines=True)
    table_dir, model_dir = output_dir / "tables", output_dir / "models" / "lightfm"
    table_dir.mkdir(parents=True, exist_ok=True)
    # Verify a saved baseline uses identical inputs before creating a comparison.
    baseline_path = table_dir / "itemcf_run.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    if baseline and any(checksums[name] != digest for name, digest in baseline["input_sha256"].items()):
        raise ValueError("ItemCF used different inputs. Rerun it before comparing models.")

    trials, best, best_key = [], None, None
    started = perf_counter()
    for mode, dimensions, alpha in itertools.product(["identity", "identity+genres"], components, alphas):
        features = build_item_features(len(movies), genres if mode == "identity+genres" else None)
        engine = LightFM(no_components=dimensions, loss=loss, learning_rate=0.05,
                         user_alpha=alpha, item_alpha=alpha, random_state=seed, max_sampled=10)
        trained, fit_seconds = 0, 0.0
        for checkpoint in epochs:
            print(f'Training {mode}, dim={dimensions}, alpha={alpha:g}, epochs={trained}->{checkpoint}', flush=True)
            start = perf_counter()
            engine.fit_partial(interactions, item_features=features, epochs=checkpoint - trained, num_threads=1)
            fit_seconds += perf_counter() - start
            trained = checkpoint
            model = LightFMRecommender.from_fitted(engine, features, user_ids)
            metrics, _ = evaluate_ranking(model.score_tokens, validation, movie_ids)
            trial = {"loss": loss, "item_features": mode, "no_components": dimensions, "alpha": alpha,
                     "epochs": checkpoint, "fit_seconds": fit_seconds, **metrics}
            trials.append(trial)
            pd.DataFrame(trials).to_csv(table_dir / "lightfm_search_progress.csv", index=False)
            # Smaller models/earlier checkpoints break equal validation scores.
            key = (metrics["NDCG@10"], metrics["Recall@10"], -dimensions, -checkpoint, -alpha, mode == "identity")
            if best_key is None or key > best_key:
                best, best_key = (model, trial.copy()), key
                model.save(model_dir / "search_best")
                (model_dir / "search_best" / "validation.json").write_text(
                    json.dumps(trial, indent=2), encoding="utf-8")
            print(f'{mode}, dim={dimensions}, alpha={alpha:g}, epoch={checkpoint}: '
                  f'NDCG@10={metrics["NDCG@10"]:.6f}, Recall@10={metrics["Recall@10"]:.6f}', flush=True)

    model, selected = best
    # Test labels are first loaded after the winner has been fixed.
    test = pd.read_json(processed_dir / "test_contexts.jsonl", lines=True)
    test_metrics, per_user = evaluate_ranking(model.score_tokens, test, movie_ids)
    model.save(model_dir)
    per_user.to_csv(model_dir / "test_per_user.csv", index=False)
    search = pd.DataFrame(trials)
    parameters = {name: selected[name] for name in ["loss", "item_features", "no_components", "alpha", "epochs"]}
    search["selected"] = np.logical_and.reduce([search[name].eq(value) for name, value in parameters.items()])
    search.to_csv(table_dir / "lightfm_validation_search.csv", index=False)
    validation_metrics = {name: selected[name] for name in test_metrics}
    results = pd.DataFrame([
        {"algorithm": "LightFM", "loss": loss, "split": "validation", **validation_metrics},
        {"algorithm": "LightFM", "loss": loss, "split": "test", **test_metrics},
    ])
    results.to_csv(table_dir / "lightfm_metrics.csv", index=False)
    examples = []
    metadata = load_movie_metadata()
    for context in test.loc[test.eligible].sort_values("user_id").head(3).to_dict("records"):
        example = recommendation_table(model, context, movie_ids, metadata, k=10)
        example.insert(0, "user_id", int(context["user_id"]))
        examples.append(example)
    pd.concat(examples, ignore_index=True).to_csv(table_dir / "lightfm_recommendation_examples.csv", index=False)
    run = {
        "algorithm": "LightFM", "loss": loss, "selected_parameters": parameters,
        "grid": {"no_components": components, "alpha": alphas, "epochs": epochs,
                 "item_features": ["identity", "identity+genres"]},
        "selection_metric": "NDCG@10 on validation",
        "tie_break": "Recall@10, smaller dimensions, earlier epochs, smaller alpha, identity features",
        "random_seed": seed, "num_threads": 1, "learning_rate": 0.05,
        "max_sampled": 10 if loss == "warp" else None,
        "feature_encoding": "User identity; item identity optionally concatenated with genres, row L1 normalized",
        "training_feedback": ("Train rating >= relevance_threshold: +1; rating 1/2: -1; rating 3 and unknown omitted"
                              if loss == "logistic" else "Training ratings >= relevance_threshold only"),
        "training_positive_count": int((interactions.data > 0).sum()),
        "training_negative_count": int((interactions.data < 0).sum()),
        "negative_sampling": ("None: only observed training ratings; no validation/test labels consulted"
                              if loss == "logistic" else
                              "Training catalog only; no validation/test labels consulted; unobserved is not confirmed dislike"),
        "model_fit_partitions": ["train"], "refit_on_validation": False,
        "validation_history": config["validation_history"], "test_history": config["test_history"],
        "history_use": "Fixed training user vectors; stage history filters candidates, no online vector update",
        "candidate_protocol": "Shared full unseen training catalog and eligible users; no sampled evaluation negatives",
        "k_values": [5, 10], "metric_aggregation": "macro-average over eligible users; warm relevant targets",
        "early_stopping": "Choose among requested epoch checkpoints by validation NDCG@10",
        "input_sha256": checksums, "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "lightfm": lightfm.__version__},
        "windows_rng_repair": (json.loads((Path(lightfm.__file__).parent / "windows_rng_repair.json")
                                         .read_text(encoding="utf-8"))
                               if (Path(lightfm.__file__).parent / "windows_rng_repair.json").is_file() else None),
        "total_seconds": perf_counter() - started,
    }
    payload = json.dumps(run, indent=2)
    (model_dir / "run.json").write_text(payload, encoding="utf-8")
    (table_dir / "lightfm_run.json").write_text(payload, encoding="utf-8")
    write_comparison(table_dir)
    print(results.to_string(index=False), flush=True)
