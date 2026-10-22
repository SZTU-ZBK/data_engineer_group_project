"""Tune ItemCF on validation, then evaluate the selected configuration on test."""

import argparse
import itertools
import json
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import load_npz

from src.data import PROJECT_ROOT
from src.evaluate import evaluate_ranking
from src.models.item_cf import ItemCF
from src.recommend import file_sha256, load_movie_metadata, recommendation_table


def run_itemcf(processed_dir, output_dir, neighbors, shrinkages):
    processed_dir, output_dir = processed_dir.resolve(), output_dir.resolve()
    required = ["config.json", "movie_mapping.csv", "positive_interactions.npz",
                "validation_contexts.jsonl", "test_contexts.jsonl"]
    missing = [name for name in required if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run 01_data_analysis.ipynb first; missing preprocessed inputs: {missing}")
    preprocessing = json.loads((processed_dir / "config.json").read_text(encoding="utf-8"))
    if preprocessing["evaluation_k"] < 10:
        raise ValueError("The shared cohort must support at least 10 candidates for Recall/NDCG@10.")
    interactions = load_npz(processed_dir / "positive_interactions.npz").tocsr()
    mapping = pd.read_csv(processed_dir / "movie_mapping.csv").sort_values("movie_index")
    movie_ids = mapping["movie_id"].to_numpy(dtype="int64")
    if (mapping["movie_index"].tolist() != list(range(len(mapping)))
            or len(movie_ids) != interactions.shape[1]
            or not np.all(np.diff(movie_ids) > 0)):
        raise ValueError("Expected contiguous indexes and ascending raw movie IDs matching matrix columns.")
    validation = pd.read_json(processed_dir / "validation_contexts.jsonl", lines=True)
    grid = sorted(set(itertools.product(neighbors, shrinkages)))
    if not grid:
        raise ValueError("The hyperparameter grid must not be empty.")
    trials, best_model, best_validation, best_key = [], None, None, None
    start = perf_counter()
    for n_neighbors, shrinkage in grid:
        model = ItemCF(n_neighbors=n_neighbors, shrinkage=shrinkage).fit(interactions)
        evaluation_start = perf_counter()
        metrics, _ = evaluate_ranking(model.score_tokens, validation, movie_ids)
        trial = {"n_neighbors": n_neighbors, "shrinkage": shrinkage,
                 "fit_seconds": model.fit_seconds_, "validation_seconds": perf_counter() - evaluation_start,
                 **metrics}
        trials.append(trial)
        key = (metrics["NDCG@10"], metrics["Recall@10"], -n_neighbors, -shrinkage)
        if best_key is None or key > best_key:
            best_model, best_validation, best_key = model, dict(metrics), key
        print(f'Validation neighbors={n_neighbors}, shrinkage={shrinkage:g}: '
              f'NDCG@10={metrics["NDCG@10"]:.6f}, Recall@10={metrics["Recall@10"]:.6f}', flush=True)

    # Only load test labels after the validation-only selection is complete.
    print(f'Selected neighbors={best_model.n_neighbors}, shrinkage={best_model.shrinkage:g}', flush=True)
    test = pd.read_json(processed_dir / "test_contexts.jsonl", lines=True)
    test_start = perf_counter()
    test_metrics, test_per_user = evaluate_ranking(best_model.score_tokens, test, movie_ids)
    test_seconds = perf_counter() - test_start

    table_dir, model_dir = output_dir / "tables", output_dir / "models" / "itemcf"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model.save(model_dir)
    test_per_user.to_csv(model_dir / "test_per_user.csv", index=False)
    search = pd.DataFrame(trials)
    search["selected"] = (search["n_neighbors"].eq(best_model.n_neighbors)
                           & search["shrinkage"].eq(best_model.shrinkage))
    search.to_csv(table_dir / "itemcf_validation_search.csv", index=False)
    results = pd.DataFrame([
        {"algorithm": "ItemCF", "split": "validation", **best_validation},
        {"algorithm": "ItemCF", "split": "test", **test_metrics},
    ])
    results.to_csv(table_dir / "itemcf_metrics.csv", index=False)

    # Predetermined ID order avoids choosing successful examples by their results.
    movies = load_movie_metadata()
    examples = []
    for context in test.loc[test["eligible"]].sort_values("user_id").head(3).to_dict("records"):
        example = recommendation_table(best_model, context, movie_ids, movies, k=10)
        example.insert(0, "user_id", int(context["user_id"]))
        example["relevant_in_test"] = example["movie_id"].isin(context["relevant_movie_ids"])
        examples.append(example)
    pd.concat(examples, ignore_index=True).to_csv(table_dir / "itemcf_recommendation_examples.csv", index=False,
                                                encoding="utf-8-sig")
    run = {
        "algorithm": "positive_feedback_itemcf", "input": "binary rating >= relevance_threshold",
        "selection_metric": "NDCG@10 on validation", "tie_break": "Recall@10, then smaller neighbors/shrinkage",
        "selected_parameters": {"n_neighbors": best_model.n_neighbors, "shrinkage": best_model.shrinkage},
        "grid": {"n_neighbors": sorted(set(neighbors)), "shrinkage": sorted(set(shrinkages))},
        "similarity": "cosine * cooccurrence/(cooccurrence+shrinkage); diagonal=0; top neighbors per source item",
        "scoring": "sum of retained similarities from known positive history; scores are not star predictions",
        "score_tie_break": "ascending raw movie ID; zero-support candidates remain eligible with score zero",
        "candidate_protocol": "full unseen training catalog, shared preprocessing cohorts; no sampled negatives",
        "validation_history": preprocessing["validation_history"], "test_history": preprocessing["test_history"],
        "model_fit_partitions": ["train"], "refit_on_validation": False,
        "k_values": [5, 10], "metric_aggregation": "macro-average over eligible users; warm relevant targets",
        "catalog_coverage_denominator": len(movie_ids), "random_seed": None,
        "early_stopping": "not applicable: closed-form neighbor fitting",
        "fit_seconds_selected_trial": best_model.fit_seconds_, "test_evaluation_seconds": test_seconds,
        "total_seconds": perf_counter() - start,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "input_sha256": {name: file_sha256(processed_dir / name) for name in required},
        "example_user_selection": "first three eligible test users in ascending user ID",
        "validation_metrics": best_validation, "test_metrics": test_metrics,
    }
    payload = json.dumps(run, ensure_ascii=False, indent=2)
    (model_dir / "run.json").write_text(payload, encoding="utf-8")
    (table_dir / "itemcf_run.json").write_text(payload, encoding="utf-8")
    print(results.to_string(index=False), flush=True)
    print(f'Results: {table_dir}\nSaved model: {model_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["itemcf"], default="itemcf")
    parser.add_argument("--neighbors", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--shrinkage", nargs="+", type=float, default=[0.0, 10.0, 50.0])
    parser.add_argument("--processed-dir", type=type(PROJECT_ROOT), default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=type(PROJECT_ROOT), default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    run_itemcf(args.processed_dir, args.output_dir, args.neighbors, args.shrinkage)


if __name__ == "__main__":
    main()
