"""Bounded first-round validation experiments; test only the frozen winner per model."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, load_npz

from src.evaluate import evaluate_ranking
from src.lightfm_features import (ContentFeatures, add_identity_features, build_enriched_features,
                                 normalize_rows, split_feature_history)
from src.models.item_cf import ItemCF
from src.models.lightfm import LightFMRecommender, build_logistic_interactions
from src.recommend import load_movie_metadata, load_user_metadata


METRICS = [f"{metric}@{k}" for k in (5, 10) for metric in ("Precision", "Recall", "NDCG")]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decay_histories(history, movie_map, half_life_days):
    """Weights depend only on the given stage's known history, never holdout events."""
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("Half-life must be finite and positive.")
    anchors = history.groupby("user_id").timestamp.max()
    positive = history.loc[history.is_relevant.eq(1) & history.movie_id.isin(movie_map)].copy()
    positive["index"] = positive.movie_id.map(movie_map)
    positive["age_days"] = (positive.user_id.map(anchors) - positive.timestamp) / 86400.0
    positive["weight"] = np.exp2(-positive.age_days / half_life_days)
    return {int(user): (group["index"].to_numpy(dtype=int), group.weight.to_numpy(dtype="float32"))
            for user, group in positive.groupby("user_id", sort=False)}


def decay_scorer(model, histories):
    def score(user, tokens):
        indexes, weights = histories[user]
        expected = np.unique(np.asarray(tokens, dtype=int))
        if not np.array_equal(np.sort(indexes + 1), expected[expected > 0]):
            raise ValueError("Time-decay history differs from the shared evaluation context.")
        return np.asarray(model.similarity_[indexes].T.dot(weights)).ravel()
    return score


def static_features(content):
    """Keep demographics, genres and release decades; drop all history-derived columns."""
    user_columns = [i for i, name in enumerate(content.user_names) if not name.startswith("preferred_")
                    and not name.startswith("preferred_genre:")]
    item_columns = [i for i, name in enumerate(content.item_names) if not name.startswith("history_")]
    return ContentFeatures(normalize_rows(content.users[:, user_columns]),
                           normalize_rows(content.items[:, item_columns]),
                           [content.user_names[i] for i in user_columns],
                           [content.item_names[i] for i in item_columns])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/round1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (out / "run.json").exists():
        raise SystemExit("Completed experiment already exists. Use a new --output-dir and a matching baseline.")
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    processed = ROOT / "data/processed"
    baselines = {name: json.loads((ROOT / f"outputs/tables/{name}_run.json").read_text(encoding="utf-8"))
                 for name in ("itemcf", "lightfm", "lrurec")}
    hashes = {}
    for name, run in baselines.items():
        for file, expected in run["input_sha256"].items():
            actual = hashes.setdefault(file, digest(processed / file))
            if actual != expected:
                raise ValueError(f"Stale baseline input: {name}/{file}")
        if digest(ROOT / f"outputs/tables/{name}_run.json") != digest(ROOT / f"outputs/models/{name}/run.json"):
            raise ValueError(f"Baseline model and table run records differ: {name}")
    initial_hashes = {f"outputs/tables/{name}_run.json": digest(ROOT / f"outputs/tables/{name}_run.json")
                      for name in baselines}
    for relative in ["outputs/models/lrurec/weights.pt", "outputs/models/lightfm/representations.npz",
                     "outputs/models/itemcf/similarity.npz"]:
        initial_hashes[relative] = digest(ROOT / relative)
    baseline = baselines["lightfm"]
    params = baseline["selected_parameters"]
    if (params["loss"] != "logistic" or params["no_components"] != 32 or params["alpha"] != 1e-6
            or baseline["random_seed"] != 42 or baseline["grid"]["epochs"] != [5, 10]):
        raise ValueError("Expected the saved 32-dim logistic 5/10-epoch baseline for controlled reuse.")
    movie_mapping = pd.read_csv(processed / "movie_mapping.csv").sort_values("movie_index")
    movie_ids = movie_mapping.movie_id.to_numpy()
    movie_map = dict(zip(movie_mapping.movie_id, movie_mapping.movie_index))
    train = pd.read_csv(processed / "train.csv")
    validation = pd.read_json(processed / "validation_contexts.jsonl", lines=True)
    rows, selected_models, configs, validation_winners = [], {}, {}, {}
    model_root = ROOT / "outputs/models/round1"
    model_root.mkdir(parents=True, exist_ok=True)

    def record(algorithm, variant, metrics, **kwargs):
        row = {"algorithm": algorithm, "variant": variant, "split": "validation", **kwargs,
               "users": metrics["users"], **{key: float(metrics[key]) for key in METRICS}}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "validation_search.csv", index=False)
        print(f'{algorithm} {variant} {kwargs}: NDCG@10={metrics["NDCG@10"]:.6f}, '
              f'P@10={metrics["Precision@10"]:.6f}, R@10={metrics["Recall@10"]:.6f}', flush=True)
        return row

    def key(row):
        return (row["NDCG@10"], row["Recall@10"])

    cf = ItemCF.load(ROOT / "outputs/models/itemcf")
    winner = record("ItemCF", "baseline", baselines["itemcf"]["validation_metrics"], half_life_days=0)
    best_half_life = None
    for days in (1, 7, 30, 90):
        metrics, _ = evaluate_ranking(decay_scorer(cf, decay_histories(train, movie_map, days)), validation, movie_ids)
        row = record("ItemCF", "time_decay", metrics, half_life_days=days)
        if key(row) > key(winner):
            winner, best_half_life = row, days
    validation_winners["ItemCF"] = winner
    configs["ItemCF"] = {"half_life_days": best_half_life, "n_neighbors": cf.n_neighbors,
                          "shrinkage": cf.shrinkage, "age_anchor": "latest known rating timestamp per user"}
    selected_models["ItemCF"] = cf
    (out / "itemcf_selection.json").write_text(json.dumps(configs["ItemCF"], indent=2), encoding="utf-8")

    import torch
    from src.models.lrurec import LRURec
    from src.train_lrurec import evaluate_batched
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    lru = LRURec.load(ROOT / "outputs/models/lrurec")
    if lru.max_length != 50:
        raise ValueError("This quick round assumes a baseline trained with sequence length 50.")
    winner = record("LRURec", "baseline", baselines["lrurec"]["validation_metrics"], inference_length=50)
    best_length = 50
    for length in (25, 100):
        lru.max_length = length
        metrics, _ = evaluate_batched(lru, validation, movie_ids, 64)
        row = record("LRURec", "fixed_weights_context", metrics, inference_length=length)
        if key(row) > key(winner):
            winner, best_length = row, length
    lru.max_length = best_length
    lru.config["max_length"] = best_length
    lru.save(model_root / "lrurec")
    validation_winners["LRURec"] = winner
    configs["LRURec"] = {"training_length": 50, "inference_length": best_length,
                          "weights": "unchanged baseline epoch 50", "retrained": False}
    selected_models["LRURec"] = lru

    from lightfm import LightFM
    from src.models.lightfm import build_user_features
    users = pd.read_csv(processed / "user_mapping.csv").sort_values("user_index")
    history, fit = split_feature_history(train)
    fit = fit.merge(users, on="user_id", validate="many_to_one").merge(movie_mapping, on="movie_id", validate="many_to_one")
    explicit = csr_matrix((fit.rating.to_numpy(dtype="float32"), (fit.user_index, fit.movie_index)),
                          shape=(len(users), len(movie_mapping)))
    labels = build_logistic_interactions(explicit, 4)
    if (int((labels.data > 0).sum()) != baseline["training_positive_count"]
            or int((labels.data < 0).sum()) != baseline["training_negative_count"]):
        raise ValueError("Ablation fitting labels differ from baseline.")
    genres = load_npz(processed / "genre_features.npz")
    genre_names = json.loads((processed / "genre_names.json").read_text())
    content = build_enriched_features(users, load_user_metadata(), movie_mapping, load_movie_metadata(),
                                      genres, genre_names, history)
    static = add_identity_features(static_features(content), users.user_id, movie_mapping.movie_id)
    id_users = identity(len(users), dtype="float32", format="csr")
    id_items = identity(len(movie_mapping), dtype="float32", format="csr")
    winner = None
    old_search = pd.read_csv(ROOT / "outputs/tables/lightfm_validation_search.csv")
    for old in old_search.to_dict("records"):
        row = record("LightFM", "id_history", old, epochs=int(old["epochs"]), reused=True)
        if winner is None or key(row) > key(winner): winner = row
    best_model = LightFMRecommender.load(ROOT / "outputs/models/lightfm")
    best_config = {"feature_set": "id_history", "epochs": params["epochs"], "reused": True}
    for variant, uf, itf in (("id_only", id_users, id_items), ("id_static", static.users, static.items)):
        engine = LightFM(no_components=32, loss="logistic", learning_rate=.05, user_alpha=1e-6,
                         item_alpha=1e-6, random_state=42)
        for epoch in (5, 10):
            tick = perf_counter()
            engine.fit_partial(labels, user_features=uf, item_features=itf, epochs=5, num_threads=1)
            model = LightFMRecommender.from_fitted(engine, itf, users.user_id, user_features=uf)
            metrics, _ = evaluate_ranking(model.score_tokens, validation, movie_ids)
            row = record("LightFM", variant, metrics, epochs=epoch, reused=False,
                         checkpoint_seconds=perf_counter()-tick)
            if key(row) > key(winner):
                winner, best_model = row, model
                best_config = {"feature_set": variant, "epochs": epoch, "reused": False}
                best_model.save(model_root / "lightfm")
    best_model.save(model_root / "lightfm")
    selected_models["LightFM"] = best_model
    validation_winners["LightFM"] = winner
    configs["LightFM"] = {**best_config, "loss": "logistic", "no_components": 32, "alpha": 1e-6,
                           "seed": 42, "fit_positive_labels": int((labels.data > 0).sum()),
                           "fit_negative_labels": int((labels.data < 0).sum()),
                           "identity_only_weight": 1.0, "hybrid_identity_weight": .5}

    # Freeze all three configurations before opening the test contexts or test-stage histories.
    selection = {"selection_metric": "validation NDCG@10, then Recall@10; keep baseline on exact ties",
                 "configurations": configs, "validation_winners": validation_winners}
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print("All configurations frozen; evaluating one winner per algorithm on test.", flush=True)
    test = pd.read_json(processed / "test_contexts.jsonl", lines=True)
    comparison = []
    for algorithm, slug in (("ItemCF", "itemcf"), ("LightFM", "lightfm"), ("LRURec", "lrurec")):
        model = selected_models[algorithm]
        if algorithm == "ItemCF":
            days = configs[algorithm]["half_life_days"]
            if days is None:
                metrics, per_user = evaluate_ranking(cf.score_tokens, test, movie_ids)
            else:
                known = pd.concat([train, pd.read_csv(processed / "validation.csv")], ignore_index=True)
                metrics, per_user = evaluate_ranking(decay_scorer(cf, decay_histories(known, movie_map, days)), test, movie_ids)
        elif algorithm == "LRURec":
            metrics, per_user = evaluate_batched(model, test, movie_ids, 64)
        else:
            metrics, per_user = evaluate_ranking(model.score_tokens, test, movie_ids)
        per_user.to_csv(model_root / f"{slug}_test_per_user.csv", index=False)
        rows.append({"algorithm": algorithm, "variant": validation_winners[algorithm]["variant"],
                     "split": "test", **metrics})
        for stage in ("validation", "test"):
            old = baselines[slug][f"{stage}_metrics"]
            new = validation_winners[algorithm] if stage == "validation" else metrics
            for version, result in (("baseline", old), ("round1_selected", new)):
                comparison.append({"algorithm": algorithm, "version": version, "split": stage,
                                   "users": result["users"], **{metric: result[metric] for metric in METRICS}})
        print(f'{algorithm} selected test NDCG@10={metrics["NDCG@10"]:.6f}', flush=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)
    unchanged = all(digest(ROOT / file) == expected for file, expected in initial_hashes.items())
    if not unchanged: raise RuntimeError("An original baseline was unexpectedly modified.")
    run = {**selection, "completed": True, "elapsed_seconds": perf_counter()-started,
           "input_sha256": hashes, "baseline_artifact_sha256": initial_hashes,
           "baseline_artifacts_unchanged": unchanged, "metric_names": METRICS,
           "lrurec_scope": "Inference context ablation at fixed weights trained with length 50, not a retraining study",
           "lightfm_scope": "Same suffix fitting labels, seed, optimizer parameters and epoch grid; identity scaling follows feature normalization",
           "time_decay_grid_days": [None, 1, 7, 30, 90]}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'Completed in {run["elapsed_seconds"]:.1f}s. Results: {out}', flush=True)


if __name__ == "__main__":
    main()
