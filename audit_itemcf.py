"""Audit the fixed ItemCF model against raw split records; do not retune it."""

import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

from src.data import PROJECT_ROOT, SCHEMAS, resolve_data_dir
from src.evaluate import evaluate_ranking, rank_candidates
from src.models.item_cf import ItemCF
from src.recommend import file_sha256


def main():
    processed = PROJECT_ROOT / "data" / "processed"
    tables = PROJECT_ROOT / "outputs" / "tables"
    model_dir = PROJECT_ROOT / "outputs" / "models" / "itemcf"
    config = json.loads((processed / "config.json").read_text(encoding="utf-8"))
    run = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    model = ItemCF.load(model_dir)
    user_map = pd.read_csv(processed / "user_mapping.csv").set_index("user_id")["user_index"]
    mapping = pd.read_csv(processed / "movie_mapping.csv").sort_values("movie_index")
    movie_ids = mapping["movie_id"].to_numpy(dtype="int64")
    item_map = mapping.set_index("movie_id")["movie_index"]
    catalog = set(movie_ids)
    splits = {name: pd.read_csv(processed / f"{name}.csv") for name in ["train", "validation", "test"]}
    train = splits["train"]
    checks = []

    def check(name, violations):
        checks.append({"check": name, "violations": int(violations)})

    check("input_checksum_mismatches", sum(
        file_sha256(processed / name) != checksum for name, checksum in run["input_sha256"].items()))
    source_dir = resolve_data_dir()
    check("source_file_checksum_mismatches", sum(
        file_sha256(source_dir / name) != checksum for name, checksum in config["source_sha256"].items()))
    raw = pd.read_csv(source_dir / "ratings.dat", sep="::", engine="python", header=None,
                      names=list(SCHEMAS["ratings"]), dtype=SCHEMAS["ratings"], encoding="latin-1",
                      keep_default_na=False, na_values=[""])
    combined = pd.concat(splits.values(), ignore_index=True).sort_values("event_id")
    rating_columns = list(SCHEMAS["ratings"])
    expected_raw = raw.iloc[combined["event_id"].to_numpy()][rating_columns].to_numpy(dtype="int64")
    actual_raw = combined[rating_columns].to_numpy(dtype="int64")
    check("source_rating_rows_mismatched", int(np.any(expected_raw != actual_raw, axis=1).sum()))
    removed = int(pd.read_csv(tables / "cleaning_audit.csv")["removed_rows"].sum())
    check("source_row_count_not_explained_by_cleaning", abs(len(raw) - len(combined) - removed))
    for stage, frame in splits.items():
        check(f"{stage}_label_mismatches", int(
            frame["is_relevant"].ne(frame["rating"].ge(config["relevance_threshold"]).astype(int)).sum()))

    rows = train["user_id"].map(user_map).to_numpy()
    columns = train["movie_id"].map(item_map).to_numpy()
    shape = (len(user_map), len(movie_ids))
    rebuilt_explicit = csr_matrix((train["rating"].to_numpy(dtype="float32"), (rows, columns)), shape=shape)
    positive_mask = train["rating"].ge(config["relevance_threshold"]).to_numpy()
    rebuilt_positive = csr_matrix((np.ones(int(positive_mask.sum()), dtype="float32"),
                                   (rows[positive_mask], columns[positive_mask])), shape=shape)
    check("explicit_matrix_entries_mismatched", (rebuilt_explicit - load_npz(processed / "explicit_ratings.npz")).nnz)
    check("positive_matrix_entries_mismatched", (rebuilt_positive - load_npz(processed / "positive_interactions.npz")).nnz)
    check("similarity_diagonal_nonzeros", np.count_nonzero(model.similarity_.diagonal()))
    check("similarity_nonfinite_values", int((~np.isfinite(model.similarity_.data)).sum()))

    # Independent row-wise dot products on real data, including pruning direction.
    support = np.asarray(rebuilt_positive.sum(axis=0)).ravel()
    bad_similarity_rows = 0
    for source in np.linspace(0, len(movie_ids) - 1, min(24, len(movie_ids)), dtype=int):
        overlap = (rebuilt_positive[:, source].T @ rebuilt_positive).toarray().ravel()
        denominator = np.sqrt(support[source] * support)
        weights = np.divide(overlap, denominator, out=np.zeros_like(overlap), where=denominator > 0)
        if model.shrinkage:
            weights *= overlap / (overlap + model.shrinkage)
        weights[source] = 0
        neighbors = np.flatnonzero(weights > 0)
        selected = neighbors[np.lexsort((neighbors, -weights[neighbors]))[:model.n_neighbors]]
        expected = np.zeros(len(movie_ids), dtype="float32")
        expected[selected] = weights[selected]
        actual = model.similarity_.getrow(source).toarray().ravel()
        bad_similarity_rows += not np.allclose(expected, actual, rtol=1e-6, atol=1e-7)
    check("sampled_similarity_rows_mismatched", bad_similarity_rows)

    train_positive_users = set(train.loc[positive_mask, "user_id"])
    status_records, scope_records, contexts_by_stage = [], [], {}
    test_per_user = pd.read_csv(model_dir / "test_per_user.csv").set_index("user_id")
    for stage in ["validation", "test"]:
        contexts = pd.read_json(processed / f"{stage}_contexts.jsonl", lines=True)
        contexts_by_stage[stage] = contexts
        history = train if stage == "validation" else pd.concat([train, splits["validation"]])
        history = history.sort_values(["user_id", "timestamp", "movie_id", "event_id"])
        seen_by_user = history.groupby("user_id")["movie_id"].agg(set).to_dict()
        positive_history = history.loc[history["rating"].ge(config["relevance_threshold"])
                                       & history["movie_id"].isin(catalog)].copy()
        positive_history["token"] = positive_history["movie_id"].map(item_map) + 1
        tokens_by_user = positive_history.groupby("user_id")["token"].agg(list).to_dict()
        holdout = splits[stage]
        ratings_by_user = {user: group.set_index("movie_id")["rating"].to_dict()
                           for user, group in holdout.groupby("user_id")}
        check(f"{stage}_context_user_mismatches", len(set(contexts["user_id"]) ^ set(holdout["user_id"])))
        mismatches = dict(history=0, tokens=0, targets=0, candidates=0, eligibility=0, seen_recommendations=0)
        if stage == "test":
            mismatches.update(independently_counted_hits=0, independently_counted_precision=0)
        categories = {"heldout_rating_4_or_5": 0, "heldout_rating_1_to_3": 0, "no_heldout_rating": 0}
        zero_score_slots, nonzero_candidates, predicted_support, candidate_support = 0, [], [], []
        for row in contexts.itertuples(index=False):
            seen = seen_by_user.get(row.user_id, set())
            expected_tokens = tokens_by_user.get(row.user_id, [])
            observed_ratings = ratings_by_user[row.user_id]
            targets = {movie for movie, rating in observed_ratings.items()
                       if rating >= config["relevance_threshold"] and movie in catalog and movie not in seen}
            candidate_count = len(catalog - seen)
            eligible = (row.user_id in train_positive_users and bool(targets)
                        and candidate_count >= config["evaluation_k"])
            mismatches["history"] += set(row.seen_movie_ids) != seen
            mismatches["tokens"] += row.positive_movie_tokens != expected_tokens
            mismatches["targets"] += set(row.relevant_movie_ids) != targets
            mismatches["candidates"] += row.candidate_count != candidate_count
            mismatches["eligibility"] += row.eligible != eligible
            if not row.eligible:
                continue
            scores = model.score(np.asarray(expected_tokens, dtype=int) - 1)
            ranked, ranked_scores = rank_candidates(scores, movie_ids, seen, 10)
            mismatches["seen_recommendations"] += bool(set(ranked) & seen)
            raw_hits = sum(observed_ratings.get(int(movie), 0) >= config["relevance_threshold"] for movie in ranked)
            if stage == "test":
                saved = test_per_user.loc[row.user_id]
                mismatches["independently_counted_hits"] += raw_hits != saved["Hits@10"]
                mismatches["independently_counted_precision"] += not np.isclose(raw_hits / 10, saved["Precision@10"])
            for movie in ranked:
                if int(movie) not in observed_ratings:
                    categories["no_heldout_rating"] += 1
                elif observed_ratings[int(movie)] >= config["relevance_threshold"]:
                    categories["heldout_rating_4_or_5"] += 1
                else:
                    categories["heldout_rating_1_to_3"] += 1
            zero_score_slots += int((ranked_scores == 0).sum())
            candidate_indexes = np.flatnonzero(~np.isin(movie_ids, list(seen)))
            nonzero_candidates.append(int((scores[candidate_indexes] > 0).sum()))
            predicted_support.append(float(support[item_map.loc[ranked].to_numpy()].mean()))
            candidate_support.append(float(support[candidate_indexes].mean()))
        for name, count in mismatches.items():
            check(f"{stage}_{name}_mismatches", count)
        eligible_rows = contexts.loc[contexts["eligible"]]
        n_users = len(eligible_rows)
        aggregate = run[f"{stage}_metrics"]
        total_hits = categories["heldout_rating_4_or_5"]
        check(f"{stage}_aggregate_precision_mismatch", not np.isclose(
            total_hits / (n_users * 10), aggregate["Precision@10"], rtol=0, atol=1e-12))
        check(f"{stage}_aggregate_mean_hits_mismatch", not np.isclose(
            total_hits / n_users, aggregate["MeanHits@10"], rtol=0, atol=1e-12))
        for category, count in categories.items():
            status_records.append({"stage": stage, "category": category, "recommendations": count,
                                   "share": count / (n_users * 10), "mean_per_user": count / n_users})
        relevant_counts = eligible_rows["relevant_movie_ids"].map(len)
        scope_records.append({
            "stage": stage, "users": n_users, "mean_candidate_count": eligible_rows["candidate_count"].mean(),
            "mean_observed_positive_targets": relevant_counts.mean(),
            "users_with_one_observed_positive": int(relevant_counts.eq(1).sum()),
            "users_with_fewer_than_10_observed_positives": int(relevant_counts.lt(10).sum()),
            "mean_observable_precision_ceiling_at_10": float((relevant_counts.clip(upper=10) / 10).mean()),
            "zero_score_top10_slots": zero_score_slots,
            "mean_nonzero_score_candidates": float(np.mean(nonzero_candidates)),
            "mean_training_support_recommended": float(np.mean(predicted_support)),
            "mean_training_support_unseen_candidates": float(np.mean(candidate_support)),
        })

    # A validation-only sanity reference; it is not an additional project model.
    popularity, _ = evaluate_ranking(lambda user, tokens: support, contexts_by_stage["validation"], movie_ids)
    reference = pd.DataFrame([
        {"method": "TrainingPopularityDiagnostic", "split": "validation", **popularity},
        {"method": "ItemCF", "split": "validation", **run["validation_metrics"]},
    ])
    integrity = pd.DataFrame(checks)
    integrity.to_csv(tables / "itemcf_integrity_checks.csv", index=False)
    pd.DataFrame(status_records).to_csv(tables / "itemcf_recommendation_status.csv", index=False)
    pd.DataFrame(scope_records).to_csv(tables / "itemcf_evaluation_scope.csv", index=False)
    reference.to_csv(tables / "itemcf_validation_reference.csv", index=False)
    distributions = []
    for k in [5, 10]:
        counts = test_per_user[f"Hits@{k}"].value_counts().reindex(range(k + 1), fill_value=0)
        for hits, count in counts.items():
            distributions.append({"k": k, "hits": hits, "users": int(count), "share": count / len(test_per_user)})
    pd.DataFrame(distributions).to_csv(tables / "itemcf_hit_distribution.csv", index=False)
    print(integrity.to_string(index=False))
    print(pd.DataFrame(status_records).to_string(index=False))
    print(pd.DataFrame(scope_records).to_string(index=False))
    if integrity["violations"].any():
        raise RuntimeError("ItemCF audit found inconsistencies; inspect itemcf_integrity_checks.csv.")
    print(f'PASS: all {len(integrity)} integrity checks; audit tables saved')


if __name__ == "__main__":
    main()
