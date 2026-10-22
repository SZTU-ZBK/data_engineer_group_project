"""Shared full-catalog ranking, deterministic tie-breaking and ranking metrics."""

import numpy as np
import pandas as pd


def rank_candidates(scores, movie_ids, seen_movie_ids, k: int = 10):
    """Return up to k unseen raw movie IDs and scores, sorted by score then ID."""
    scores = np.asarray(scores)
    movie_ids = np.asarray(movie_ids, dtype="int64")
    if k < 1 or scores.ndim != 1 or scores.shape != movie_ids.shape:
        raise ValueError("Provide matching one-dimensional scores/IDs and a positive k.")
    if not np.isfinite(scores).all():
        raise ValueError("All ranking scores must be finite.")
    candidates = np.flatnonzero(~np.isin(movie_ids, list(seen_movie_ids)))
    order = np.lexsort((movie_ids[candidates], -scores[candidates]))[:k]
    ranked = candidates[order]
    return movie_ids[ranked], scores[ranked]


def ranking_metrics(recommended_movie_ids, relevant_movie_ids, k_values=(5, 10)) -> dict:
    """Binary-relevance Recall/NDCG. Recall denominator is ALL warm targets."""
    recommended = list(recommended_movie_ids)
    relevant = set(relevant_movie_ids)
    if not relevant or len(set(recommended)) != len(recommended):
        raise ValueError("Metrics require nonempty ground truth and unique recommendations.")
    result = {}
    for k in k_values:
        if k < 1 or len(recommended) < k:
            raise ValueError("Provide at least k recommendations for every requested cutoff.")
        hits = np.asarray([movie in relevant for movie in recommended[:k]], dtype="float64")
        discounts = 1 / np.log2(np.arange(2, k + 2))
        ideal = discounts[:min(k, len(relevant))].sum()
        result[f"Recall@{k}"] = float(hits.sum() / len(relevant))
        result[f"NDCG@{k}"] = float(np.dot(hits, discounts) / ideal)
    return result


def evaluate_ranking(score_fn, contexts: pd.DataFrame, movie_ids, k_values=(5, 10)):
    """Evaluate the shared eligible cohort against the entire unseen train catalog.

    score_fn receives only user_id and known positive history tokens. Held-out
    relevant items are used by metrics after ranking, never passed to the scorer.
    Aggregation is a macro-average across eligible users, not across ratings.
    """
    cohort = contexts.loc[contexts["eligible"]]
    if cohort.empty:
        raise ValueError("No eligible users to evaluate.")
    k_values = tuple(sorted(set(k_values)))
    if not k_values or min(k_values) < 1:
        raise ValueError("Provide at least one positive ranking cutoff.")
    movie_ids = np.asarray(movie_ids, dtype="int64")
    recommended_catalog = {k: set() for k in k_values}
    rows = []
    for row in cohort.itertuples(index=False):
        scores = score_fn(int(row.user_id), row.positive_movie_tokens)
        ranked, _ = rank_candidates(scores, movie_ids, row.seen_movie_ids, max(k_values))
        metrics = ranking_metrics(ranked, row.relevant_movie_ids, k_values)
        rows.append({"user_id": int(row.user_id), "relevant_items": len(row.relevant_movie_ids),
                     "candidate_count": int(row.candidate_count), **metrics})
        for k in k_values:
            recommended_catalog[k].update(ranked[:k].tolist())
    per_user = pd.DataFrame(rows)
    aggregate = {"users": len(per_user)}
    for k in k_values:
        aggregate[f"Recall@{k}"] = float(per_user[f"Recall@{k}"].mean())
        aggregate[f"NDCG@{k}"] = float(per_user[f"NDCG@{k}"].mean())
        aggregate[f"CatalogCoverage@{k}"] = len(recommended_catalog[k]) / len(movie_ids)
    return aggregate, per_user
