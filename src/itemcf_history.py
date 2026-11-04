"""Stage-local positive-history weights for time-decayed ItemCF scoring."""

import numpy as np


def decay_histories(history, movie_map, half_life_days):
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("Half-life must be finite and positive.")
    anchors = history.groupby("user_id").timestamp.max()
    positive = history.loc[history.is_relevant.eq(1) & history.movie_id.isin(movie_map)].copy()
    positive["index"] = positive.movie_id.map(movie_map)
    age = (positive.user_id.map(anchors) - positive.timestamp) / 86400.0
    positive["weight"] = np.exp2(-age / half_life_days)
    return {int(user): (group["index"].to_numpy(dtype=int), group.weight.to_numpy(dtype="float32"))
            for user, group in positive.groupby("user_id", sort=False)}


def decay_scorer(model, histories):
    def score(user, tokens):
        indexes, weights = histories[user]
        expected = np.unique(np.asarray(tokens, dtype=int))
        if not np.array_equal(np.sort(indexes + 1), expected[expected > 0]):
            raise ValueError("Time-decay history differs from the shared evaluation context.")
        return model.score(indexes, weights)
    return score


def add_context_weights(context, histories):
    result = dict(context)
    if not any(token > 0 for token in result["positive_movie_tokens"]):
        result["positive_history_weights"] = [0.0] * len(result["positive_movie_tokens"])
        return result
    indexes, weights = histories[int(result["user_id"])]
    by_token = dict(zip(indexes + 1, weights))
    result["positive_history_weights"] = [float(by_token[token]) if token > 0 else 0.0
                                           for token in result["positive_movie_tokens"]]
    return result
