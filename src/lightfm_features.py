"""Attribute/history features without identity embeddings or target-row statistics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.models.lightfm import build_item_features, build_user_features


@dataclass
class ContentFeatures:
    users: csr_matrix
    items: csr_matrix
    user_names: list[str]
    item_names: list[str]


def normalize_rows(matrix):
    matrix = csr_matrix(matrix, dtype="float32")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return matrix.multiply(inverse[:, None]).tocsr()


def split_feature_history(train):
    """Nearest half per user by timestamp group: prefix for features, suffix for fit.

    A user with one timestamp group contributes only feature history. Entire
    groups stay together, and no parameter-fit event enters feature statistics.
    """
    histories, targets = [], []
    for _, group in train.sort_values(["user_id", "timestamp", "movie_id"]).groupby("user_id", sort=False):
        counts = group.groupby("timestamp", sort=True).size()
        if len(counts) < 2:
            histories.append(group)
            continue
        boundary = int(np.argmin(np.abs(counts.cumsum().to_numpy()[:-1] - len(group) / 2)))
        earlier = group.timestamp.le(counts.index[boundary])
        histories.append(group.loc[earlier])
        targets.append(group.loc[~earlier])
    if not targets:
        raise ValueError("Feature history leaves no later train events for parameter fitting.")
    return pd.concat(histories, ignore_index=True), pd.concat(targets, ignore_index=True)


def build_enriched_features(users, user_metadata, movies, movie_metadata, genres, genre_names,
                            history, relevance_threshold=4, prior_weight=20.0):
    """Fit attributes to train mappings; derive statistics only from supplied history.

    Each demographic/genre-preference/decade-preference user block has unit mass
    before final row normalization. Movie features are genres, release decade,
    scaled log count, smoothed mean rating and smoothed high-rating proportion.
    """
    if history.empty or prior_weight <= 0 or not history.rating.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Use nonempty 1-5 history ratings and positive smoothing strength.")
    demographics, demographic_names = build_user_features(users, user_metadata)
    genre_features = build_item_features(genres)
    if genres.shape != (len(movies), len(genre_names)):
        raise ValueError("Movie rows and genre vocabulary must align.")
    aligned = movies[["movie_id"]].merge(movie_metadata[["movie_id", "title"]], on="movie_id",
                                          how="left", validate="one_to_one", sort=False)
    years = pd.to_numeric(aligned.title.str.extract(r"\((\d{4})\)\s*$", expand=False), errors="coerce")
    if years.isna().any():
        raise ValueError("Every training movie must have an identifiable release year.")
    encoded_decades = pd.get_dummies((years // 10 * 10).astype(int).astype(str),
                                     prefix="release_decade", dtype="float32")
    decades = csr_matrix(encoded_decades.to_numpy())
    events = (history[["user_id", "movie_id", "rating"]]
              .merge(users[["user_id", "user_index"]], on="user_id", how="left", validate="many_to_one")
              .merge(movies[["movie_id", "movie_index"]], on="movie_id", how="left", validate="many_to_one"))
    if events[["user_index", "movie_index"]].isna().any().any():
        raise ValueError("Feature history contains users/movies outside the training mappings.")
    positive = events.loc[events.rating.ge(relevance_threshold)]
    positives = csr_matrix((np.ones(len(positive), dtype="float32"),
                            (positive.user_index, positive.movie_index)), shape=(len(users), len(movies)))
    genre_preferences = normalize_rows(positives @ genre_features)
    decade_preferences = normalize_rows(positives @ decades)
    user_features = normalize_rows(hstack([demographics, genre_preferences, decade_preferences], format="csr"))

    indexes = events.movie_index.to_numpy(dtype=int)
    counts = np.bincount(indexes, minlength=len(movies)).astype("float32")
    sums = np.bincount(indexes, weights=events.rating, minlength=len(movies))
    high = np.bincount(indexes, weights=events.rating.ge(relevance_threshold), minlength=len(movies))
    popularity = np.log1p(counts) / np.log1p(max(float(counts.max()), 1.0))
    mean = (sums + prior_weight * events.rating.mean()) / (counts + prior_weight) / 5.0
    high_ratio = (high + prior_weight * events.rating.ge(relevance_threshold).mean()) / (counts + prior_weight)
    statistics = csr_matrix(np.column_stack([popularity, mean, high_ratio]), dtype="float32")
    item_features = normalize_rows(hstack([genre_features, decades, statistics], format="csr"))
    return ContentFeatures(
        user_features, item_features,
        [*demographic_names, *[f"preferred_genre:{name}" for name in genre_names],
         *[f"preferred_{name}" for name in encoded_decades.columns]],
        [*[f"genre:{name}" for name in genre_names], *encoded_decades.columns,
         "history_log_popularity", "history_smoothed_mean_rating", "history_smoothed_positive_ratio"],
    )
