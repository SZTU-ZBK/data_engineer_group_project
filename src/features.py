"""Training-only model inputs and a shared offline recommendation protocol."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, save_npz


@dataclass
class ModelInputs:
    user_mapping: pd.DataFrame
    movie_mapping: pd.DataFrame
    train_interactions: pd.DataFrame
    explicit_ratings: csr_matrix
    positive_interactions: csr_matrix
    genre_features: csr_matrix
    genre_names: list[str]
    train_sequences: pd.DataFrame


def build_model_inputs(train: pd.DataFrame, movies: pd.DataFrame) -> ModelInputs:
    """Fit vocabularies and features exclusively to the training partition.

    Sparse-matrix indexes start at zero. Sequential movie tokens are movie_index
    + 1, reserving token zero for padding. Sequences contain positive training
    interactions only, without truncation. Equal-time events are ordered by movie
    ID for reproducibility; this is not evidence of an actual viewing order.
    """
    if train.empty:
        raise ValueError("Training data must not be empty.")
    user_mapping = pd.DataFrame({"user_id": sorted(train["user_id"].unique())})
    user_mapping["user_index"] = range(len(user_mapping))
    movie_mapping = pd.DataFrame({"movie_id": sorted(train["movie_id"].unique())})
    movie_mapping["movie_index"] = range(len(movie_mapping))
    interactions = (train.merge(user_mapping, on="user_id", validate="many_to_one")
                    .merge(movie_mapping, on="movie_id", validate="many_to_one")
                    .sort_values(["user_id", "timestamp", "movie_id", "event_id"], kind="stable")
                    .reset_index(drop=True))
    shape = (len(user_mapping), len(movie_mapping))
    explicit = csr_matrix((interactions["rating"].to_numpy(dtype="float32"),
                           (interactions["user_index"], interactions["movie_index"])), shape=shape)
    positive = interactions.loc[interactions["is_relevant"].eq(1)].copy()
    implicit = csr_matrix((positive["is_relevant"].to_numpy(dtype="float32"),
                           (positive["user_index"], positive["movie_index"])), shape=shape)

    training_movies = movie_mapping.merge(movies[["movie_id", "genres"]],
                                         on="movie_id", how="left", validate="one_to_one")
    if training_movies["genres"].isna().any():
        raise ValueError("Every training movie must have genre metadata.")
    multi_hot = training_movies["genres"].str.get_dummies(sep="|").astype("float32")
    genre_features = csr_matrix(multi_hot.to_numpy())

    positive["movie_token"] = positive["movie_index"] + 1
    grouped = positive.groupby("user_id").agg(
        movie_tokens=("movie_token", list), timestamps=("timestamp", list), event_ids=("event_id", list),
    )
    sequences = user_mapping.merge(grouped, on="user_id", how="left", validate="one_to_one")
    for column in ["movie_tokens", "timestamps", "event_ids"]:
        sequences[column] = sequences[column].map(lambda value: value if isinstance(value, list) else [])
    return ModelInputs(user_mapping, movie_mapping, interactions, explicit, implicit,
                       genre_features, list(multi_hot.columns), sequences)


def build_evaluation_context(
    splits: dict[str, pd.DataFrame], inputs: ModelInputs, stage: str, k: int = 10,
) -> pd.DataFrame:
    """Use train history for validation, train+validation history for test.

    Model parameters remain trained on train only in this protocol. Candidate
    movies come from the training catalog; all previously rated movies, including
    low ratings, are masked. Ground truth is warm, unseen positive holdout items.
    Users require a positive training history, at least one warm relevant target,
    and at least k candidates, giving all three models the same evaluation cohort.
    Cold targets and excluded users are explicitly counted, not treated as hits.
    """
    if stage not in ("validation", "test") or k < 1:
        raise ValueError("stage must be validation/test and k must be positive.")
    train = splits["train"]
    history = (train if stage == "validation" else pd.concat([train, splits["validation"]]))
    history = history.sort_values(["user_id", "timestamp", "movie_id", "event_id"], kind="stable")
    holdout = splits[stage]
    catalog = set(inputs.movie_mapping["movie_id"])
    token_map = inputs.movie_mapping.set_index("movie_id")["movie_index"] + 1
    seen_by_user = history.groupby("user_id")["movie_id"].agg(list).to_dict()
    positives = history.loc[history["is_relevant"].eq(1) & history["movie_id"].isin(catalog)].copy()
    positives["movie_token"] = positives["movie_id"].map(token_map)
    tokens_by_user = positives.groupby("user_id")["movie_token"].agg(list).to_dict()
    targets_by_user = (holdout.loc[holdout["is_relevant"].eq(1)]
                       .groupby("user_id")["movie_id"].agg(list).to_dict())
    train_positive_users = set(train.loc[train["is_relevant"].eq(1), "user_id"])
    records = []
    for user_id in sorted(holdout["user_id"].unique()):
        seen = set(seen_by_user.get(user_id, []))
        targets = set(targets_by_user.get(user_id, []))
        relevant = (targets & catalog) - seen
        candidate_count = len(catalog - seen)
        if user_id not in train_positive_users:
            reason = "no_train_positive_history"
        elif not relevant:
            reason = "no_warm_relevant_items"
        elif candidate_count < k:
            reason = "fewer_than_k_candidates"
        else:
            reason = "eligible"
        records.append({
            "user_id": user_id, "seen_movie_ids": sorted(seen),
            "positive_movie_tokens": tokens_by_user.get(user_id, []),
            "relevant_movie_ids": sorted(relevant), "candidate_count": candidate_count,
            "cold_relevant_count": len(targets - catalog),
            "eligible": reason == "eligible", "exclusion_reason": reason,
        })
    return pd.DataFrame(records, columns=[
        "user_id", "seen_movie_ids", "positive_movie_tokens", "relevant_movie_ids",
        "candidate_count", "cold_relevant_count", "eligible", "exclusion_reason",
    ]).astype({"eligible": bool})


def candidate_movie_ids(inputs: ModelInputs, seen_movie_ids: list[int]) -> list[int]:
    """Return the shared, deterministically ordered candidate set in raw movie IDs."""
    seen = set(seen_movie_ids)
    return [int(movie_id) for movie_id in inputs.movie_mapping["movie_id"] if movie_id not in seen]


def evaluation_summary(contexts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, frame in contexts.items():
        rows.append({
            "stage": stage, "holdout_users": len(frame),
            "eligible_users": int(frame["eligible"].sum()),
            "excluded_no_train_positive_history": int(frame["exclusion_reason"].eq("no_train_positive_history").sum()),
            "excluded_no_warm_relevant_items": int(frame["exclusion_reason"].eq("no_warm_relevant_items").sum()),
            "excluded_fewer_than_k_candidates": int(frame["exclusion_reason"].eq("fewer_than_k_candidates").sum()),
            "positive_pairs_with_cold_movie": int(frame["cold_relevant_count"].sum()),
            "eligible_relevant_pairs": int(frame.loc[frame["eligible"], "relevant_movie_ids"].map(len).sum()),
        })
    return pd.DataFrame(rows).set_index("stage")


def build_preprocessing_config(
    data_dir: str | Path, cleaned_ratings: pd.DataFrame, *,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    relevance_threshold: int = 4, evaluation_k: int = 10,
) -> dict:
    """Build the existing export metadata without cluttering the analysis notebook."""
    data_dir = Path(data_dir)
    return {
        "dataset": "MovieLens 1M",
        "source_sha256": {f"{name}.dat": hashlib.sha256((data_dir / f"{name}.dat").read_bytes()).hexdigest()
                          for name in ["users", "movies", "ratings"]},
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "split_ratios": dict(zip(["train", "validation", "test"], split_ratios)),
        "split_method": "per_user_nearest_feasible_cumulative_timestamp_groups",
        "boundary_tie_break": "earlier_timestamp_group",
        "users_with_fewer_than_3_timestamp_groups": "train_only",
        "train_only_user_count": int(cleaned_ratings.groupby("user_id")["timestamp"].nunique().lt(3).sum()),
        "global_calendar_split": False,
        "relevance_threshold": relevance_threshold,
        "label_definition": "rating >= relevance_threshold; original ratings retained",
        "mapping_fit_partition": "train",
        "sequence_source": "positive training interactions, ordered by timestamp then movie_id",
        "sequence_max_length": None,
        "padding_token": 0,
        "sequence_movie_token": "movie_index + 1",
        "candidate_universe": "all movies observed in train",
        "history_filter": "exclude all previously rated movies, including low ratings",
        "validation_history": ["train"],
        "test_history": ["train", "validation"],
        "model_fit_partitions": ["train"],
        "evaluation_k": evaluation_k,
        "evaluation_users": "positive training history, warm positive target, at least k candidates",
        "cold_target_policy": "reported separately, excluded from primary warm-target ranking metrics",
        "negative_sampling": "not performed",
        "random_seed": None,
    }


def save_preprocessed_artifacts(
    splits: dict[str, pd.DataFrame], inputs: ModelInputs,
    contexts: dict[str, pd.DataFrame], output_dir: str | Path, config: dict,
) -> None:
    """Save reproducible local inputs; this directory must stay out of version control."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    inputs.user_mapping.to_csv(directory / "user_mapping.csv", index=False)
    inputs.movie_mapping.to_csv(directory / "movie_mapping.csv", index=False)
    for name in ("explicit_ratings", "positive_interactions", "genre_features"):
        save_npz(directory / f"{name}.npz", getattr(inputs, name))
    inputs.train_sequences.to_json(directory / "train_sequences.jsonl", orient="records", lines=True)
    for stage, frame in contexts.items():
        frame.to_json(directory / f"{stage}_contexts.jsonl", orient="records", lines=True)
    (directory / "genre_names.json").write_text(json.dumps(inputs.genre_names, indent=2), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
