"""Read MovieLens 1M and inspect its raw tables without changing the records."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {
    "users": {
        "user_id": "Int64",
        "gender": "string",
        "age": "Int64",
        "occupation": "Int64",
        "zip_code": "string",
    },
    "movies": {
        "movie_id": "Int64",
        "title": "string",
        "genres": "string",
    },
    "ratings": {
        "user_id": "Int64",
        "movie_id": "Int64",
        "rating": "Int64",
        "timestamp": "Int64",
    },
}


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Find all three files; explicit relative paths are relative to the repo root."""
    if data_dir is not None:
        candidate = Path(data_dir).expanduser()
        candidates = [candidate if candidate.is_absolute() else PROJECT_ROOT / candidate]
    else:
        candidates = [
            PROJECT_ROOT / "data" / "ml-1m",
            PROJECT_ROOT / "ml-1m" / "ml-1m",
            PROJECT_ROOT / "ml-1m",
        ]

    for candidate in candidates:
        if all((candidate / f"{name}.dat").is_file() for name in SCHEMAS):
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Expected users.dat, movies.dat and ratings.dat in one directory.\n"
        f"Searched:\n{searched}\n"
        "Download and extract MovieLens 1M, or pass data_dir explicitly."
    )


def load_movielens(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load raw tables, preserving ID values, ZIP strings and Unix timestamps.

    The double-colon separator requires pandas' Python parser. Latin-1 handles
    accented movie titles. Only empty fields are treated as missing; text such
    as 'NA' is retained. Nullable integer columns allow missing-value inspection.
    Age is an age-group code, occupation is a category code, and genres remain
    pipe-separated strings. Encoding and train/validation/test splits come later.
    """
    directory = resolve_data_dir(data_dir)
    return {
        name: pd.read_csv(
            directory / f"{name}.dat",
            sep="::",
            engine="python",
            header=None,
            names=list(schema),
            dtype=schema,
            encoding="latin-1",
            keep_default_na=False,
            na_values=[""],
            on_bad_lines="error",
        )
        for name, schema in SCHEMAS.items()
    }


def dataset_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per table: shape, missing fields and identical duplicate rows."""
    return pd.DataFrame(
        [
            {
                "table": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "missing_values": int(frame.isna().sum().sum()),
                "columns_with_missing": ", ".join(
                    frame.columns[frame.isna().any()].tolist()
                ) or "(none)",
                "duplicate_rows": int(frame.duplicated().sum()),
            }
            for name, frame in tables.items()
        ]
    ).set_index("table")


def column_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report each column's dtype, missing count and non-null unique count."""
    return pd.DataFrame(
        [
            {
                "table": name,
                "column": column,
                "dtype": str(frame[column].dtype),
                "missing_values": int(frame[column].isna().sum()),
                "unique_values": int(frame[column].nunique(dropna=True)),
            }
            for name, frame in tables.items()
            for column in frame.columns
        ]
    ).set_index(["table", "column"])


def quality_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count suspicious records; this reports issues without removing any data."""
    users, movies, ratings = (tables[name] for name in ("users", "movies", "ratings"))
    checks = {
        "duplicate_user_ids": int(users["user_id"].duplicated().sum()),
        "duplicate_movie_ids": int(movies["movie_id"].duplicated().sum()),
        "duplicate_user_movie_pairs": int(
            ratings.duplicated(subset=["user_id", "movie_id"]).sum()
        ),
        "ratings_outside_1_to_5": int(
            (ratings["rating"].notna() & ~ratings["rating"].between(1, 5)).sum()
        ),
        "ratings_with_unknown_user": int(
            (ratings["user_id"].notna() & ~ratings["user_id"].isin(users["user_id"])).sum()
        ),
        "ratings_with_unknown_movie": int(
            (ratings["movie_id"].notna() & ~ratings["movie_id"].isin(movies["movie_id"])).sum()
        ),
        "negative_timestamps": int((ratings["timestamp"] < 0).sum()),
    }
    return pd.DataFrame(checks.items(), columns=["check", "count"]).set_index("check")


def clean_ratings(
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove invalid/exact duplicate rows with an audit; retain original scores.

    event_id is the zero-based row position in the input ratings file. Repeated
    user/movie pairs with conflicting scores/times require an explicit policy
    and raise an error rather than silently choosing a future observation.
    """
    frame = tables["ratings"].copy().reset_index(drop=True)
    frame.insert(0, "event_id", np.arange(len(frame), dtype="int64"))
    required = ["user_id", "movie_id", "rating", "timestamp"]
    audit = [{"action": "input", "removed_rows": 0, "remaining_rows": len(frame)}]

    def remove(mask: pd.Series, action: str) -> None:
        nonlocal frame
        removed = int(mask.sum())
        frame = frame.loc[~mask].copy()
        audit.append({"action": action, "removed_rows": removed, "remaining_rows": len(frame)})

    remove(frame[required].isna().any(axis=1), "missing_required_fields")
    remove(~frame["rating"].isin([1, 2, 3, 4, 5]), "invalid_rating")
    frame["rated_at"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True, errors="coerce")
    remove((frame["timestamp"] < 0) | frame["rated_at"].isna(), "invalid_timestamp")
    remove(~frame["user_id"].isin(tables["users"]["user_id"]), "unknown_user")
    remove(~frame["movie_id"].isin(tables["movies"]["movie_id"]), "unknown_movie")
    remove(frame.duplicated(subset=required), "exact_duplicate")
    if frame.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Repeated user/movie pairs remain; define a repeat-rating policy first.")
    if frame.empty:
        raise ValueError("No valid ratings remain after cleaning.")
    return frame.reset_index(drop=True), pd.DataFrame(audit).set_index("action")


def chronological_split(
    ratings: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, pd.DataFrame]:
    """Split each user's timestamp groups, approximately by rating counts.

    Choose the nearest feasible cumulative boundaries to train and train+val
    targets, resolving equal distances toward the earlier boundary. Users with
    fewer than three timestamp groups stay entirely in train. Same-time rows
    never cross splits. This is per-user chronology, not a global date cutoff.
    """
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Provide three positive split ratios summing to one.")
    if ratings.empty or ratings["event_id"].duplicated().any():
        raise ValueError("Provide nonempty cleaned ratings with unique event_id values.")
    groups = (ratings.groupby(["user_id", "timestamp"], sort=True)
              .size().rename("rows").reset_index())
    assignments = np.full(len(groups), "train", dtype=object)
    for _, user_groups in groups.groupby("user_id", sort=False):
        count = len(user_groups)
        if count < 3:
            continue
        cumulative = user_groups["rows"].to_numpy().cumsum()
        total = cumulative[-1]
        train_end = int(np.abs(cumulative[:count - 2] - total * ratios[0]).argmin()) + 1
        val_end = train_end + int(np.abs(
            cumulative[train_end:count - 1] - total * (ratios[0] + ratios[1])
        ).argmin()) + 1
        positions = user_groups.index.to_numpy()
        assignments[positions[train_end:val_end]] = "validation"
        assignments[positions[val_end:]] = "test"
    groups["split"] = assignments
    assigned = ratings.merge(groups[["user_id", "timestamp", "split"]],
                             on=["user_id", "timestamp"], validate="many_to_one")
    assigned = assigned.sort_values(["user_id", "timestamp", "movie_id", "event_id"], kind="stable")
    return {
        name: assigned.loc[assigned["split"].eq(name)].drop(columns="split").reset_index(drop=True)
        for name in ("train", "validation", "test")
    }


def add_relevance_labels(
    splits: dict[str, pd.DataFrame], threshold: int = 4,
) -> dict[str, pd.DataFrame]:
    """Attach a binary relevance flag after splitting; keep every original rating."""
    if threshold not in [1, 2, 3, 4, 5]:
        raise ValueError("Relevance threshold must be an integer from 1 to 5.")
    return {
        name: frame.assign(is_relevant=frame["rating"].ge(threshold).astype("int8"))
        for name, frame in splits.items()
    }


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report realized split proportions, labels, and training-unseen entities."""
    total = sum(len(frame) for frame in splits.values())
    train = splits["train"]
    train_users, train_movies = set(train["user_id"]), set(train["movie_id"])
    rows = []
    for name, frame in splits.items():
        rows.append({
            "split": name, "rows": len(frame), "row_share": len(frame) / total,
            "users": frame["user_id"].nunique(), "movies": frame["movie_id"].nunique(),
            "positive_rows": int(frame["is_relevant"].sum()),
            "positive_share": float(frame["is_relevant"].mean()) if len(frame) else 0.0,
            "cold_users": len(set(frame["user_id"]) - train_users),
            "cold_movies": len(set(frame["movie_id"]) - train_movies),
            "cold_movie_rows": int((~frame["movie_id"].isin(train_movies)).sum()),
            "first_rating_utc": frame["rated_at"].min(), "last_rating_utc": frame["rated_at"].max(),
        })
    return pd.DataFrame(rows).set_index("split")


def validate_splits(
    splits: dict[str, pd.DataFrame], original: pd.DataFrame,
) -> pd.DataFrame:
    """Check conservation, disjointness, tied timestamps and per-user ordering."""
    combined = pd.concat([frame.assign(split=name) for name, frame in splits.items()], ignore_index=True)
    checks = {
        "row_count_difference": abs(len(combined) - len(original)),
        "duplicate_event_ids": int(combined["event_id"].duplicated().sum()),
        "missing_event_ids": len(set(original["event_id"]) - set(combined["event_id"])),
        "unexpected_event_ids": len(set(combined["event_id"]) - set(original["event_id"])),
        "timestamp_groups_crossing_splits": int(
            combined.groupby(["user_id", "timestamp"])["split"].nunique().gt(1).sum()
        ),
        "user_movie_pairs_crossing_splits": int(
            combined.groupby(["user_id", "movie_id"])["split"].nunique().gt(1).sum()
        ),
    }
    for earlier, later in [("train", "validation"), ("validation", "test"), ("train", "test")]:
        boundaries = pd.concat([
            splits[earlier].groupby("user_id")["timestamp"].max().rename("earlier"),
            splits[later].groupby("user_id")["timestamp"].min().rename("later"),
        ], axis=1).dropna()
        checks[f"{earlier}_not_before_{later}"] = int(boundaries["earlier"].ge(boundaries["later"]).sum())
    report = pd.DataFrame(checks.items(), columns=["check", "violations"]).set_index("check")
    if report["violations"].any():
        raise ValueError(f"Invalid data split:\n{report}")
    return report
