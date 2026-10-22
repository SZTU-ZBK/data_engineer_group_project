"""Read MovieLens 1M and inspect its raw tables without changing the records."""

from pathlib import Path

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
