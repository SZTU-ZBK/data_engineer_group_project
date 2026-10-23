"""Load saved recommenders and display movie titles and genres."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, SCHEMAS, resolve_data_dir
from src.evaluate import rank_candidates
from src.models.item_cf import ItemCF
from src.models.lightfm import LightFMRecommender


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_movie_metadata() -> pd.DataFrame:
    return pd.read_csv(
        resolve_data_dir() / "movies.dat", sep="::", engine="python", header=None,
        names=list(SCHEMAS["movies"]), dtype=SCHEMAS["movies"], encoding="latin-1",
        keep_default_na=False, na_values=[""],
    )


def load_user_metadata() -> pd.DataFrame:
    return pd.read_csv(
        resolve_data_dir() / "users.dat", sep="::", engine="python", header=None,
        names=list(SCHEMAS["users"]), dtype=SCHEMAS["users"], encoding="latin-1",
        keep_default_na=False, na_values=[""],
    )


def recommendation_table(model, context, movie_ids, movies: pd.DataFrame, k=10) -> pd.DataFrame:
    """Rank unseen items; ItemCF additionally explains the strongest history edge."""
    movie_ids = np.asarray(movie_ids, dtype="int64")
    tokens = np.asarray(context["positive_movie_tokens"], dtype="int64")
    history = np.unique(tokens[tokens > 0] - 1)
    scores = (model.score(history) if isinstance(model, ItemCF)
              else model.score_tokens(int(context["user_id"]), context["positive_movie_tokens"]))
    ranked_ids, ranked_scores = rank_candidates(scores, movie_ids, context["seen_movie_ids"], k)
    movie_to_index = {int(movie): index for index, movie in enumerate(movie_ids)}
    metadata = movies.set_index("movie_id")
    records = []
    for rank, (movie, score) in enumerate(zip(ranked_ids, ranked_scores), start=1):
        record = {
            "rank": rank, "movie_id": int(movie), "title": str(metadata.loc[movie, "title"]),
            "genres": str(metadata.loc[movie, "genres"]), "score": float(score),
        }
        if not isinstance(model, ItemCF):
            records.append(record)
            continue
        reason_id, reason_title, reason_similarity = None, "No contributing positive-history neighbor", 0.0
        if len(history):
            contributions = model.similarity_[history, movie_to_index[int(movie)]].toarray().ravel()
            if np.any(contributions > 0):
                strongest = np.lexsort((movie_ids[history], -contributions))[0]
                reason_id = int(movie_ids[history[strongest]])
                reason_title = str(metadata.loc[reason_id, "title"])
                reason_similarity = float(contributions[strongest])
        record.update({
            "because_movie_id": reason_id, "because_title": reason_title,
            "because_similarity": reason_similarity,
        })
        records.append(record)
    return pd.DataFrame(records)


def recommend_for_user(
    user_id: int, stage: str = "test", k: int = 10,
    processed_dir: str | Path | None = None, model_dir: str | Path | None = None,
    algorithm: str = "itemcf",
) -> pd.DataFrame:
    """Use the selected saved model and the specified simulated evaluation history."""
    if stage not in ("validation", "test"):
        raise ValueError("Choose validation or test history.")
    if algorithm not in ("itemcf", "lightfm", "lrurec"):
        raise ValueError("Choose itemcf, lightfm or lrurec.")
    processed_dir = Path(processed_dir) if processed_dir else PROJECT_ROOT / "data" / "processed"
    model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "outputs" / "models" / algorithm
    if not (model_dir / "run.json").is_file():
        raise FileNotFoundError(f"Run python train.py --model {algorithm} before requesting recommendations.")
    run = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    required = ["config.json", "movie_mapping.csv", "positive_interactions.npz", f"{stage}_contexts.jsonl"]
    if algorithm == "lightfm":
        required += ["user_mapping.csv", "genre_features.npz"]
        if run["loss"] == "logistic":
            required.append("explicit_ratings.npz")
    elif algorithm == "lrurec":
        required.append("train_sequences.jsonl")
    for name in required:
        if file_sha256(processed_dir / name) != run["input_sha256"][name]:
            raise ValueError(f"Preprocessed file changed: {name}. Rerun the baseline to match the new inputs.")
    mapping = pd.read_csv(processed_dir / "movie_mapping.csv").sort_values("movie_index")
    movie_ids = mapping["movie_id"].to_numpy(dtype="int64")
    contexts = pd.read_json(processed_dir / f"{stage}_contexts.jsonl", lines=True)
    matches = contexts.loc[contexts["user_id"].eq(user_id)]
    if matches.empty:
        raise ValueError(f"User {user_id} has no {stage} context; choose a user listed in that split.")
    if algorithm == "lrurec":
        from src.models.lrurec import LRURec
        model = LRURec.load(model_dir)
    else:
        model = (ItemCF if algorithm == "itemcf" else LightFMRecommender).load(model_dir)
    if len(movie_ids) != model.n_items_:
        raise ValueError("Movie mapping and saved model have incompatible shapes.")
    return recommendation_table(model, matches.iloc[0], movie_ids, load_movie_metadata(), k)
