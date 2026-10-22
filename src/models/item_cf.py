"""Positive-feedback ItemCF with cosine similarity and top-k item neighbors."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


class ItemCF:
    """Learn source-item -> candidate-item similarities from training positives.

    similarity(i,j) = cooccurrence(i,j) / sqrt(support(i)*support(j)).
    Optional shrinkage multiplies this by cooccurrence/(cooccurrence+shrinkage).
    Self similarities are zero. Keep n_neighbors per SOURCE item; truncation can
    make the matrix asymmetric. Scores sum similarities from the user's known
    positive items. Scores are ranking values, not predicted 1-5 star ratings.
    """

    def __init__(self, n_neighbors: int = 50, shrinkage: float = 0.0):
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be a finite nonnegative number.")
        self.n_neighbors = n_neighbors
        self.shrinkage = float(shrinkage)
        self.similarity_: csr_matrix | None = None

    def fit(self, positive_interactions: csr_matrix) -> "ItemCF":
        start = perf_counter()
        interactions = csr_matrix(positive_interactions, dtype="float32", copy=True)
        interactions.sum_duplicates()
        interactions.eliminate_zeros()
        if not interactions.shape[0] or not interactions.shape[1] or not interactions.nnz:
            raise ValueError("Provide a nonempty positive-interaction matrix.")
        if not np.all(interactions.data == 1):
            raise ValueError("ItemCF expects binary positive interactions, with nonzeros equal to one.")
        cooccurrence = (interactions.T @ interactions).tocsr()
        support = cooccurrence.diagonal().copy()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        indices, weights, indptr = [], [], [0]
        for source in range(interactions.shape[1]):
            start_pos, end_pos = cooccurrence.indptr[source:source + 2]
            neighbors = cooccurrence.indices[start_pos:end_pos]
            counts = cooccurrence.data[start_pos:end_pos]
            similarities = counts / np.sqrt(support[source] * support[neighbors])
            if self.shrinkage:
                similarities = similarities * counts / (counts + self.shrinkage)
            # Equal similarities resolve to the lower item INDEX. The shared
            # mapping is sorted by raw movie ID, making neighbor selection stable.
            selected = np.lexsort((neighbors, -similarities))[:self.n_neighbors]
            indices.append(neighbors[selected])
            weights.append(similarities[selected])
            indptr.append(indptr[-1] + len(selected))
        self.similarity_ = csr_matrix(
            (np.concatenate(weights), np.concatenate(indices), np.asarray(indptr)),
            shape=(interactions.shape[1], interactions.shape[1]), dtype="float32",
        )
        self.similarity_.sort_indices()
        self.n_users_ = interactions.shape[0]
        self.n_items_ = interactions.shape[1]
        self.supported_items_ = int(np.count_nonzero(support))
        self.fit_seconds_ = perf_counter() - start
        return self

    def score(self, history_item_indices) -> np.ndarray:
        """Score all items using zero-based indexes of known positive history."""
        if self.similarity_ is None:
            raise RuntimeError("Fit or load the model before scoring.")
        history = np.asarray(history_item_indices, dtype="int64")
        if history.ndim != 1 or np.any(history < 0) or np.any(history >= self.n_items_):
            raise ValueError("History must contain valid zero-based item indexes.")
        history = np.unique(history)
        if not len(history):
            return np.zeros(self.n_items_, dtype="float32")
        return np.asarray(self.similarity_[history].sum(axis=0)).ravel().astype("float32")

    def score_tokens(self, user_id: int, positive_movie_tokens: list[int]) -> np.ndarray:
        """Adapter for shared contexts; token zero is reserved for padding."""
        tokens = np.asarray(positive_movie_tokens, dtype="int64")
        if np.any(tokens < 0):
            raise ValueError("Movie tokens must be nonnegative.")
        return self.score(tokens[tokens > 0] - 1)

    def save(self, directory: str | Path) -> None:
        if self.similarity_ is None:
            raise RuntimeError("Cannot save an unfitted model.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / "similarity.npz", self.similarity_)
        metadata = {
            "algorithm": "positive_feedback_itemcf", "n_neighbors": self.n_neighbors,
            "shrinkage": self.shrinkage, "n_users": self.n_users_, "n_items": self.n_items_,
            "supported_items": self.supported_items_, "fit_seconds": self.fit_seconds_,
        }
        (directory / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "ItemCF":
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        model = cls(metadata["n_neighbors"], metadata["shrinkage"])
        model.similarity_ = load_npz(directory / "similarity.npz").tocsr()
        model.n_users_, model.n_items_ = metadata["n_users"], metadata["n_items"]
        model.supported_items_, model.fit_seconds_ = metadata["supported_items"], metadata["fit_seconds"]
        if model.similarity_.shape != (model.n_items_, model.n_items_):
            raise ValueError("Saved model shape does not match its metadata.")
        return model
