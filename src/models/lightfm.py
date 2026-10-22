"""LightFM item features and portable inference from learned representations."""

from pathlib import Path
import importlib.util
import re
import sys

import numpy as np
from scipy.sparse import csr_matrix, hstack, identity


WINDOWS_RNG_CAST = re.compile(
    r"\(\(\(long\)(?P<call>__pyx_f_\w+_temper\(\(__pyx_v_seed\[0\]\)\))\) / 2\)"
)


def check_training_backend(loss="warp"):
    """Reject the known signed-long RNG conversion before Windows native code runs."""
    if loss == "logistic" or sys.platform != "win32":
        return
    spec = importlib.util.find_spec("lightfm")
    if spec is None:
        raise ImportError("Install LightFM using environment.yml before training.")
    source = Path(spec.origin).parent / "_lightfm_fast_no_openmp.c"
    if source.is_file() and WINDOWS_RNG_CAST.search(source.read_text(encoding="utf-8")):
        raise RuntimeError(
            "This Windows LightFM build contains an unsafe signed-long RNG conversion; "
            "WARP/BPR can crash with 0xC0000005. Use --loss logistic, or install "
            "Microsoft C++ Build Tools and run: python repair_lightfm_windows.py. See README.md."
        )


def build_logistic_interactions(explicit_ratings, relevance_threshold=4):
    """Use observed train scores: >=threshold is +1, 1/2 is -1; omit 3/unknown."""
    interactions = csr_matrix(explicit_ratings, dtype="float32", copy=True)
    interactions.eliminate_zeros()
    if relevance_threshold not in (3, 4, 5) or not np.isin(interactions.data, [1, 2, 3, 4, 5]).all():
        raise ValueError("Expected 1-5 ratings and a positive threshold above 2.")
    interactions.data = np.where(interactions.data >= relevance_threshold, 1,
                                 np.where(interactions.data <= 2, -1, 0)).astype("float32")
    interactions.eliminate_zeros()
    if not (interactions.data > 0).any() or not (interactions.data < 0).any():
        raise ValueError("Logistic training requires observed positive and negative ratings.")
    return interactions


def build_item_features(n_items, genres=None):
    """Keep item identities; optionally append genres and L1-normalize each row."""
    features = identity(n_items, format="csr", dtype="float32")
    if genres is not None:
        genres = csr_matrix(genres, dtype="float32")
        if genres.shape[0] != n_items or not np.isfinite(genres.data).all() or (genres.data < 0).any():
            raise ValueError("Genre features must be finite, nonnegative and aligned with items.")
        features = hstack([features, genres], format="csr")
    return features.multiply(1 / np.asarray(features.sum(axis=1))).tocsr()


class LightFMRecommender:
    """Score training users using fixed fitted embeddings; history tokens are unused.

    Validation history affects test candidate filtering, not the learned user
    vectors. Snapshot copies prevent later fit_partial calls changing a winner.
    """

    def __init__(self, user_ids, user_embeddings, user_biases, item_embeddings, item_biases):
        self.user_ids = np.array(user_ids, dtype="int64", copy=True)
        for name, values in [("user_embeddings", user_embeddings), ("user_biases", user_biases),
                             ("item_embeddings", item_embeddings), ("item_biases", item_biases)]:
            setattr(self, name, np.array(values, dtype="float32", copy=True))
        if (self.user_embeddings.ndim != 2 or self.item_embeddings.ndim != 2
                or self.user_embeddings.shape[1] != self.item_embeddings.shape[1]
                or self.user_embeddings.shape[0] != len(self.user_ids)
                or self.user_biases.shape != (len(self.user_ids),)
                or self.item_biases.shape != (len(self.item_embeddings),)
                or len(np.unique(self.user_ids)) != len(self.user_ids)
                or not all(np.isfinite(getattr(self, name)).all() for name in
                           ["user_embeddings", "user_biases", "item_embeddings", "item_biases"])):
            raise ValueError("Invalid LightFM representation shapes or values.")
        self.n_items_ = len(self.item_embeddings)
        self.user_index = {int(user): index for index, user in enumerate(self.user_ids)}

    @classmethod
    def from_fitted(cls, model, item_features, user_ids):
        user_biases, user_embeddings = model.get_user_representations()
        item_biases, item_embeddings = model.get_item_representations(item_features)
        return cls(user_ids, user_embeddings, user_biases, item_embeddings, item_biases)

    def score_tokens(self, user_id, positive_movie_tokens):
        if user_id not in self.user_index:
            raise ValueError(f"User {user_id} was not present in LightFM training.")
        index = self.user_index[user_id]
        return (np.einsum("ij,j->i", self.item_embeddings, self.user_embeddings[index])
                + self.item_biases + self.user_biases[index])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "representations.npz", user_ids=self.user_ids,
                            user_embeddings=self.user_embeddings, user_biases=self.user_biases,
                            item_embeddings=self.item_embeddings, item_biases=self.item_biases)

    @classmethod
    def load(cls, directory):
        with np.load(Path(directory) / "representations.npz", allow_pickle=False) as saved:
            return cls(**{name: saved[name] for name in saved.files})
