"""Stable observed-feedback logistic loss for LightFM capacity experiments."""

import numpy as np
from scipy.sparse import coo_matrix


def observed_logistic_loss(model, interactions, batch_size=32768):
    """Mean log(1 + exp(-y*s)) over observed +/-1 labels; excludes unknowns.

    This is data loss, without the L2 penalty. Indexes must match the exported
    user/movie representation rows. Batching bounds temporary embedding copies.
    """
    events = coo_matrix(interactions, copy=True)
    events.eliminate_zeros()
    if (batch_size < 1 or not events.nnz or not np.isin(events.data, [-1, 1]).all()
            or events.shape != (len(model.user_embeddings), len(model.item_embeddings))):
        raise ValueError("Provide aligned nonempty +/-1 labels and a positive batch size.")
    sums = {1: 0.0, -1: 0.0}
    counts = {1: 0, -1: 0}
    for start in range(0, events.nnz, batch_size):
        stop = min(start + batch_size, events.nnz)
        users, items, labels = events.row[start:stop], events.col[start:stop], events.data[start:stop]
        scores = (np.einsum("ij,ij->i", model.user_embeddings[users], model.item_embeddings[items])
                  + model.user_biases[users] + model.item_biases[items]).astype("float64")
        losses = np.logaddexp(0.0, -labels * scores)
        if not np.isfinite(losses).all():
            raise ValueError("Nonfinite logistic loss.")
        for label in (1, -1):
            mask = labels == label
            sums[label] += float(losses[mask].sum())
            counts[label] += int(mask.sum())
    return {"loss": (sums[1] + sums[-1]) / events.nnz,
            "positive_loss": sums[1] / counts[1] if counts[1] else None,
            "negative_loss": sums[-1] / counts[-1] if counts[-1] else None,
            "positive_count": counts[1], "negative_count": counts[-1]}


def plot_capacity_curves(search, output_path):
    """Plot data loss and validation ranking; choose each dimension's alpha on validation."""
    from pathlib import Path
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for dimensions, group in search.groupby("no_components", sort=True):
        best = group.sort_values(["NDCG@10", "Recall@10", "epochs", "alpha"],
                                 ascending=[False, False, True, True]).iloc[0]
        curve = group.loc[group.alpha.eq(best.alpha)].sort_values("epochs")
        for axis, column in zip(axes, ["train_loss", "validation_loss", "NDCG@10"]):
            axis.plot(curve.epochs, curve[column], marker="o", markersize=3, label=f"{dimensions} dimensions")
    for axis, title, label in zip(axes, ["Training data loss", "Validation data loss", "Validation ranking"],
                                  ["Logistic cross entropy", "Logistic cross entropy", "NDCG@10"]):
        axis.set(title=title, xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    return figure
