"""Training-only next-item windows; zero is reserved for padding/ignored labels."""

import numpy as np


def pad_history(tokens, max_length, n_items):
    tokens = np.asarray(tokens, dtype=np.int64)
    if (max_length < 1 or tokens.ndim != 1
            or np.any((tokens < 1) | (tokens > n_items))):
        raise ValueError("Expected positive catalog tokens and a positive sequence length.")
    result = np.zeros(max_length, dtype=np.int64)
    recent = tokens[-max_length:]
    if len(recent):
        result[-len(recent):] = recent
    return result


def training_windows(sequences, max_length, n_items):
    """Cover every adjacent train transition once, excluding tied-time targets.

    Nonoverlapping input chunks bound memory. Each target is the following item;
    no invented order within a timestamp group is used as a training label.
    """
    inputs, labels = [], []
    for row in sequences.itertuples(index=False):
        tokens = np.asarray(row.movie_tokens, dtype=np.int64)
        times = np.asarray(row.timestamps, dtype=np.int64)
        pad_history(tokens, max_length, n_items)  # Validate the entire sequence.
        if len(tokens) != len(times) or np.any(np.diff(times) < 0):
            raise ValueError("Tokens and chronological timestamps must align.")
        for start in range(0, len(tokens) - 1, max_length):
            stop = min(start + max_length, len(tokens) - 1)
            target = tokens[start + 1:stop + 1].copy()
            target[times[start + 1:stop + 1] == times[start:stop]] = 0
            if not np.any(target):
                continue
            x = pad_history(tokens[start:stop], max_length, n_items)
            y = np.zeros(max_length, dtype=np.int64)
            y[-len(target):] = target
            inputs.append(x)
            labels.append(y)
    if not inputs:
        raise ValueError("No strictly later next-item training targets are available.")
    return np.stack(inputs), np.stack(labels)
