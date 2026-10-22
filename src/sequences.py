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


def validation_windows(train_sequences, validation_sequences, max_length, n_items):
    """Teacher-forced validation: only strictly later validation items are labels.

    Reserve half of each window for prior context when available. Earlier
    validation events may enter later histories, never future events or test.
    Input tables use user_id, movie_tokens and timestamps, as in train sequences.
    """
    histories = {int(row.user_id): row for row in train_sequences.itertuples(index=False)}
    inputs, labels = [], []
    chunk_size = max(1, max_length // 2)
    for row in validation_sequences.itertuples(index=False):
        previous = histories[int(row.user_id)]
        history, history_times = list(previous.movie_tokens), list(previous.timestamps)
        future, future_times = list(row.movie_tokens), list(row.timestamps)
        pad_history(history + future, max_length, n_items)
        times = history_times + future_times
        if (len(history) != len(history_times) or len(future) != len(future_times)
                or np.any(np.diff(times) < 0)):
            raise ValueError("Validation sequences must follow aligned chronological train histories.")
        for start in range(0, len(future), chunk_size):
            stop = min(start + chunk_size, len(future))
            prefix_size = max_length - (stop - start) + 1
            prefix, prefix_times = history[-prefix_size:], history_times[-prefix_size:]
            tokens = np.asarray(prefix + future[start:stop], dtype=np.int64)
            times = np.asarray(prefix_times + future_times[start:stop], dtype=np.int64)
            target = tokens[1:].copy()
            target[:max(0, len(prefix) - 1)] = 0  # Train/prior-context labels are ignored.
            target[np.diff(times) == 0] = 0
            if np.any(target):
                inputs.append(pad_history(tokens[:-1], max_length, n_items))
                y = np.zeros(max_length, dtype=np.int64)
                y[-len(target):] = target
                labels.append(y)
            history.extend(future[start:stop])
            history_times.extend(future_times[start:stop])
    if not inputs:
        raise ValueError("No strictly later warm validation targets are available.")
    return np.stack(inputs), np.stack(labels)
