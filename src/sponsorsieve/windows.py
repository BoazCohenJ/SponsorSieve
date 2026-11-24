"""Shared windowing between training and inference.

A transcript is a list of segments. Each segment's text is tokenized on its own and the token
ids are concatenated, remembering which segment every token came from. The long sequence is cut
into overlapping windows that fit the model. At inference the per-token probabilities from all
windows are averaged back into one probability per segment.
"""
from __future__ import annotations

from typing import Callable, Sequence

LABELS = ["content", "ad", "self_promo"]
LABEL_IDS = {None: 0, "sponsor": 1, "promo": 1, "ad": 1, "self_promo": 2}


def flatten(segments: Sequence[dict], encode: Callable[[str], list[int]]) -> tuple[list[int], list[int]]:
    """Return (token_ids, segment_index_per_token)."""
    ids, owner = [], []
    for i, s in enumerate(segments):
        t = encode(" " + s["text"].strip())
        ids.extend(t)
        owner.extend([i] * len(t))
    return ids, owner


def window_starts(n: int, size: int, stride: int) -> list[int]:
    if n <= size:
        return [0]
    starts = list(range(0, n - size, stride))
    starts.append(n - size)
    return starts


def segment_probs(owner: list[int], token_probs: list[list[float]], n_segments: int) -> list[list[float]]:
    """Average token class probabilities into per-segment probabilities."""
    k = len(token_probs[0]) if token_probs else len(LABELS)
    sums = [[0.0] * k for _ in range(n_segments)]
    counts = [0] * n_segments
    for seg, p in zip(owner, token_probs):
        counts[seg] += 1
        row = sums[seg]
        for j in range(k):
            row[j] += p[j]
    out = []
    for row, c in zip(sums, counts):
        out.append([v / c for v in row] if c else [1.0] + [0.0] * (k - 1))
    return out
