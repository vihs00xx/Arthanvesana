"""Greedy segmentation after Sinha et al.: repeatedly merge the adjacent pair
with the highest association score. Fast-collapsing trees indicate reusable
sub-units combining into longer texts.
"""

from __future__ import annotations


def greedy_segmentation(
    seq: list[str], pair_score: dict[tuple, float], min_score: float = 0.0
) -> tuple[list, int]:
    tokens = [(s,) for s in seq]
    rounds = 0
    while len(tokens) > 1:
        best = None
        best_score = min_score
        for i in range(len(tokens) - 1):
            s = pair_score.get((tokens[i][-1], tokens[i + 1][0]), 0.0)
            if s > best_score:
                best_score = s
                best = i
        if best is None:
            break
        tokens = tokens[:best] + [tokens[best] + tokens[best + 1]] + tokens[best + 2 :]
        rounds += 1
    return tokens, rounds


def segmentation_heights(
    seqs: list[list[str]], pair_score: dict[tuple, float]
) -> list[tuple[int, int]]:
    return [(len(s), greedy_segmentation(s, pair_score)[1]) for s in seqs]
