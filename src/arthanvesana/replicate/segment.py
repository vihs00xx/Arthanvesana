"""Greedy segmentation after Sinha et al.: repeatedly merge the adjacent pair
with the highest association score. Fast-collapsing trees indicate reusable
sub-units combining into longer texts.
"""

from __future__ import annotations


def greedy_segmentation(
    seq: list[str], pair_score: dict[tuple[str, str], float], min_score: float = 0.0
) -> tuple[list[tuple[str, ...]], int]:
    tokens: list[tuple[str, ...]] = [(s,) for s in seq]
    merge_count = 0
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
        merge_count += 1
    return tokens, merge_count


def segmentation_merge_counts(
    seqs: list[list[str]], pair_score: dict[tuple[str, str], float]
) -> list[tuple[int, int]]:
    return [(len(s), greedy_segmentation(s, pair_score)[1]) for s in seqs]


segmentation_heights = segmentation_merge_counts
