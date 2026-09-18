"""Unigram and conditional entropy. Context counts are taken over context
positions only (a sequence-final token never serves as a context), with an
optional Miller-Madow bias correction on the conditional estimate.
"""

from __future__ import annotations

import math
from collections import Counter

from .ngrams import count_ngrams


def unigram_entropy(seqs: list[list[str]]) -> float:
    counts: Counter[str] = Counter(s for seq in seqs for s in seq)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counts.values())


def conditional_entropy(seqs: list[list[str]], order: int) -> tuple[float, float]:
    if order < 0:
        raise ValueError("order must be nonnegative")
    n = order + 1
    ng = count_ngrams(seqs, n)
    ctx: Counter[tuple[str, ...]] = Counter()
    for gram, count in ng.items():
        ctx[gram[:-1]] += count
    total = sum(ng.values())
    if total == 0:
        return 0.0, 0.0
    plugin = -sum(
        (v / total) * math.log2(v / ctx[gram[:-1]]) for gram, v in ng.items()
    )
    miller_madow = plugin + (len(ng) - len(ctx)) / (2 * total * math.log(2))
    return plugin, miller_madow


def mutual_information(seqs: list[list[str]], order: int = 1) -> float:
    if order < 0:
        raise ValueError("order must be nonnegative")
    ng = count_ngrams(seqs, order + 1)
    total = sum(ng.values())
    if total == 0:
        return 0.0
    contexts: Counter[tuple[str, ...]] = Counter()
    outcomes: Counter[str] = Counter()
    for gram, count in ng.items():
        contexts[gram[:-1]] += count
        outcomes[gram[-1]] += count
    return math.fsum(
        (count / total)
        * math.log2(count * total / (contexts[gram[:-1]] * outcomes[gram[-1]]))
        for gram, count in ng.items()
    )
