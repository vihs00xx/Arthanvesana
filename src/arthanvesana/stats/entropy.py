"""Unigram and conditional entropy. Context counts are taken over context
positions only (a sequence-final token never serves as a context), with an
optional Miller-Madow bias correction on the conditional estimate.
"""

from __future__ import annotations

import math
from collections import Counter

from .ngrams import count_ngrams


def unigram_entropy(seqs: list[list[str]]) -> float:
    counts: Counter = Counter(s for seq in seqs for s in seq)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counts.values())


def conditional_entropy(seqs: list[list[str]], order: int) -> tuple[float, float]:
    n = order + 1
    ng = count_ngrams(seqs, n)
    ctx: Counter = Counter()
    for seq in seqs:
        for i in range(len(seq) - order):
            ctx[tuple(seq[i : i + order])] += 1
    total = sum(ng.values())
    if total == 0:
        return 0.0, 0.0
    plugin = -sum(
        (v / total) * math.log2(v / ctx[gram[:-1]]) for gram, v in ng.items()
    )
    miller_madow = plugin + (len(ng) - len(ctx)) / (2 * total)
    return plugin, miller_madow
