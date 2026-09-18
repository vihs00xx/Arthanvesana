"""Small-sample entropy estimators. Plugin entropy underestimates when samples
are scarce relative to vocabulary; Chao-Shen corrects via sample coverage
(singleton rate) and James-Stein shrinkage pulls frequencies toward uniform.
All values in bits.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def chao_shen(counts: Counter[K]) -> float:
    if any(not math.isfinite(v) or v < 0 or v != int(v) for v in counts.values()):
        raise ValueError("counts must be finite nonnegative integers")
    n = sum(counts.values())
    if n == 0:
        return 0.0
    singletons = sum(1 for v in counts.values() if v == 1)
    if singletons == n:
        raise ValueError("Chao-Shen estimate is unsupported for all-singleton samples")
    coverage = (n - singletons) / n
    terms = []
    for v in counts.values():
        p = coverage * (v / n)
        if p == 0.0 or p == 1.0:
            continue
        denom = -math.expm1(n * math.log1p(-p))
        terms.append(-p * math.log2(p) / denom)
    return math.fsum(terms)


def shrinkage_entropy(counts: Counter[K]) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    types = list(counts)
    p = len(types)
    ml = {k: counts[k] / n for k in types}
    target = 1.0 / p
    if n <= 1:
        return -math.log2(target)
    var_sum = sum(v * (1.0 - v) / (n - 1) for v in ml.values())
    sq_sum = sum((target - v) ** 2 for v in ml.values())
    lam = min(var_sum / sq_sum, 1.0) if sq_sum else 1.0
    shrunk = {k: lam * target + (1.0 - lam) * v for k, v in ml.items()}
    return -sum(v * math.log2(v) for v in shrunk.values() if v > 0.0)


def unigram_counts(seqs: list[list[str]]) -> Counter[str]:
    return Counter(s for seq in seqs for s in seq)
