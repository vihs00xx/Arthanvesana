"""Evaluation metrics beyond raw perplexity and top-1 accuracy. Bootstrap
confidence intervals resample whole inscriptions; MRR uses full rankings;
Brier/ECE measure whether predicted probabilities match observed accuracy;
Jensen-Shannon checks whether model-generated texts reproduce real
distributional shapes; normalized perplexity divides by vocabulary size so
corpora with different sign inventories become comparable.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


def bootstrap_ci(
    seqs: list[T],
    stat: Callable[[list[T]], float],
    n_reps: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(seqs)
    vals = []
    for _ in range(n_reps):
        sample = [seqs[rng.randrange(n)] for _ in range(n)]
        vals.append(stat(sample))
    vals.sort()
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean": sum(vals) / len(vals),
        "lo": vals[int(lo_q * n_reps)],
        "hi": vals[min(int(hi_q * n_reps), n_reps - 1)],
        "reps": n_reps,
    }


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r) / len(ranks)


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal lengths")
    if any(not math.isfinite(c) or not 0.0 <= c <= 1.0 for c in confidences):
        raise ValueError("confidences must be finite probabilities")
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for c, ok in zip(confidences, correct):
        b = min(int(c * bins), bins - 1)
        buckets[b].append((c, 1.0 if ok else 0.0))
    total = len(confidences)
    err = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        acc = math.fsum(ok for _, ok in bucket) / len(bucket)
        conf = math.fsum(c for c, _ in bucket) / len(bucket)
        err += len(bucket) / total * abs(acc - conf)
    return err


def js_divergence(p: Counter[K], q: Counter[K]) -> float:
    keys = set(p) | set(q)
    tp = sum(p.values())
    tq = sum(q.values())
    if tp == 0 or tq == 0:
        return float("nan")
    total = 0.0
    for k in keys:
        a = p.get(k, 0) / tp
        b = q.get(k, 0) / tq
        m = (a + b) / 2.0
        if a > 0.0:
            total += a * math.log2(a / m)
        if b > 0.0:
            total += b * math.log2(b / m)
    return total / 2.0


def normalized_perplexity(ppl: float, vocab_size: int) -> float:
    return ppl / vocab_size
