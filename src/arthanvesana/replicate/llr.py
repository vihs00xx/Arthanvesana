from __future__ import annotations

import math
from collections import Counter


def _llr_2x2(k11: int, k12: int, k21: int, k22: int) -> float:
    n = k11 + k12 + k21 + k22
    if n == 0:
        return 0.0
    r1 = k11 + k12
    r2 = k21 + k22
    c1 = k11 + k21
    c2 = k12 + k22
    total = 0.0
    for k, e in (
        (k11, r1 * c1 / n),
        (k12, r1 * c2 / n),
        (k21, r2 * c1 / n),
        (k22, r2 * c2 / n),
    ):
        if k > 0 and e > 0:
            total += k * math.log(k / e)
    return 2.0 * total


def bigram_llr(seqs: list[list[str]]) -> list[tuple[tuple, int, float]]:
    first: Counter = Counter()
    second: Counter = Counter()
    pairs: Counter = Counter()
    total = 0
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            first[a] += 1
            second[b] += 1
            pairs[(a, b)] += 1
            total += 1
    rows = []
    for (a, b), c in pairs.items():
        k11 = c
        k12 = first[a] - c
        k21 = second[b] - c
        k22 = total - k11 - k12 - k21
        rows.append(((a, b), c, _llr_2x2(k11, k12, k21, k22)))
    rows.sort(key=lambda r: -r[2])
    return rows


def trigram_llr(seqs: list[list[str]]) -> list[tuple[tuple, int, float]]:
    triples: Counter = Counter()
    contexts: Counter = Counter()
    follows: Counter = Counter()
    for seq in seqs:
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            triples[(a, b, c)] += 1
            contexts[(a, b)] += 1
            follows[(b, c)] += 1
    pair_ctx: Counter = Counter()
    for seq in seqs:
        for a, b in zip(seq, seq[1:]):
            pair_ctx[(a, b)] += 1
    b_ctx: Counter = Counter()
    for (a, b), v in pair_ctx.items():
        b_ctx[b] += v
    rows = []
    for (a, b, c), t in triples.items():
        k11 = t
        k12 = contexts[(a, b)] - t
        k21 = follows[(b, c)] - t
        k22 = b_ctx[b] - k11 - k12 - k21
        rows.append(((a, b, c), t, _llr_2x2(k11, k12, k21, k22)))
    rows.sort(key=lambda r: -r[2])
    return rows
