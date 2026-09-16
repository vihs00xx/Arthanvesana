"""Effect-size association measures to complement LLR significance. LLR reacts
strongly to raw frequency; NPMI is bounded in [-1, 1] and comparable across
pairs, while log-Dice is independent of corpus size and therefore comparable
across subcorpora (e.g. sites).
"""

from __future__ import annotations

import math
from collections import Counter

from .llr import _llr_2x2


def pair_contingency(seqs: list[list[str]]) -> dict[tuple, tuple]:
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
    out = {}
    for (a, b), c in pairs.items():
        k11 = c
        k12 = first[a] - c
        k21 = second[b] - c
        k22 = total - k11 - k12 - k21
        out[(a, b)] = (k11, k12, k21, k22, total)
    return out


def pmi(table: tuple) -> float:
    k11, k12, k21, k22, n = table
    p_ab = k11 / n
    p_a = (k11 + k12) / n
    p_b = (k11 + k21) / n
    if p_ab <= 0.0 or p_a <= 0.0 or p_b <= 0.0:
        return 0.0
    return math.log2(p_ab / (p_a * p_b))


def npmi(table: tuple) -> float:
    k11, _, _, _, n = table
    p_ab = k11 / n
    if p_ab <= 0.0 or p_ab >= 1.0:
        return 0.0
    return pmi(table) / (-math.log2(p_ab))


def logdice(table: tuple) -> float:
    k11, k12, k21, _, _ = table
    f_a = k11 + k12
    f_b = k11 + k21
    if f_a + f_b == 0:
        return float("-inf")
    dice = 2 * k11 / (f_a + f_b)
    if dice <= 0.0:
        return float("-inf")
    return 14 + math.log2(dice)


def log_odds(table: tuple) -> float:
    k11, k12, k21, k22, _ = table
    a, b, c, d = k11 + 0.5, k12 + 0.5, k21 + 0.5, k22 + 0.5
    return math.log2(a * d / (b * c))


def ranked_pairs(
    seqs: list[list[str]], min_count: int = 3, llr_min: float = 10.83
) -> list[dict]:
    rows = []
    for pair, table in pair_contingency(seqs).items():
        if table[0] < min_count:
            continue
        llr = _llr_2x2(*table[:4])
        if llr < llr_min:
            continue
        rows.append(
            {
                "pair": pair,
                "count": table[0],
                "llr": llr,
                "npmi": npmi(table),
                "logdice": logdice(table),
                "log_odds": log_odds(table),
            }
        )
    rows.sort(key=lambda r: -r["npmi"])
    return rows
