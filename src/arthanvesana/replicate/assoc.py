"""Effect-size association measures to complement LLR significance. LLR reacts
strongly to raw frequency; NPMI is bounded in [-1, 1] and comparable across
pairs, while log-Dice is independent of corpus size and therefore comparable
across subcorpora (e.g. sites).
"""

from __future__ import annotations

import math
from collections import Counter

from scipy.stats import fisher_exact

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


bigram_tables = pair_contingency
rank_pairs = ranked_pairs


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p-values must be finite and between 0 and 1")
    n = len(p_values)
    order = sorted(range(n), key=p_values.__getitem__)
    adjusted = [1.0] * n
    running = 1.0
    for i in range(n - 1, -1, -1):
        index = order[i]
        running = min(running, p_values[index] * n / (i + 1))
        adjusted[index] = running
    return adjusted


def analyze_pairs(
    seqs: list[list[str]], *, alpha: float = 0.05, min_count: int = 1, exact: bool = True
) -> dict:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    tables = pair_contingency(seqs)
    rows = []
    for pair, table in tables.items():
        k11, k12, k21, k22, total = table
        p_value = (
            float(fisher_exact([[k11, k12], [k21, k22]], alternative="greater").pvalue)
            if exact else None
        )
        rows.append(
            {
                "pair": pair,
                "count": k11,
                "first_count": k11 + k12,
                "second_count": k11 + k21,
                "total": total,
                "llr": _llr_2x2(*table[:4]),
                "pmi": pmi(table),
                "npmi": npmi(table),
                "logdice": logdice(table),
                "log_odds": log_odds(table),
                "p_value": p_value,
                "p_adj": None,
                "retained": False,
            }
        )
    if exact:
        adjusted = benjamini_hochberg([row["p_value"] for row in rows])
        for row, q in zip(rows, adjusted, strict=True):
            row["p_adj"] = q
            row["retained"] = q <= alpha and row["count"] >= min_count
    rows.sort(key=lambda r: -r["npmi"])
    return {
        "pairs": rows,
        "n_observed": len(tables),
        "n_tested": len(tables) if exact else 0,
        "n_retained": sum(row["retained"] for row in rows),
        "alpha": alpha,
        "min_count": min_count,
    }
