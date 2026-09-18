"""Per-sign positional profiles. Counting is slot-based: a length-1
inscription contributes its sign to both 'begin' and 'end', so fractions
always sum to 1.
"""

from __future__ import annotations

from collections import Counter


def positional_counts(seqs: list[list[str]]) -> dict[str, Counter]:
    pos: dict[str, Counter] = {
        "begin": Counter(),
        "middle": Counter(),
        "end": Counter(),
    }
    for seq in seqs:
        if not seq:
            continue
        pos["begin"][seq[0]] += 1
        pos["end"][seq[-1]] += 1
        for s in seq[1:-1]:
            pos["middle"][s] += 1
    return pos


def positional_profile(
    seqs: list[list[str]], min_count: int = 5
) -> list[dict]:
    pos = positional_counts(seqs)
    totals: Counter = Counter()
    for counter in pos.values():
        totals.update(counter)
    rows = []
    for sign, total in totals.items():
        if total < min_count:
            continue
        b = pos["begin"][sign]
        m = pos["middle"][sign]
        e = pos["end"][sign]
        rows.append(
            {
                "sign": sign,
                "count": total,
                "p_begin": b / total,
                "p_middle": m / total,
                "p_end": e / total,
            }
        )
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows
