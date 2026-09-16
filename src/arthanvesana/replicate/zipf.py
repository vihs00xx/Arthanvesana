"""Rank-frequency analysis. Fits f(r) = a*(r+c)^-b (Zipf-Mandelbrot) and
reports how many distinct signs cover 80% of inscription starts vs ends.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def rank_frequencies(seqs: list[list[str]]) -> np.ndarray:
    counts = Counter(s for seq in seqs for s in seq)
    return np.array(sorted(counts.values(), reverse=True), dtype=float)


def mandelbrot_fit(freqs: np.ndarray) -> dict:
    from scipy.optimize import curve_fit

    ranks = np.arange(1, len(freqs) + 1, dtype=float)

    def model(r, a, b, c):
        return a * np.power(r + c, -b)

    popt, _ = curve_fit(
        model, ranks, freqs, p0=(freqs[0], 1.0, 1.0), maxfev=20000
    )
    pred = model(ranks, *popt)
    ss_res = float(np.sum((freqs - pred) ** 2))
    ss_tot = float(np.sum((freqs - freqs.mean()) ** 2))
    return {
        "a": float(popt[0]),
        "b": float(popt[1]),
        "c": float(popt[2]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
        "predicted": pred,
    }


def coverage(counts: Counter, frac: float = 0.8) -> int:
    total = sum(counts.values())
    acc = 0
    for i, v in enumerate(sorted(counts.values(), reverse=True), start=1):
        acc += v
        if acc / total >= frac:
            return i
    return len(counts)


def beginner_ender_coverage(seqs: list[list[str]]) -> dict:
    beginners: Counter = Counter()
    enders: Counter = Counter()
    for seq in seqs:
        if not seq:
            continue
        beginners[seq[0]] += 1
        enders[seq[-1]] += 1
    return {
        "beginner_80": coverage(beginners),
        "ender_80": coverage(enders),
        "n_beginners": len(beginners),
        "n_enders": len(enders),
    }
