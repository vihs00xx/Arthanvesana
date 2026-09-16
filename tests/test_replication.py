import math
from collections import Counter

import numpy as np

from arthanvesana.replicate.llr import bigram_llr, trigram_llr
from arthanvesana.replicate.region import cross_perplexity
from arthanvesana.replicate.restore import restoration_accuracy, restore_rank
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.replicate.segment import greedy_segmentation
from arthanvesana.replicate.zipf import (
    beginner_ender_coverage,
    coverage,
    mandelbrot_fit,
    rank_frequencies,
)


def test_bigram_llr_perfect_association():
    seqs = [["a", "b"]] * 10 + [["c", "d"]] * 10
    rows = bigram_llr(seqs)
    assert len(rows) == 2
    assert rows[0][0] == ("a", "b") or rows[0][0] == ("c", "d")
    assert abs(rows[0][2] - 40 * math.log(2)) < 1e-9


def test_bigram_llr_independence_near_zero():
    seqs = (
        [["a", "b"]] * 10
        + [["a", "c"]] * 10
        + [["d", "b"]] * 10
        + [["d", "c"]] * 10
    )
    for _, _, llr in bigram_llr(seqs):
        assert abs(llr) < 1e-9


def test_trigram_llr_sorted_nonnegative():
    seqs = [["a", "b", "c"]] * 5 + [["x", "b", "c"]] * 5 + [["a", "b", "z"]] * 5
    rows = trigram_llr(seqs)
    assert all(llr >= 0 for _, _, llr in rows)
    vals = [llr for _, _, llr in rows]
    assert vals == sorted(vals, reverse=True)


def test_coverage():
    assert coverage(Counter({"a": 80, "b": 10, "c": 10})) == 1
    assert coverage(Counter({"a": 50, "b": 30, "c": 20})) == 2


def test_beginner_ender_coverage():
    seqs = [["a", "b"], ["a", "c"], ["b", "c"]]
    out = beginner_ender_coverage(seqs)
    assert out["beginner_80"] == 2
    assert out["ender_80"] == 2


def test_mandelbrot_fit_recovers_power_law():
    ranks = np.arange(1, 51)
    freqs = 10000.0 * ranks ** (-1.2)
    fit = mandelbrot_fit(freqs)
    assert fit["r_squared"] > 0.99
    assert abs(fit["b"] - 1.2) < 0.05


def test_rank_frequencies_sorted():
    f = rank_frequencies([["a", "b"], ["a"]])
    assert list(f) == [2.0, 1.0]


def test_restoration_perfect_toy():
    train = [["a", "b"]] * 10
    out = restoration_accuracy(train, [["a", "b"]])
    assert out["top_1"] == 1.0
    assert out["n_masked"] == 2


def test_restore_rank_positions():
    model = NGramModel([["a", "b", "c"]] * 5, 2, method="wittenbell")
    mats = {ctx: model.dist((ctx,)) for ctx in list(model.vocab) + ["<S>"]}
    assert restore_rank(mats, model.vocab, ["a", "b", "c"], 1) == 1
    assert restore_rank(mats, model.vocab, ["a", "b", "c"], 0) == 1
    assert restore_rank(mats, model.vocab, ["zzz"], 0) is None


def test_cross_perplexity_self_better():
    sites = {"s1": [["a", "b"]] * 10, "s2": [["c", "d"]] * 10}
    m = cross_perplexity(sites)
    assert set(m) == {"s1", "s2"}
    assert m["s1"]["s1"] < m["s1"]["s2"]
    assert m["s2"]["s2"] < m["s2"]["s1"]


def test_greedy_segmentation_merges_best_pair():
    tokens, rounds = greedy_segmentation(["a", "b", "c"], {("a", "b"): 10.0})
    assert rounds == 1
    assert tokens == [("a", "b"), ("c",)]


def test_greedy_segmentation_no_score_no_merge():
    tokens, rounds = greedy_segmentation(["a", "b"], {})
    assert rounds == 0
    assert tokens == [("a",), ("b",)]
