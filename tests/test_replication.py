import math
from collections import Counter

import numpy as np
import pytest

from arthanvesana.replicate.assoc import (
    analyze_pairs,
    benjamini_hochberg,
    bigram_tables,
    logdice,
    npmi,
    pair_contingency,
    pmi,
    rank_pairs,
    ranked_pairs,
)
from arthanvesana.replicate.llr import bigram_llr, trigram_llr
from arthanvesana.replicate.region import cross_perplexity
from arthanvesana.replicate.restore import restoration_accuracy, restore_rank
from arthanvesana.replicate.segment import (
    greedy_segmentation,
    segmentation_heights,
    segmentation_merge_counts,
)
from arthanvesana.replicate.zipf import (
    beginner_ender_coverage,
    coverage,
    mandelbrot_fit,
    rank_frequencies,
)
from arthanvesana.stats.ngrams import NGramModel


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


def test_trigram_llr_ignores_nontriple_windows():
    seqs = [["a", "b", "c"]] * 10 + [["x", "b", "z"]] * 10
    expected = 40 * math.log(2)
    rows = trigram_llr(seqs)
    assert len(rows) == 2
    for _, count, score in rows:
        assert count == 10
        assert score == pytest.approx(expected)
    assert trigram_llr(seqs + [["a", "b"]] * 100 + [["b"], []]) == rows


def test_trigram_llr_conditions_on_middle_sign():
    seqs = [[a, "b", c] for a in ("a", "x") for c in ("c", "z")]
    seqs += [["a", "other", "c"]] * 40
    assert all(score == pytest.approx(0.0, abs=1e-12) for _, _, score in trigram_llr(seqs))


def test_trigram_llr_overlapping_windows_match_contingency():
    seqs = [["a", "b", "c", "b", "a", "b"]]
    windows = [["a", "b", "c"], ["b", "c", "b"], ["c", "b", "a"], ["b", "a", "b"]]
    assert trigram_llr(seqs) == trigram_llr(windows)
    by_gram = {gram: score for gram, _, score in trigram_llr(seqs)}
    assert by_gram[("a", "b", "c")] == pytest.approx(4 * math.log(2))


@pytest.mark.parametrize("seqs", [[], [[]], [["a"]], [["a", "b"]]])
def test_trigram_llr_no_triples(seqs):
    assert trigram_llr(seqs) == []


def test_segmentation_reports_merge_count_not_tree_height():
    seq = ["a", "b", "c", "d"]
    scores = {("a", "b"): 3.0, ("c", "d"): 2.0, ("b", "c"): 1.0}
    tokens, merge_count = greedy_segmentation(seq, scores)
    assert tokens == [("a", "b", "c", "d")]
    assert merge_count == 3
    seqs = [[], ["a"], seq]
    assert segmentation_merge_counts(seqs, scores) == [(0, 0), (1, 0), (4, 3)]
    assert segmentation_heights(seqs, scores) == segmentation_merge_counts(seqs, scores)


def test_segmentation_threshold_and_ties_are_deterministic():
    seq = ["a", "b", "c"]
    scores = {("a", "b"): 2.0, ("b", "c"): 2.0}
    assert greedy_segmentation(seq, scores, min_score=2.0) == ([("a",), ("b",), ("c",)], 0)
    assert greedy_segmentation(seq, scores) == ([("a", "b", "c")], 2)
    assert greedy_segmentation([], scores) == ([], 0)
    assert seq == ["a", "b", "c"]


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("shift", [0.0, 2.5])
def test_mandelbrot_fit_stays_in_valid_power_domain(shift):
    ranks = np.arange(1, 51, dtype=float)
    freqs = 10000.0 * (ranks + shift) ** (-1.2)
    with np.errstate(invalid="raise", over="raise", divide="raise"):
        fit = mandelbrot_fit(freqs)
    assert all(math.isfinite(fit[key]) and fit[key] >= 0 for key in ("a", "b", "c"))
    assert np.all(np.isfinite(fit["predicted"]))
    assert np.all(fit["predicted"] > 0)
    assert np.all(np.diff(fit["predicted"]) <= 0)
    assert fit["r_squared"] > 0.999
    np.testing.assert_allclose(fit["predicted"], freqs, rtol=0.01)


@pytest.mark.parametrize(
    "freqs",
    [[], [1], [2, 1], [[3, 2, 1]], [3, 2, 0], [3, 2, -1],
     [3, 2, float("nan")], [float("inf"), 2, 1]],
)
def test_mandelbrot_fit_rejects_unsupported_data(freqs):
    with pytest.raises(ValueError):
        mandelbrot_fit(np.array(freqs))


def test_mandelbrot_fit_constant_frequencies():
    fit = mandelbrot_fit(np.full(10, 5.0))
    assert fit["r_squared"] == 0.0
    np.testing.assert_allclose(fit["predicted"], 5.0, rtol=0.01)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),
        ([0.03], [0.03]),
        ([0.04, 0.01, 0.03, 0.002], [0.04, 0.02, 0.04, 0.008]),
        ([0.02, 0.02, 0.8], [0.03, 0.03, 0.8]),
        ([0.0, 1.0, 0.9], [0.0, 1.0, 1.0]),
    ],
)
def test_benjamini_hochberg_hand_calculated(values, expected):
    original = values.copy()
    assert benjamini_hochberg(values) == pytest.approx(expected)
    assert values == original


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_benjamini_hochberg_rejects_invalid_p_values(value):
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, value])


def test_association_fisher_tiny_perfect_pairs_and_q_boundary():
    seqs = [["a", "b"]] * 2 + [["c", "d"]] * 2
    result = analyze_pairs(seqs, alpha=1 / 6)
    assert result["n_observed"] == result["n_tested"] == result["n_retained"] == 2
    assert {row["pair"] for row in result["pairs"]} == {("a", "b"), ("c", "d")}
    for row in result["pairs"]:
        assert row["count"] == row["first_count"] == row["second_count"] == 2
        assert row["total"] == 4
        assert row["p_value"] == pytest.approx(1 / 6)
        assert row["p_adj"] == pytest.approx(1 / 6)
        assert row["retained"]
        assert row["llr"] < 10.83
        assert row["pmi"] == pytest.approx(1.0)
        assert row["npmi"] == pytest.approx(1.0)
        assert row["logdice"] == pytest.approx(14.0)
    assert analyze_pairs(seqs, alpha=0.16)["n_retained"] == 0


def test_association_bh_uses_all_observed_pairs_before_count_filter():
    seqs = [["a", "b"]] * 3 + [["c", "d"]] * 2 + [["e", "f"]]
    result = analyze_pairs(seqs, alpha=0.09, min_count=3)
    rows = {row["pair"]: row for row in result["pairs"]}
    assert result["n_observed"] == result["n_tested"] == 3
    assert result["n_retained"] == 0
    assert rows[("a", "b")]["p_value"] == pytest.approx(1 / 20)
    assert rows[("c", "d")]["p_value"] == pytest.approx(1 / 15)
    assert rows[("e", "f")]["p_value"] == pytest.approx(1 / 6)
    assert rows[("a", "b")]["p_adj"] == pytest.approx(0.1)
    assert rows[("c", "d")]["p_adj"] == pytest.approx(0.1)
    assert rows[("e", "f")]["p_adj"] == pytest.approx(1 / 6)
    retained = analyze_pairs(seqs, alpha=0.1, min_count=3)
    assert retained["n_retained"] == 1
    assert [row["pair"] for row in retained["pairs"] if row["retained"]] == [("a", "b")]


def test_association_fisher_is_one_sided_attraction_not_repulsion():
    seqs = [["a", "b"]] + [["a", "d"]] * 9 + [["c", "b"]] * 9 + [["c", "d"]]
    result = analyze_pairs(seqs)
    rows = {row["pair"]: row for row in result["pairs"]}
    assert rows[("a", "b")]["llr"] > 10.83
    assert rows[("a", "b")]["p_value"] == pytest.approx(1 - 1 / math.comb(20, 10))
    assert not rows[("a", "b")]["retained"]
    assert rows[("a", "d")]["p_value"] == pytest.approx(101 / math.comb(20, 10))
    assert rows[("a", "d")]["retained"]
    assert result["n_retained"] == 2


def test_association_can_skip_exact_tests(monkeypatch):
    def unexpected_fisher(*args, **kwargs):
        pytest.fail("Fisher should not be called when exact=False")

    monkeypatch.setattr("arthanvesana.replicate.assoc.fisher_exact", unexpected_fisher)
    result = analyze_pairs([["a", "b"], ["c", "d"]], exact=False)
    assert result["n_observed"] == 2
    assert result["n_tested"] == result["n_retained"] == 0
    for row in result["pairs"]:
        assert row["p_value"] is None
        assert row["p_adj"] is None
        assert not row["retained"]
        assert row["pmi"] == 1.0


@pytest.mark.parametrize("seqs", [[], [[]], [["a"]]])
def test_association_no_observed_pairs(seqs):
    result = analyze_pairs(seqs)
    assert result["pairs"] == []
    assert result["n_observed"] == result["n_tested"] == result["n_retained"] == 0


def test_association_degenerate_single_pair():
    result = analyze_pairs([["a", "b"]] * 4)
    row, = result["pairs"]
    assert row["p_value"] == row["p_adj"] == 1.0
    assert not row["retained"]
    assert row["npmi"] == 0.0


def test_association_tables_preserve_windows_and_effect_sizes():
    seqs = [["a", "b", "a", "b"], ["c", "b"], [], ["z"]]
    tables = bigram_tables(seqs)
    assert tables == pair_contingency(seqs)
    assert tables == {
        ("a", "b"): (2, 0, 1, 1, 4),
        ("b", "a"): (1, 0, 0, 3, 4),
        ("c", "b"): (1, 0, 2, 1, 4),
    }
    result = analyze_pairs(seqs)
    for row in result["pairs"]:
        table = tables[row["pair"]]
        assert row["pmi"] == pmi(table)
        assert row["npmi"] == npmi(table)
        assert row["logdice"] == logdice(table)
    assert rank_pairs(seqs, min_count=1, llr_min=0) == ranked_pairs(
        seqs, min_count=1, llr_min=0
    )
    assert [row["npmi"] for row in result["pairs"]] == sorted(
        [row["npmi"] for row in result["pairs"]], reverse=True
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_association_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        analyze_pairs([], alpha=alpha)


@pytest.mark.parametrize("min_count", [0, -1, 1.5, True])
def test_association_rejects_invalid_min_count(min_count):
    with pytest.raises(ValueError):
        analyze_pairs([], min_count=min_count)
