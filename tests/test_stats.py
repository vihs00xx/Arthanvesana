import math

from arthanvesana.stats.entropy import conditional_entropy, unigram_entropy
from arthanvesana.stats.ngrams import NGramModel, count_ngrams, top_ngrams
from arthanvesana.stats.position import positional_counts, positional_profile
from arthanvesana.stats.sampling import (
    deduplicated,
    shuffled_corpus,
    train_test_split,
)

TRAIN = [["a", "b"], ["a", "c"], ["b", "a"]]


def test_count_ngrams():
    assert count_ngrams(TRAIN, 1) == {("a",): 3, ("b",): 2, ("c",): 1}
    assert count_ngrams(TRAIN, 2)[("a", "b")] == 1
    assert count_ngrams([["a"]], 2) == {}


def test_unigram_perplexity_matches_hand_computation():
    model = NGramModel(TRAIN, 1, k=1.0)
    assert model.vocab == {"a", "b", "c", "<UNK>"}
    denom = 6 + 4
    expect = 2.0 ** (
        -(
            math.log2((3 + 1) / denom)
            + math.log2((2 + 1) / denom)
            + math.log2((0 + 1) / denom)
        )
        / 3
    )
    assert model.perplexity([["a", "b", "zzz"]]) == expect


def test_bigram_start_and_oov_handling():
    model = NGramModel(TRAIN, 2, k=1.0)
    assert model.logprob([]) == 0.0
    lp = model.logprob(["a", "b"])
    expect = math.log2(3 / 7) + math.log2(2 / 6)
    assert lp == expect
    assert math.isnan(model.perplexity([]))


def test_top_ngrams_order():
    assert top_ngrams(TRAIN, 1, 1) == [(("a",), 3)]


def test_mkn_matches_hand_computation():
    train = [["a", "b"], ["a", "b"], ["a", "c"]]
    model = NGramModel(train, 2, method="mkn")
    d = model.dist(("a",))
    assert abs(d["b"] - 11 / 18) < 1e-9
    assert abs(d["c"] - 5 / 18) < 1e-9


def test_distributions_sum_to_one():
    cases = [
        (1, [()]),
        (2, [("a",), ("b",), ("c",), ("<S>",), ("<UNK>",), ("q",)]),
        (3, [("<S>", "<S>"), ("<S>", "a"), ("a", "b"), ("q", "z")]),
    ]
    for n, ctxs in cases:
        for method in ("laplace", "wittenbell", "interp", "mkn"):
            model = NGramModel(TRAIN, n, method=method)
            for ctx in ctxs:
                total = sum(model.dist(ctx).values())
                assert abs(total - 1.0) < 1e-9, (n, method, ctx, total)


def test_unigram_entropy():
    assert unigram_entropy([["a", "a", "b", "b"]]) == 1.0
    assert unigram_entropy([]) == 0.0


def test_conditional_entropy_deterministic():
    seqs = [["a", "b", "a", "b", "a", "b"]]
    plugin, mm = conditional_entropy(seqs, 1)
    assert plugin == 0.0
    assert mm == 0.0


def test_conditional_entropy_miller_madow_exceeds_plugin():
    plugin, mm = conditional_entropy([["a", "b"], ["a", "c"]], 1)
    assert plugin == 1.0
    assert mm == 1.25


def test_positional_counts_singletons_count_both_ends():
    pos = positional_counts([["x"], ["x", "y"]])
    assert pos["begin"]["x"] == 2
    assert pos["end"]["x"] == 1
    assert pos["middle"]["x"] == 0
    assert pos["end"]["y"] == 1


def test_positional_profile_threshold_and_fractions():
    rows = positional_profile([["a", "b"], ["a", "b"], ["b", "a"]], min_count=2)
    by_sign = {r["sign"]: r for r in rows}
    assert set(by_sign) == {"a", "b"}
    assert by_sign["a"]["p_begin"] == 2 / 3
    assert by_sign["a"]["p_begin"] + by_sign["a"]["p_middle"] + by_sign["a"]["p_end"] == 1.0
    solo = positional_profile([["a"]], min_count=2)
    assert solo == [
        {"sign": "a", "count": 2, "p_begin": 0.5, "p_middle": 0.0, "p_end": 0.5}
    ]


def test_shuffled_corpus_preserves_multisets():
    seqs = [["a", "b", "c"], ["a", "a"]]
    reps = shuffled_corpus(seqs, seed=3, n_replicates=4)
    assert len(reps) == 4
    for rep in reps:
        assert [sorted(s) for s in rep] == [["a", "b", "c"], ["a", "a"]]
    assert shuffled_corpus(seqs, seed=3)[0] == shuffled_corpus(seqs, seed=3)[0]


def test_split_disjoint_reproducible():
    seqs = [[str(i)] for i in range(100)]
    train, test = train_test_split(seqs, train_frac=0.8, seed=1)
    assert len(train) == 80
    assert len(test) == 20
    assert {tuple(s) for s in train}.isdisjoint({tuple(s) for s in test})
    again, _ = train_test_split(seqs, train_frac=0.8, seed=1)
    assert train == again


def test_deduplicated():
    seqs = [["a"], ["b"], ["a"], ["a", "b"]]
    assert deduplicated(seqs) == [["a"], ["b"], ["a", "b"]]
