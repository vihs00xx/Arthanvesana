import math

import pytest

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
    assert math.isclose(mm, 1.0 + 1.0 / (4.0 * math.log(2.0)))


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


def test_artifact_identity_links_across_sites():
    from arthanvesana.stats.sampling import split_records

    records = [
        {"inscription_id": "a", "artifact_id": "shared", "site": "north", "sequence": ["a"]},
        {"inscription_id": "b", "artifact_id": "shared", "site": "south", "sequence": ["b"]},
    ]
    train, test, report = split_records(records, 0.5, track="artifact")
    assert report["n_groups"] == 1
    assert not train or not test


def _split_fixture():
    return [
        {"inscription_id": "a", "sequence": ["x"], "span_index": 0, "artifact_id": "obj1", "site": "s1"},
        {"inscription_id": "a", "sequence": ["y"], "span_index": 1, "artifact_id": "obj1", "site": "s1"},
        {"inscription_id": "b", "sequence": ["z"], "artifact_id": "obj1", "site": "s1"},
        {"inscription_id": "c", "sequence": ["z"], "artifact_id": "obj2", "site": "s2"},
        {"inscription_id": "d", "sequence": ["w"], "artifact_id": "obj3", "site": "s3"},
        {"inscription_id": "e", "sequence": ["v"], "artifact_id": "obj4", "site": "s4"},
    ]


@pytest.mark.parametrize("track", ["record", "sequence", "artifact", "site"])
def test_record_splits_keep_spans_and_track_groups_together(track):
    from arthanvesana.stats.sampling import split_records

    records = _split_fixture()
    train, test, report = split_records(records, 0.5, seed=9, track=track)
    assert {r["inscription_id"] for r in train}.isdisjoint(r["inscription_id"] for r in test)
    assert report["overlap"][track] == 0
    assert report["overlap"]["record"] == 0
    assert (train, test, report) == split_records(list(reversed(records)), 0.5, seed=9, track=track)
    assert len(train) + len(test) == len(records)
    assert all("split_group" not in r for r in records)


def test_artifact_duplicate_grouping_is_transitive():
    from arthanvesana.stats.sampling import split_records

    train, test, report = split_records(_split_fixture(), 0.5, track="artifact")
    linked = [r for r in train + test if r["inscription_id"] in {"a", "b", "c"}]
    assert len({r["split_group"] for r in linked}) == 1
    assert report["n_groups"] == 3
    assert report["largest_group"] == 4
    assert report["overlap"]["sequence"] == 0


def test_group_keys_legacy_split_and_invalid_arguments():
    from arthanvesana.stats.sampling import split_records

    seqs = [[str(i)] for i in range(6)]
    train, test = train_test_split(seqs, 0.5, group_keys=[0, 0, 1, 1, 2, 2])
    for pair in (seqs[:2], seqs[2:4], seqs[4:]):
        assert all(s in train for s in pair) or all(s in test for s in pair)
    for fraction in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="train_frac"):
            train_test_split(seqs, fraction)
    with pytest.raises(ValueError, match="group_keys"):
        train_test_split(seqs, group_keys=[])
    with pytest.raises(ValueError, match="track"):
        split_records([], track="bad")
    with pytest.raises(ValueError, match="inscription_id"):
        split_records([{"sequence": ["a"]}])
    train, test, report = split_records([], 0.5)
    assert train == test == []
    assert report["warnings"]


def test_missing_group_metadata_does_not_merge_unrelated_records():
    from arthanvesana.stats.sampling import split_records

    records = [{"inscription_id": str(i), "sequence": [str(i)]} for i in range(5)]
    _, _, report = split_records(records, track="artifact")
    assert report["n_groups"] == 5
    assert report["missing_artifacts"] == 5


def test_restoration_keeps_oov_failures_and_identity():
    from arthanvesana.replicate.restore import restoration_accuracy, restoration_records

    source = {"inscription_id": "test1", "artifact_id": "artifact1", "site": "site1",
              "split_group": "group1", "sequence": ["a", "new"]}
    rows, skipped = restoration_records([["a", "b"]] * 10, [source])
    assert len(rows) == 2
    assert skipped == 0
    assert all(r["inscription_id"] == "test1" and r["split_group"] == "group1" for r in rows)
    oov = rows[1]
    assert oov["oov"] and oov["rank"] is None and not oov["hit"]
    assert oov["p_true"] == 0.0
    assert 0 < oov["top_p"] < 1
    assert oov["unknown_p"] > 0
    summary = restoration_accuracy([["a", "b"]] * 10, [source])
    assert summary["n_masked"] == 2 and summary["n_oov"] == 1
    assert summary["top_10"] == 0.5
    assert summary["n_skipped_oov"] == 0


def test_restoration_singleton_and_empty_training():
    from arthanvesana.replicate.restore import restoration_records

    rows, _ = restoration_records([["a"]], [["new"]])
    assert len(rows) == 1 and rows[0]["oov"]
    with pytest.raises(ValueError, match="training"):
        restoration_records([], [["a"]])
    with pytest.raises(ValueError, match="mask_length"):
        restoration_records(TRAIN, TRAIN, mask_length=0)


def test_contiguous_mask_predictions_do_not_observe_masked_truth():
    from arthanvesana.replicate.restore import restoration_records

    train = [["a", "b", "c", "d"], ["a", "x", "y", "d"]] * 5
    left, _ = restoration_records(train, [["a", "b", "c", "d"]], mask_length=2)
    right, _ = restoration_records(train, [["a", "x", "y", "d"]], mask_length=2)
    left = [r for r in left if r["mask_start"] == 1]
    right = [r for r in right if r["mask_start"] == 1]
    assert len(left) == len(right) == 2
    assert [(r["prediction"], r["top_p"]) for r in left] == [
        (r["prediction"], r["top_p"]) for r in right
    ]


def test_restoration_incomplete_start_uses_unigram_prior():
    from arthanvesana.replicate.restore import restoration_records

    train = [["a", "b", "b", "b"]] * 10
    complete, _ = restoration_records(train, [["a"]])
    incomplete, _ = restoration_records(train, [{
        "inscription_id": "gap", "sequence": ["a"], "start_complete": "False",
    }])
    assert complete[0]["prediction"] == "a"
    assert incomplete[0]["prediction"] == "b"


def test_region_known_direction_and_inscription_threshold():
    from arthanvesana.data.parse import to_tidy
    from arthanvesana.replicate.region import site_records, site_sequences

    frame = to_tidy([
        {"id": "one", "site": "s", "direction": "L/R", "complete": True, "symbols": ["a", "000", "b"]},
        {"id": "two", "site": "s", "direction": "OTHER", "symbols": ["c"]},
    ])
    assert site_records(frame, min_inscriptions=2) == {}
    assert site_sequences(frame, min_inscriptions=1) == {"s": [["a"], ["b"]]}
    assert len(site_records(frame, 2, known_direction_only=False)["s"]) == 3


def test_region_common_support_equal_size_and_no_inscription_leakage():
    from arthanvesana.replicate.region import cross_perplexity

    sites = {
        site: [{"inscription_id": f"{site}-{i}", "sequence": [sign], "span_index": span}
               for i in range(size) for span in range(2)]
        for site, sign, size in (("s1", "a", 10), ("s2", "b", 20))
    }
    report = cross_perplexity(sites, equal_train_size=True, return_report=True)
    assert report["vocabulary"] == ["<UNK>", "a", "b"]
    assert all(v == report["vocabulary"] for v in report["model_vocabularies"].values())
    sizes = {r["n_train_inscriptions"] for r in report["sites"].values()}
    assert len(sizes) == 1
    for site in report["sites"].values():
        assert set(site["train_ids"]).isdisjoint(site["test_ids"])
        assert site["n_train_spans"] == 2 * site["n_train_inscriptions"]
    assert report == cross_perplexity(sites, equal_train_size=True, return_report=True)


def test_region_vocabulary_excludes_test_only_signs():
    from arthanvesana.replicate.region import cross_perplexity

    sites = {"s": [[str(i)] for i in range(10)]}
    report = cross_perplexity(sites, train_frac=0.5, return_report=True)
    assert len(report["vocabulary"]) == 6
    assert report["sites"]["s"]["n_test_oov"] == 5
    assert math.isfinite(report["matrix"]["s"]["s"])
    with pytest.raises(ValueError, match="vocabulary"):
        cross_perplexity(sites, vocabulary={"bad"})


def test_region_record_direction_filter_and_legacy_span_identity():
    from arthanvesana.data.parse import to_tidy
    from arthanvesana.replicate.region import cross_perplexity, site_sequences

    frame = to_tidy([
        {"id": str(i), "site": "s", "direction": "L/R", "complete": True,
         "symbols": ["a", "000", "b"]} for i in range(10)
    ])
    report = cross_perplexity(site_sequences(frame, 1), return_report=True)
    assert report["split"]["n_groups"] == 10
    assert report["split"]["overlap"]["record"] == 0
    sites = {"s": [
        {"inscription_id": str(i), "sequence": ["a"], "direction": "L/R"}
        for i in range(10)
    ] + [{"inscription_id": "unknown", "sequence": ["z"], "direction": "OTHER"}]}
    report = cross_perplexity(sites, return_report=True)
    assert report["split"]["n_records"] == 10
    assert "z" not in report["vocabulary"]


def test_contiguous_mask_marginals_match_enumeration():
    from itertools import product

    from arthanvesana.replicate.restore import _mask_distributions, _matrix

    model = NGramModel([["a", "b"], ["b", "a"], ["a", "a"]], 2, method="wittenbell")
    mats = _matrix(model)
    marginals = _mask_distributions(mats, mats["<UNK>"], model.vocab, ["a", "b"], 0, 2, True)
    weights = {
        (a, b): mats["<S>"][a] * mats[a][b]
        for a, b in product(model.vocab, repeat=2)
    }
    total = sum(weights.values())
    for position in range(2):
        assert sum(marginals[position].values()) == pytest.approx(1.0)
        for sign in model.vocab:
            expected = sum(p for pair, p in weights.items() if pair[position] == sign) / total
            assert marginals[position][sign] == pytest.approx(expected)
