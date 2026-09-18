import math
import random
from collections import Counter

import morfessor
import pytest

from arthanvesana.replicate.assoc import (
    log_odds,
    logdice,
    npmi,
    pair_contingency,
    pmi,
    ranked_pairs,
)
from arthanvesana.replicate.entropy2 import (
    chao_shen,
    shrinkage_entropy,
    unigram_counts,
)
from arthanvesana.replicate.metrics import (
    bootstrap_ci,
    ece,
    js_divergence,
    mrr,
    normalized_perplexity,
)
from arthanvesana.replicate.segment_morfessor import (
    boundaries,
    boundary_f1,
    segment,
    train,
)
from arthanvesana.stats.entropy import conditional_entropy, mutual_information
from arthanvesana.stats.ngrams import NGramModel, _mkn_discounts


def test_chao_shen_matches_hand_computation():
    assert abs(chao_shen(Counter({"a": 2, "b": 1})) - 1.5383) < 1e-3


def test_chao_shen_bounds():
    assert chao_shen(Counter()) == 0.0
    assert chao_shen(Counter({"a": 50})) == 0.0
    uni = Counter({str(i): 200 for i in range(8)})
    assert abs(chao_shen(uni) - 3.0) < 1e-9


def test_shrinkage_bounds():
    assert shrinkage_entropy(Counter()) == 0.0
    assert shrinkage_entropy(Counter({"a": 50})) == 0.0
    uni = Counter({str(i): 200 for i in range(8)})
    assert abs(shrinkage_entropy(uni) - 3.0) < 1e-9


def test_shrinkage_full_shrink_tiny_sample():
    assert abs(shrinkage_entropy(Counter({"a": 2, "b": 1})) - 1.0) < 1e-9


def test_unigram_counts():
    assert unigram_counts([["a", "b"], ["a"]]) == Counter({"a": 2, "b": 1})


def test_npmi_perfect_association_near_one():
    seqs = [["a", "b"]] * 10 + [["c", "d"]] * 10
    table = pair_contingency(seqs)[("a", "b")]
    assert abs(npmi(table) - 1.0) < 0.05


def test_npmi_independence_near_zero():
    seqs = (
        [["a", "b"]] * 10
        + [["a", "c"]] * 10
        + [["d", "b"]] * 10
        + [["d", "c"]] * 10
    )
    table = pair_contingency(seqs)[("a", "b")]
    assert abs(npmi(table)) < 1e-9
    assert abs(pmi(table)) < 1e-9


def test_logdice_perfect_is_fourteen():
    seqs = [["a", "b"]] * 10 + [["c", "d"]] * 10
    table = pair_contingency(seqs)[("a", "b")]
    assert logdice(table) == 14.0


def test_log_odds_sign():
    seqs = [["a", "b"]] * 10 + [["c", "d"]] * 10
    table = pair_contingency(seqs)[("a", "b")]
    assert log_odds(table) > 0


def test_ranked_pairs_filters_and_orders():
    seqs = [["a", "b"]] * 10 + [["c", "d"]] * 10 + [["a", "x"], ["y", "b"]]
    rows = ranked_pairs(seqs, min_count=3)
    assert {tuple(r["pair"]) for r in rows} == {("a", "b"), ("c", "d")}
    npm = [r["npmi"] for r in rows]
    assert npm == sorted(npm, reverse=True)


def test_mrr():
    assert mrr([1, 2, 4]) == (1 + 0.5 + 0.25) / 3
    assert mrr([]) == 0.0
    assert mrr([None, 1]) == 0.5


def test_ece_perfect_calibration():
    assert ece([0.95] * 10, [True] * 10, bins=10) < 0.1
    assert ece([0.9] * 10, [False] * 10, bins=10) > 0.5


def test_js_divergence():
    assert js_divergence(Counter({"a": 5}), Counter({"a": 5})) == 0.0
    assert abs(js_divergence(Counter({"a": 1}), Counter({"b": 1})) - 1.0) < 1e-9


def test_bootstrap_ci_deterministic_stat():
    out = bootstrap_ci([[1], [2], [3]], lambda s: len(s), n_reps=50, seed=0)
    assert out["mean"] == 3
    assert out["lo"] == 3 and out["hi"] == 3


def test_normalized_perplexity():
    assert normalized_perplexity(48.32, 713) == 48.32 / 713


def test_boundaries_and_f1():
    seg = [("a", "b"), ("c",)]
    assert boundaries(seg) == {2}
    assert boundaries([("a", "b", "c")]) == set()
    assert boundary_f1(seg, seg) == 1.0
    assert boundary_f1(seg, [("a",), ("b", "c")]) == 0.0
    assert boundary_f1([("a",)], [("a",)]) == 1.0


def test_morfessor_segment_covers_sequence():
    seqs = [["a", "b", "c"]] * 10 + [["a", "b"]] * 10 + [["c", "d"]] * 5
    model = train(seqs)
    for seq in (["a", "b", "c"], ["c", "d"]):
        seg = segment(model, seq)
        flat = [s for morph in seg for s in morph]
        assert flat == seq


@pytest.mark.parametrize("order", [0, 1, 2])
def test_miller_madow_correction_is_in_bits(order):
    seqs = [["x"] * order + [s] for s in ("a", "b")]
    plugin, corrected = conditional_entropy(seqs, order)
    assert plugin == 1.0
    assert corrected == pytest.approx(1.0 + 1.0 / (4 * math.log(2)))


def test_mutual_information_uses_matched_outcome_marginals():
    seqs = [["a", "x"], ["b", "x"]] + [["z"]] * 20
    assert mutual_information(seqs) == 0.0
    assert mutual_information([["a", "x"], ["b", "y"]]) == 1.0
    independent = [[a, b] for a in ("a", "b") for b in ("x", "y")]
    assert mutual_information(independent) == 0.0


def test_mutual_information_joint_context_xor():
    seqs = [[str(a), str(b), str(a ^ b)] for a in (0, 1) for b in (0, 1)]
    assert mutual_information(seqs, order=2) == 1.0
    assert conditional_entropy(seqs, order=2) == (0.0, 0.0)


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_entropy_degenerate_windows(order):
    assert conditional_entropy([], order) == (0.0, 0.0)
    assert mutual_information([], order) == 0.0
    assert mutual_information([["a"]], order) == 0.0
    assert mutual_information([["a", "b"]], order=0) == 0.0


@pytest.mark.parametrize("estimator", [conditional_entropy, mutual_information])
def test_entropy_rejects_negative_order(estimator):
    with pytest.raises(ValueError, match="order"):
        estimator([["a"]], -1)


@pytest.mark.parametrize("counts", [Counter(a=1), Counter(a=1, b=1, zero=0)])
def test_chao_shen_rejects_all_singletons(counts):
    with pytest.raises(ValueError, match="all-singleton"):
        chao_shen(counts)


def test_chao_shen_tiny_probability_is_not_discarded():
    n = 10**20 + 2
    p = 2 / n
    expected = -p * math.log2(p) / (-math.expm1(-2.0))
    assert chao_shen(Counter(a=10**20, b=2)) == pytest.approx(expected, rel=1e-12, abs=0)
    assert chao_shen(Counter(a=0)) == 0.0
    assert chao_shen(Counter(a=2, b=1, zero=0)) == chao_shen(Counter(a=2, b=1))


@pytest.mark.parametrize("count", [-1, 0.5, float("nan"), float("inf")])
def test_chao_shen_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="counts"):
        chao_shen(Counter(a=count))


def test_ece_uses_mean_confidence_not_bin_midpoint():
    assert ece([0.0, 1.0], [False, True]) == 0.0
    assert ece([0.91, 0.99], [True, False]) == pytest.approx(0.45)
    assert ece([0.1, 0.2, 0.9], [False, True, True], bins=2) == pytest.approx(0.8 / 3)
    assert ece([0.2, 0.8], [False, True], bins=1) == 0.0
    assert ece([], []) == 0.0


@pytest.mark.parametrize(
    "confidences,correct,bins",
    [([0.5], [], 10), ([], [], 0), ([], [], -1), ([], [], 1.5),
     ([-0.1], [False], 10), ([1.1], [True], 10),
     ([float("nan")], [True], 10), ([float("inf")], [True], 10)],
)
def test_ece_rejects_invalid_inputs(confidences, correct, bins):
    with pytest.raises(ValueError):
        ece(confidences, correct, bins)


def test_mkn_zero_discount_falls_back_to_absolute_discount():
    assert _mkn_discounts(Counter(a=2, b=2)) == (0.75, 0.75, 0.75)
    model = NGramModel([["a"], ["a"], ["b"], ["b"]], 1, method="mkn")
    assert model.dist(()) == pytest.approx({"a": 7 / 16, "b": 7 / 16, "<UNK>": 1 / 8})


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seqs", [[], [[]], [["a"]] * 2, [["a", "b"]] * 2])
def test_mkn_sparse_distributions_are_positive_and_normalized(order, seqs):
    model = NGramModel(seqs, order, method="mkn")
    for context in [("<S>",) * (order - 1), ("a",) * (order - 1), ("missing",) * (order - 1)]:
        dist = model.dist(context)
        assert sum(dist.values()) == pytest.approx(1.0)
        assert all(0.0 < p <= 1.0 for p in dist.values())
    assert math.isfinite(model.perplexity([["a", "unseen", "b"]]))


@pytest.mark.parametrize("method", ["laplace", "wittenbell", "interp", "mkn"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_distribution_iteration_is_sorted(method, order):
    seqs = [["z", "a", "x"], ["x", "a", "z"]]
    model = NGramModel(seqs, order, method=method)
    context = ("<S>",) * (order - 1)
    dist = model.dist(context)
    assert list(dist) == sorted(model.vocab)
    assert list(dist.items()) == list(NGramModel(list(reversed(seqs)), order, method=method).dist(context).items())


def test_mkn_existing_hand_computed_distribution_is_preserved():
    model = NGramModel([["a", "b"], ["a", "b"], ["a", "c"]], 2, method="mkn")
    dist = model.dist(("a",))
    assert dist["b"] == pytest.approx(11 / 18)
    assert dist["c"] == pytest.approx(5 / 18)


def test_morfessor_loads_counted_sign_tuples(monkeypatch):
    captured = []
    original = morfessor.BaselineModel.load_data

    def capture(model, data):
        rows = list(data)
        captured.extend(rows)
        return original(model, rows)

    monkeypatch.setattr(morfessor.BaselineModel, "load_data", capture)
    seqs = [["001", "023a"], ["001", "023a"], ["104"], []]
    model = train(seqs)
    assert captured == [(2, ("001", "023a")), (1, ("104",))]
    assert {compound: count for count, compound, _ in model.get_segmentations()} == {
        ("001", "023a"): 2, ("104",): 1,
    }
    for seq in seqs:
        assert [sign for morph in segment(model, seq) for sign in morph] == seq


def test_morfessor_training_is_repeatable_and_restores_random_state():
    seqs = [["001", "023a", "104"]] * 10 + [["001", "023a"]] * 8 + [["104", "201"]] * 5
    state = random.getstate()
    first = train(seqs, seed=17)
    assert random.getstate() == state
    second = train(list(reversed(seqs)), seed=17)
    assert random.getstate() == state
    assert list(first.get_segmentations()) == list(second.get_segmentations())
    assert first.get_cost() == second.get_cost()
    assert train(seqs).get_cost() == train(seqs, seed=0).get_cost()


def test_morfessor_seed_is_applied_and_state_restored_on_failure(monkeypatch):
    draws = []

    def fail(model):
        draws.append(random.random())
        raise RuntimeError("training failed")

    monkeypatch.setattr(morfessor.BaselineModel, "train_batch", fail)
    state = random.getstate()
    with pytest.raises(RuntimeError, match="training failed"):
        train([["001"]], seed=23)
    assert draws == [random.Random(23).random()]
    assert random.getstate() == state


@pytest.mark.parametrize("seqs", [[], [[]], [[], []]])
def test_morfessor_rejects_empty_training(seqs):
    state = random.getstate()
    with pytest.raises(ValueError, match="nonempty"):
        train(seqs)
    assert random.getstate() == state
