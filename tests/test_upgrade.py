import math
from collections import Counter

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
