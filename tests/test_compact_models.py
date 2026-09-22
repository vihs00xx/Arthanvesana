"""Regression tests for the compact structural models.

These check scientific failure modes:

* artifact isolation survives the complete runner;
* HMM state-count selection uses a GROUPED inner split;
* one unknown sign does not discard the known-token contributions of a sequence;
* predictive distributions are normalized and never empty;
* position smoothing behaves sensibly for empty and rare buckets;
* training caps use whole components and are deterministic and visible;
* restoration scoring never credits an unseen sign via ``<UNK>``;
* the HMM is deterministic, records convergence, and counts parameters honestly.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from arthanvesana.data.parse import to_tidy
from arthanvesana.replicate.compact import (
    UNK,
    DiscreteHMM,
    PositionModel,
    Vocabulary,
    cap_records_by_group,
    crossfit_compact,
    inner_grouped_split,
    oov_rate,
    relative_bin,
    restoration_top1,
    token_logloss,
)
from arthanvesana.stats.sampling import connected_groups

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(n=30, with_unk=False):
    recs = []
    for i in range(n):
        seq = ["001", f"{200 + i:03d}", "003"]
        if with_unk and i == 0:
            seq = [*seq, "zzz"]
        recs.append({
            "inscription_id": f"i{i}", "artifact_id": f"A{i}", "site": "S",
            "artifact_group": (None, "explicit", f"A{i}"),
            "sequence": seq, "span_index": 0, "span_start": 0,
            "start_complete": True, "end_complete": True,
        })
    return recs


# ---------------------------------------------------------------------------
# Unknown-sign handling
# ---------------------------------------------------------------------------

def test_vocabulary_is_training_only_and_always_has_unk():
    vocab = Vocabulary([["a", "b"], ["b", "c"]])
    assert set(vocab.signs) == {"a", "b", "c"}
    assert vocab.size == 4
    assert vocab.global_prob(UNK) > 0.0
    assert vocab.global_prob("never_seen") == 0.0
    assert vocab.map(["a", "zzz"]) == ["a", UNK]


def test_one_unknown_sign_does_not_discard_known_tokens():
    records = _records(with_unk=True)
    vocab = Vocabulary([r["sequence"] for r in records[1:]])

    hmm = DiscreteHMM(n_states=2, n_iter=3, seed=0).fit(records[1:])
    pos = PositionModel("exact", 1.0).fit(records[1:], vocab)

    # same length, same buckets, first three signs identical and known in
    # training; only the final sign differs. "001" is frequent in training, so a
    # known final sign must score better than the unseen one.
    bad = dict(records[0], sequence=["001", "201", "003", "zzz"])
    known = dict(records[0], sequence=["001", "201", "003", "001"])

    for model in (hmm, pos):
        # the whole-sequence likelihood is finite, not replaced by a penalty
        assert model.logprob(bad) is not None
        assert np.isfinite(model.logprob(bad))
        # the known tokens still contribute: one unseen sign costs a finite
        # amount rather than erasing the sequence's other tokens
        assert model.logprob(bad) < model.logprob(known)
        assert model.logprob(bad) - model.logprob(known) > -30

    # token_logloss never substitutes a per-sequence penalty
    loss, ntok = token_logloss(hmm, [bad])
    assert ntok == 4
    assert loss is not None and np.isfinite(loss)


def test_oov_rate_is_reported_per_model():
    records = _records(with_unk=True)
    vocab = Vocabulary([r["sequence"] for r in records[1:]])
    rate = oov_rate(vocab, records)
    assert 0 < rate < 1
    res = crossfit_compact(records, n_folds=3, seed=0, state_counts=(2,),
                           inits=("uniform",), alphas=(1.0,), hmm_iter=2,
                           budget="matched", cap_records=None)
    for name in ("bigram", "position_exact", "position_relative",
                 "position_exact_complete", "hmm"):
        assert res[name]["oov_rate"] is not None


def test_restoration_never_credits_an_unseen_sign_via_unk():
    records = _records(with_unk=True)
    vocab = Vocabulary([r["sequence"] for r in records[1:]])
    pos = PositionModel("exact", 1.0).fit(records[1:], vocab)
    acc, total = restoration_top1(pos, vocab, records)
    assert total == sum(len(r["sequence"]) for r in records)
    # the unseen sign can only ever be counted as a miss
    assert 0.0 <= acc < 1.0


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def test_position_distributions_are_normalized_and_smooth_rare_buckets():
    records = _records()
    vocab = Vocabulary([r["sequence"] for r in records])
    model = PositionModel("exact", 1.0).fit(records, vocab)

    seen = model.token_dist(3, 0)
    assert sum(seen.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(p > 0 for p in seen.values())

    # an empty bucket falls back to the global distribution, not to zero
    empty = model.token_dist(999, 7)
    assert sum(empty.values()) == pytest.approx(1.0, abs=1e-9)
    for sign in vocab.outcomes():
        assert empty[sign] == pytest.approx(vocab.global_prob(sign), abs=1e-12)

    # a larger alpha pulls a rare bucket closer to the global distribution
    weak = PositionModel("exact", 0.01).fit(records, vocab).token_dist(3, 0)
    strong = PositionModel("exact", 1000.0).fit(records, vocab).token_dist(3, 0)
    glob = {s: vocab.global_prob(s) for s in vocab.outcomes()}

    def tv(dist):
        return 0.5 * sum(abs(dist[s] - glob[s]) for s in glob)

    assert tv(strong) < tv(weak)


def test_relative_bins_are_defined_and_singletons_are_separate():
    assert relative_bin(0, 1) == "singleton"
    assert relative_bin(0, 5) == "0-20%"
    assert relative_bin(4, 5) == "80-100%"
    labels = {relative_bin(p, 10) for p in range(10)}
    assert len(labels) >= 4


def test_hmm_distributions_are_normalized_and_deterministic():
    records = _records()
    a = DiscreteHMM(n_states=2, n_iter=3, seed=0).fit(records)
    b = DiscreteHMM(n_states=2, n_iter=3, seed=0).fit(records)
    assert a.logprob(records[0]) == b.logprob(records[0])
    assert a.logprob(records[0]) < 0
    assert np.allclose(np.exp(a.log_pi).sum(), 1.0, atol=1e-6)
    assert np.allclose(np.exp(a.log_a).sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.exp(a.log_b).sum(axis=1), 1.0, atol=1e-6)


def test_hmm_records_convergence_and_parameter_counts():
    records = _records()
    model = DiscreteHMM(n_states=3, n_iter=6, seed=1,
                        init="frequency_slice").fit(records)
    assert model.iterations_run >= 1
    assert len(model.loglik_history) == model.iterations_run
    assert model.n_parameters > 0
    v = len(model.vocab)
    n = model.n_states_used
    assert model.n_parameters == (n - 1) + n * (n - 1) + n * (v - 1)
    assert UNK in model.vocab


def test_hmm_state_count_is_capped_by_vocabulary():
    model = DiscreteHMM(n_states=10, n_iter=2, seed=0).fit(_records(2))
    assert model.log_pi.size <= len(model.vocab)


# ---------------------------------------------------------------------------
# Grouping, caps, inner selection
# ---------------------------------------------------------------------------

def test_inner_split_never_shares_a_component():
    records = _records(40)
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    fit, valid = inner_grouped_split(records, groups, seed=3)
    assert fit and valid
    gid = {i: g["group_id"] for g in groups for i in g["indices"]}
    index_of = {id(r): i for i, r in enumerate(records)}
    fit_groups = {gid[index_of[id(r)]] for r in fit}
    valid_groups = {gid[index_of[id(r)]] for r in valid}
    assert not (fit_groups & valid_groups), (
        "inner selection must be grouped: no component may straddle the inner split"
    )


def test_training_cap_uses_whole_components_and_is_deterministic():
    records = _records(40)
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    a, info_a = cap_records_by_group(records, groups, 10, seed=0)
    b, info_b = cap_records_by_group(records, groups, 10, seed=0)
    assert [r["inscription_id"] for r in a] == [r["inscription_id"] for r in b]
    assert info_a == info_b
    assert info_a["capped"] is True
    assert len(a) >= 10
    # whole components only: every component is either fully in or fully out
    chosen = {r["inscription_id"] for r in a}
    for g in groups:
        members = {records[i]["inscription_id"] for i in g["indices"]}
        assert members <= chosen or not (members & chosen)


def test_crossfit_preserves_artifact_identity_and_reports_caps():
    records = _records(40)
    res = crossfit_compact(records, n_folds=3, seed=0, state_counts=(2,),
                           inits=("uniform",), alphas=(1.0,), hmm_iter=2,
                           budget="matched", cap_records=12)
    assert res["n_records"] == len(records)
    assert res["leakage"]["groups_split_across_folds"] == 0
    assert res["leakage"]["sequence_crossings"] == 0
    assert res["budget"] == "matched"
    assert res["training_budgets"], "realized training sizes must be visible"
    assert all(b["cap"] == 12 for b in res["training_budgets"])
    assert any(b["capped"] for b in res["training_budgets"])


def test_crossfit_reports_all_models_and_is_deterministic():
    records = _records(40)
    kw = dict(n_folds=3, seed=5, state_counts=(2,), inits=("uniform",),
              alphas=(1.0,), hmm_iter=2, budget="matched", cap_records=None)
    a = crossfit_compact(records, **kw)
    b = crossfit_compact(records, **kw)
    assert a == b
    for name in ("bigram", "position_exact", "position_relative",
                 "position_exact_complete", "hmm"):
        assert a[name]["bits_per_token"] is not None
        assert a[name]["bits_per_token"] > 0
        assert a[name]["perplexity"] == pytest.approx(
            2 ** a[name]["bits_per_token"])
    assert len(a["selected_hmm_states"]) == 3
    assert set(a["selected_hmm_states"]) <= {2}


def test_full_budget_uses_more_training_data_than_matched():
    records = _records(60)
    matched = crossfit_compact(records, n_folds=3, seed=0, state_counts=(2,),
                               inits=("uniform",), alphas=(1.0,), hmm_iter=2,
                               budget="matched", cap_records=10)
    full = crossfit_compact(records, n_folds=3, seed=0, state_counts=(2,),
                            inits=("uniform",), alphas=(1.0,), hmm_iter=2,
                            budget="full", cap_records=None)
    m = max(b["n_records"] for b in matched["training_budgets"])
    f = max(b["n_records"] for b in full["training_budgets"])
    assert f > m, "the full budget must actually use more training data"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _corpus(tmp_path, n=30):
    raw = [
        {"id": str(i), "cisi": str(i), "site": "A", "direction": "L/R",
         "complete": True, "symbols": ["001", f"{100 + (i % 3):03d}", "003"]}
        for i in range(n)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    return corpus


def test_compact_runner_reports_both_budgets(tmp_path):
    runner = load_runner("run_compact_models")
    corpus = _corpus(tmp_path)
    output = tmp_path / "out"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output), "--repeats", "1",
        "--cap-records", "12",
    ])
    saved = json.loads(
        (output / "compact_models_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert set(summary["budgets"]) == {"matched", "full"}
    for budget in ("matched", "full"):
        agg = summary["budgets"][budget]["aggregate"]
        assert agg["bigram"]["n_runs"] == 1
        assert agg["hmm"]["bits_per_token_mean"] is not None
        assert agg["selected_hmm_states"]
    report = (output / "compact_models_report.txt").read_text(encoding="utf-8")
    assert "TWO BUDGET EXPERIMENTS" in report
    assert "budget: matched" in report
    assert "budget: full" in report
    assert "not a fair" in report
    assert "training-only vocabulary" in report
    assert "confidence intervals" in report
    assert summary["manifest"]["grouping_policy"].startswith("artifact_grouped")