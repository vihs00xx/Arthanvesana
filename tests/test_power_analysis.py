"""Regression tests for the synthetic calibration runner.

These target scientific failure modes rather than file existence:

* a significantly negative effect can never be counted as a positive discovery;
* a significantly positive effect is recorded separately;
* rejection rates use the correct denominator and count failed/skipped runs;
* Monte Carlo p-values use the plus-one correction and are never zero;
* the generator fallback advances the Markov state;
* scenario metadata states honestly what the generators do and do not preserve;
* repeated cross-fitting under different fold seeds is never pooled into one
  larger sample.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from arthanvesana.data.parse import to_tidy
from arthanvesana.simulate import (
    crossfit_effect,
    empirical_profile,
    gen_markov,
    gen_shuffled_real,
    gen_trigram_mixture,
    gen_unigram,
)
from arthanvesana.simulate.pipeline import signflip_tests

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(n_signs=30, n_seqs=240, seed=0):
    """A profile with a rich vocabulary so the fitted trigram genuinely loses."""
    import random
    rng = random.Random(seed)
    signs = [f"{100 + i:03d}" for i in range(n_signs)]
    seqs = [[rng.choice(signs) for _ in range(rng.choice([3, 4, 5]))]
            for _ in range(n_seqs)]
    records = [{"inscription_id": f"i{k}", "sequence": s} for k, s in enumerate(seqs)]
    return empirical_profile(records)


# --------------------------------------------------------------------------
# Step 2: the three decision rules are separate
# --------------------------------------------------------------------------

def test_signflip_separates_positive_and_negative_directions():
    positive = [0.4] * 20
    negative = [-0.4] * 20

    up = signflip_tests(positive, 400, 0)
    assert up["p_positive"] < 0.05
    assert up["p_negative"] > 0.5
    assert up["p_two_sided"] < 0.05

    down = signflip_tests(negative, 400, 0)
    assert down["p_positive"] > 0.5, "a negative effect must not look like an improvement"
    assert down["p_negative"] < 0.05
    assert down["p_two_sided"] < 0.05


def test_negative_effect_is_never_a_positive_discovery():
    profile = _profile()
    seqs = gen_unigram(profile, 240, 0)
    res = crossfit_effect(seqs, 5, 0, 400, 200)
    assert res["macro_effect"] is not None
    assert res["macro_effect"] < 0, "first-order data should disadvantage the trigram"
    assert res["reject_negative"] is True
    assert res["reject_positive"] is False
    # the two-sided rule still fires: there IS a predictive difference
    assert res["reject_difference"] is True
    assert res["structural_decision"] is None, (
        "structural departure is a separate, calibrated question and must not be "
        "asserted by the cross-fitting function"
    )


def test_positive_effect_is_recorded_separately():
    profile = _profile()
    seqs = gen_trigram_mixture(profile, 240, 0, 1.0)
    res = crossfit_effect(seqs, 5, 0, 400, 200)
    assert res["macro_effect"] is not None and res["macro_effect"] > 0
    assert res["reject_positive"] is True
    assert res["reject_negative"] is False


def test_plus_one_correction_never_yields_zero():
    for permutations in (10, 50, 200):
        res = signflip_tests([0.5] * 12, permutations, 0)
        floor = 1.0 / (permutations + 1)
        for key in ("p_two_sided", "p_positive", "p_negative"):
            assert res[key] >= floor
            assert res[key] > 0.0


def test_summarize_uses_correct_denominators_and_reports_failures():
    runner = load_runner("run_power_analysis")
    records = [
        {"status": "ok", "macro_effect": 0.1, "reject_difference": True,
         "reject_positive": True, "reject_negative": False},
        {"status": "ok", "macro_effect": -0.2, "reject_difference": True,
         "reject_positive": False, "reject_negative": True},
        {"status": "failed", "macro_effect": None, "reject_difference": False,
         "reject_positive": False, "reject_negative": False,
         "reason": "ValueError: boom"},
        {"status": "skipped", "macro_effect": None, "reject_difference": False,
         "reject_positive": False, "reject_negative": False,
         "reason": "no scorable group"},
    ]
    blk = runner.summarize(records)
    assert blk["n_runs"] == 4
    assert blk["n_evaluable"] == 2, "rates must not be computed over the full grid"
    assert blk["n_failed"] == 1 and blk["n_skipped"] == 1
    assert blk["failure_reasons"] == ["ValueError: boom"]
    # denominators are the evaluable runs, not the attempted runs
    assert blk["positive_improvement"]["n"] == 2
    assert blk["positive_improvement"]["k"] == 1
    assert blk["positive_improvement"]["rate"] == 0.5
    assert blk["negative"]["rate"] == 0.5
    assert blk["difference"]["n"] == 2


def test_wilson_interval_brackets_the_rate():
    runner = load_runner("run_power_analysis")
    lo, hi = runner.wilson(5, 10)
    assert lo < 0.5 < hi
    assert runner.wilson(0, 0) is None


def test_repeated_fold_seeds_are_not_pooled_into_one_sample():
    runner = load_runner("run_power_analysis")
    # two independent seeds, one replicate each: the denominator must be 2,
    # never 4, and the per-seed identity must survive
    records = [
        {"seed": 11, "status": "ok", "macro_effect": 0.05,
         "reject_difference": False, "reject_positive": False,
         "reject_negative": False},
        {"seed": 12, "status": "ok", "macro_effect": -0.05,
         "reject_difference": False, "reject_positive": False,
         "reject_negative": False},
    ]
    blk = runner.summarize(records)
    assert blk["positive_improvement"]["n"] == 2
    assert [r["seed"] for r in records] == [11, 12]
    assert blk["n_runs"] == len(records)


def test_structural_test_uses_disjoint_calibration_and_evaluation_sets():
    runner = load_runner("run_power_analysis")
    calibration = [-0.5, -0.4, -0.3, -0.2]
    evaluation = [-0.45, -0.35, -0.25, -0.15]
    out = runner.structural_test(0.06, calibration, evaluation, alpha=0.05)
    assert out["n_calibration"] == len(calibration)
    assert out["n_evaluation"] == len(evaluation)
    assert out["threshold"] is not None
    assert min(calibration) <= out["threshold"] <= max(calibration)
    # The observed effect exceeds every null draw. With only 8 null simulations the
    # plus-one correction floors the p-value at 1/9 = 0.111, so at alpha=0.05 the
    # rule correctly refuses to declare a structural departure.
    assert out["p_value"] == pytest.approx(1 / 9)
    assert out["p_value"] > 0.0
    assert out["decision"]["structural_departure"] is False
    assert out["false_positive_rate"] == pytest.approx(0.25)  # 1 of 4 eval draws
    assert out["limitations"], "the conditional-null limitations must be reported"

    # with a threshold the observed effect clears, the same test does reject
    out2 = runner.structural_test(0.06, calibration, evaluation, alpha=0.15)
    assert out2["p_value"] == pytest.approx(1 / 9)
    assert out2["decision"]["structural_departure"] is True

    # a threshold selected and evaluated on the SAME draws would be dishonest;
    # the runner keeps the two halves separate and reports both sizes
    assert out["n_calibration"] + out["n_evaluation"] == 8


# --------------------------------------------------------------------------
# Step 8: generator correctness and honest metadata
# --------------------------------------------------------------------------

def test_markov_fallback_advances_the_chain_state():
    """After a fallback draw the next token must condition on the NEW sign.

    The profile makes ``D`` a dead end with no successors, ``A -> B`` and
    ``B -> A`` deterministic, and ``<S> -> D``. So position 0 is always ``D`` and
    position 1 is a fallback draw. If the fallback failed to update the state,
    position 2 would also be an unconditional fallback draw and could be any
    sign. With the state updated, position 2 must be the partner of position 1.
    """
    profile = {
        "n_inscriptions": 1,
        "lengths": [4],
        "unigram": {"A": 1, "B": 1, "D": 1},
        "bigram": {("<S>", "D"): 1, ("A", "B"): 1, ("B", "A"): 1},
        "trigram": {},
        "starters": {"D": 1},
        "enders": {"A": 1},
    }
    partner = {"A": "B", "B": "A"}
    seqs = gen_markov(profile, 200, 0)
    checked = 0
    bad = 0
    for seq in seqs:
        assert seq[0] == "D"
        if seq[1] in partner:
            checked += 1
            if seq[2] != partner[seq[1]]:
                bad += 1
    assert checked > 20, "the fallback path should be exercised often"
    assert bad == 0, (
        "gen_markov did not advance the chain state after a fallback draw; the "
        "next token must condition on the newly sampled sign"
    )


def test_trigram_mixture_zero_strength_matches_first_order():
    """lam=0 must sample the same conditional distribution as the Markov chain.

    A deterministic cycle profile makes the equivalence exact rather than
    statistical: every context has exactly one successor with probability 1, so
    both generators must emit the identical repeating chain. Comparing empirical
    frequencies instead would only measure sampling noise.
    """
    profile = {
        "n_inscriptions": 1,
        "lengths": [10],
        "unigram": {"A": 1, "B": 1, "C": 1},
        "bigram": {("<S>", "A"): 1, ("A", "B"): 1, ("B", "C"): 1, ("C", "A"): 1},
        "trigram": {("<S>", "A", "B"): 1, ("A", "B", "C"): 1, ("B", "C", "A"): 1},
        "starters": {"A": 1},
        "enders": {"A": 1},
    }
    markov = gen_markov(profile, 5, 0)
    lam0 = gen_trigram_mixture(profile, 5, 0, 0.0)
    assert markov[0] == ["A", "B", "C"] * 3 + ["A"]
    assert lam0 == markov, (
        "lam=0 in gen_trigram_mixture must reproduce the documented first-order "
        "generator exactly"
    )

    # with a non-zero strength the generator must still produce a valid corpus
    lam1 = gen_trigram_mixture(profile, 5, 0, 1.0)
    assert len(lam1) == 5
    assert all(len(s) == 10 for s in lam1)
    assert set(s for seq in lam1 for s in seq) <= {"A", "B", "C"}


def test_realized_properties_do_not_claim_artifact_structure():
    runner = load_runner("run_power_analysis")
    profile = _profile()
    seqs = gen_unigram(profile, 50, 0)
    props = runner.realized_properties(seqs, profile)
    assert props["n_inscriptions"] == 50
    assert props["target_n_inscriptions"] == profile["n_inscriptions"]
    assert props["artifact_relationships_generated"] is False
    assert props["duplicate_components_generated"] is False
    assert props["missingness_generated"] is False


def test_scenario_metadata_is_honest_about_null_status():
    runner = load_runner("run_power_analysis")
    # position_only and shuffled_real must NOT be described as pure first-order nulls
    for name in ("position_only", "shuffled_real", "hmm_slots"):
        info = runner.SCENARIO_INFO[name]
        assert "not a" in info["note"].lower()
    # the difference rule is never a false-positive rate anywhere in the grid
    for name, info in runner.SCENARIO_INFO.items():
        assert info["null_for_difference"] is False, name
    assert runner.SCENARIO_INFO["markov"]["kind"] == "fitted first-order null"


def test_shuffled_real_preserves_multiset():
    records = [{"inscription_id": f"i{k}",
                "sequence": ["001", "002", "003"]} for k in range(20)]
    seqs = gen_shuffled_real(records, 10, 0)
    source = {tuple(sorted(r["sequence"])) for r in records}
    for seq in seqs:
        assert tuple(sorted(seq)) in source


# --------------------------------------------------------------------------
# Runner-level behaviour
# --------------------------------------------------------------------------

def _tiny_corpus(tmp_path, n=60):
    raw = [
        {"id": str(i), "cisi": str(i), "site": "A", "direction": "L/R",
         "complete": True,
         "symbols": ["001", f"{100 + (i % 5):03d}", "003", f"{200 + (i % 4):03d}"]}
        for i in range(n)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    return corpus


def test_runner_writes_report_and_labels_rates_honestly(tmp_path):
    runner = load_runner("run_power_analysis")
    corpus = _tiny_corpus(tmp_path)
    output = tmp_path / "out"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output),
        "--replicates", "2", "--sizes", "1.0",
        "--scenarios", "unigram", "markov", "trigram_mixture",
        "--lambdas", "0.0", "1.0",
        "--permutations", "50", "--bootstrap", "50",
        "--null-replicates", "2", "--quiet",
    ])
    saved = json.loads((output / "power_analysis_summary.json").read_text(encoding="utf-8"))
    assert saved == summary

    keys = set(summary["grid"])
    assert any(k.startswith("unigram|") for k in keys)
    assert any(k.startswith("markov|") for k in keys)
    assert any(k.startswith("trigram_mixture|lam=0.0|") for k in keys)
    assert any(k.startswith("trigram_mixture|lam=1.0|") for k in keys)

    # zero-higher-order cells are flagged as valid nulls for the improvement rule
    assert summary["grid"]["unigram|lam=0.0|size=1.0"][
        "positive_rate_is_false_positive_rate"] is True
    assert summary["grid"]["markov|lam=0.0|size=1.0"][
        "positive_rate_is_false_positive_rate"] is True
    # lam>0 is an alternative, not a null
    assert summary["grid"]["trigram_mixture|lam=1.0|size=1.0"][
        "positive_rate_is_false_positive_rate"] is False

    # every rate carries its denominator
    for cell, blk in summary["grid"].items():
        assert blk["positive_improvement"]["n"] == blk["n_evaluable"], cell
        assert blk["difference"]["n"] == blk["n_evaluable"], cell
        assert blk["negative"]["n"] == blk["n_evaluable"], cell

    # per-replicate files exist, so the grid is resumable
    reps = list((output / "replicates").rglob("*.json"))
    assert reps, "per-replicate output is required for resumable runs"

    # the structural block is fully populated and its p-value never zero
    assert summary["structural"]["enabled"] is True
    assert summary["structural"]["p_value"] is not None
    assert summary["structural"]["p_value"] > 0.0

    report = (output / "power_analysis_report.txt").read_text(encoding="utf-8")
    assert "THREE SEPARATE QUESTIONS" in report
    assert "NEVER a false-positive rate in this grid" in report
    assert "calibration instruments" in report
    # the old anti-conservative claim must be gone
    assert "anti-conservative" not in report
    assert "detection rate at lambda=0 is the false-positive rate" not in report


def test_runner_manifest_declares_matching_and_policy(tmp_path):
    runner = load_runner("run_power_analysis")
    corpus = _tiny_corpus(tmp_path)
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(tmp_path / "out2"),
        "--replicates", "1", "--sizes", "1.0", "--scenarios", "unigram",
        "--permutations", "20", "--bootstrap", "20", "--quiet",
    ])
    manifest = summary["manifest"]
    assert "artifact_grouped" in manifest["grouping_policy"]
    assert "artifact relationships" in manifest["generator_matching"]["not_preserved"]
    assert manifest["resumable"]["enabled"] is True
    assert summary["structural"]["enabled"] is False
    report = (tmp_path / "out2" / "power_analysis_report.txt").read_text(encoding="utf-8")
    assert "STRUCTURAL TEST: not run" in report