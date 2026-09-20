import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from arthanvesana.data.parse import to_tidy
from arthanvesana.replicate.compact import (
    DiscreteHMM,
    crossfit_compact,
    relative_position_logprob,
)

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seqs():
    """Distinct sequences so that exact-sequence connectivity yields many groups."""
    seqs = []
    for i in range(15):
        seqs.append(["001", f"{200 + i:03d}", "003"])
        seqs.append(["002", f"{300 + i:03d}", "001"])
    return seqs


def test_hmm_logprob_is_normalized_and_deterministic():
    seqs = _seqs()
    a = DiscreteHMM(n_states=2, n_iter=3, seed=0).fit(seqs)
    b = DiscreteHMM(n_states=2, n_iter=3, seed=0).fit(seqs)
    assert a.logprob(["001", "002", "003"]) == b.logprob(["001", "002", "003"])
    # a log-probability is negative for any sequence of length >= 1
    assert a.logprob(["001", "002"]) < 0
    # unseen signs are refused rather than silently scored
    assert a.logprob(["001", "zzz"]) is None


def test_hmm_probabilities_are_valid():
    model = DiscreteHMM(n_states=3, n_iter=5, seed=1).fit(_seqs())
    assert np.allclose(np.exp(model.log_pi).sum(), 1.0, atol=1e-6)
    assert np.allclose(np.exp(model.log_a).sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.exp(model.log_b).sum(axis=1), 1.0, atol=1e-6)
    assert model.n_parameters > 0


def test_hmm_state_count_is_capped_by_vocabulary():
    model = DiscreteHMM(n_states=10, n_iter=2, seed=0).fit([["a", "b"]])
    assert model.log_pi.size == 2


def test_viterbi_path_length_matches_sequence():
    model = DiscreteHMM(n_states=3, n_iter=4, seed=2).fit(_seqs())
    path = model.viterbi_states(["001", "002", "003"])
    assert len(path) == 3
    assert all(0 <= s < 3 for s in path)


def test_relative_position_model_scores_seen_and_unseen():
    logprob, stats = relative_position_logprob(_seqs())
    assert logprob(["001", "002", "003"]) is not None
    assert logprob(["001", "002"]) is not None
    assert logprob(["zzz"]) is None          # unseen sign refused
    assert stats["n_parameters"] > 0
    # probabilities never exceed one: a 3-token sequence cannot beat 0 bits
    assert logprob(["001", "002", "003"]) <= 0


def test_crossfit_compact_reports_all_models():
    res = crossfit_compact(_seqs(), n_folds=5, seed=0, state_counts=(2, 3))
    for name in ("bigram", "relative_position", "hmm"):
        assert name in res
        assert res[name]["bits_per_token"] is not None
        assert res[name]["bits_per_token"] > 0
        assert res[name]["perplexity"] == pytest.approx(
            2 ** res[name]["bits_per_token"])
    assert res["n_groups"] > 0
    # nested state-count selection produces one choice per outer fold
    assert len(res["selected_hmm_states"]) == 5
    assert set(res["selected_hmm_states"]) <= {2, 3}


def test_crossfit_compact_deterministic():
    a = crossfit_compact(_seqs(), n_folds=3, seed=5, state_counts=(2,))
    b = crossfit_compact(_seqs(), n_folds=3, seed=5, state_counts=(2,))
    assert a == b


def test_compact_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_compact_models")
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A", "direction": "L/R",
            "complete": True, "symbols": ["001", f"{100 + (i % 3):03d}", "003"],
        }
        for i in range(30)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    output = tmp_path / "out"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output), "--repeats", "1",
    ])
    saved = json.loads(
        (output / "compact_models_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["aggregate"]["bigram"]["n_runs"] == 1
    assert summary["aggregate"]["hmm"]["bits_per_token_mean"] is not None
    assert summary["aggregate"]["selected_hmm_states"]
    report = (output / "compact_models_report.txt").read_text(encoding="utf-8")
    assert "NOT"  in report
    assert "confidence intervals" in report
    assert "economical latent-state representation" in report
    assert math.isfinite(summary["aggregate"]["relative_position"]["perplexity_mean"])